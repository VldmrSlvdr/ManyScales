"""Coauthor replicate: the semantic-fidelity cosine hierarchy and its checks.

This module writes the confirmatory cosine file. The study question, whether
``delta_z`` shifts the rating gap, preference, and the LLM guess, is
``python -m h4.survey.delta_pipeline``. The Wasserstein-1 and Wasserstein-2
comparison is ``python -m h4.survey.wasserstein_models``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from h4.config import SENTENCE_FIELDS, STAGE_ORDER, Config, load_config
from h4.embedders import build_embedder
from h4.embedders.base import Embedder
from h4.errors import SchemaError
from h4.io import (
    align_field,
    file_sha256,
    read_embeddings,
    read_item_table,
    write_embeddings,
    write_json,
    write_table,
)
from h4.metadata import collect_metadata
from h4.scoring import iter_specifications, score_items
from h4.statistics.bootstrap import hierarchical_bootstrap
from h4.statistics.mixed import fit_hierarchical
from h4.statistics.paired import wide_delta
from h4.statistics.specification_curve import specification_curve
from h4.validation import (
    build_validation_pairs,
    load_diagnostic_pairs,
    score_control_pairs,
    summarize_controls,
)
from h4.validation.lexical_overlap import jaccard

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the ManyScales H4 semantic-fidelity pipeline."
    )
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--through", default="figures", choices=STAGE_ORDER)
    parser.add_argument("--results", default=None)
    parser.add_argument("--processed", default=None)
    args = parser.parse_args(argv)
    run(
        args.config,
        through=args.through,
        results=args.results,
        processed=args.processed,
    )


def run(
    config_path: str | Path,
    *,
    through: str = "figures",
    results: str | Path | None = None,
    processed: str | Path | None = None,
) -> Config:
    _configure_logging()
    if through not in STAGE_ORDER:
        raise SchemaError(
            "The main pipeline stages are "
            + ", ".join(STAGE_ORDER)
            + ". Exploratory analyses have their own entry point."
        )
    config = load_config(config_path)
    overrides = {}
    if results is not None:
        overrides["results"] = results
    if processed is not None:
        overrides["processed"] = processed
    if overrides:
        config = config.with_paths(**overrides)
    if not config.preregistered:
        logger.warning(
            "%s is a demonstration configuration. Its estimate is not the preregistered H4 result.",
            config.study_id,
        )
    items = read_item_table(
        config.paths["items"],
        preanalysis_dir=config.paths["processed"].parent,
    )
    processed_path = config.paths["processed"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(processed_path, index=False)
    logger.info("Validated %s items and wrote %s", len(items), processed_path)
    if not _reached(through, "embed"):
        return config

    embedders = _embed(config, items)
    if not _reached(through, "similarity"):
        return config

    long_df = _similarities(config, items)
    if not _reached(through, "metric_validation"):
        return config

    _metric_validation(config, items, embedders)
    if not _reached(through, "confirmatory"):
        return config

    wide, dropped = wide_delta(long_df)
    if dropped:
        raise SchemaError(f"{dropped} similarity rows had no paired translation and were not analyzed.")
    _confirmatory(config, items, wide)
    if not _reached(through, "robustness"):
        return config

    _robustness(config, items, wide)
    if not _reached(through, "figures"):
        return config
    from h4.plotting import make_figures

    written = make_figures(config.paths["results"])
    logger.info("Wrote %s figures", len(written))
    return config


def _reached(through: str, stage: str) -> bool:
    return STAGE_ORDER.index(through) >= STAGE_ORDER.index(stage)


def _embed(config: Config, items: pd.DataFrame) -> dict[str, Embedder]:
    results = config.paths["results"] / "sentence_embeddings"
    results.mkdir(parents=True, exist_ok=True)
    items_hash = file_sha256(config.paths["items"])
    embedders: dict[str, Embedder] = {}
    for spec in config.models:
        embedder = build_embedder(
            spec,
            device=str(config.runtime.get("device", "cpu")),
            batch_size=int(config.runtime.get("batch_size", 32)),
        )
        embedders[spec.name] = embedder
        cache = results / f"{spec.slug}.parquet"
        meta_path = results / f"{spec.slug}.meta.json"
        expected = _embedding_meta(config, spec, embedder, items_hash)
        if cache.exists() and meta_path.exists() and _meta_matches(meta_path, expected):
            logger.info("Reusing sentence embeddings for %s", spec.name)
            continue
        frame = _encode_fields(embedder, items)
        write_embeddings(frame, cache)
        write_json(collect_metadata(config.root, **expected), meta_path)
        logger.info("Encoded %s with %s", spec.name, embedder.pooling)
    return embedders


def _encode_fields(embedder: Embedder, items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text_field in SENTENCE_FIELDS:
        matrix = embedder.encode(items[text_field].astype(str).tolist())
        for position, record in enumerate(items.itertuples(index=False)):
            rows.append(
                {
                    "scale_id": str(record.scale_id),
                    "item_id": str(record.item_id),
                    "text_field": text_field,
                    "embedding_model": embedder.name,
                    "embedding": matrix[position],
                }
            )
    return pd.DataFrame(rows)


def _embedding_meta(config: Config, spec, embedder: Embedder, items_hash: str | None) -> dict:
    return {
        "model_name": spec.name,
        "model_revision": embedder.revision,
        "pooling_strategy": embedder.pooling,
        "normalized": spec.normalize,
        "prompt_policy": spec.prompt_policy,
        "instruction": spec.instruction,
        "dtype": config.runtime.get("dtype", "float32"),
        "items_sha256": items_hash,
        "text_normalization": "nfkc_strip_collapse_whitespace",
        "random_seed": config.seed,
    }


def _meta_matches(path: Path, expected: dict) -> bool:
    from h4.io import read_json

    current = read_json(path)
    return all(current.get(key) == value for key, value in expected.items())


def _similarities(config: Config, items: pd.DataFrame) -> pd.DataFrame:
    frames = []
    cache_root = config.paths["results"] / "sentence_embeddings"
    loaded: dict[str, pd.DataFrame] = {}
    for model, comparison, metric, tier in iter_specifications(config):
        if model.name not in loaded:
            loaded[model.name] = read_embeddings(cache_root / f"{model.slug}.parquet")
        fields = config.comparison_fields[comparison]
        vectors = {
            name: align_field(loaded[model.name], items, name)
            for name in fields.values()
        }
        frames.append(
            score_items(
                items,
                vectors,
                model_name=model.name,
                short_name=model.short_name,
                metric_name=metric.name,
                independent_geometry=metric.independent_geometry,
                comparison=comparison,
                fields=fields,
                analysis_tier=tier,
            )
        )
    long_df = pd.concat(frames, ignore_index=True)
    destination = config.paths["results"] / "similarities"
    destination.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(destination / "similarity_long.parquet", index=False)
    write_json(
        collect_metadata(
            config.root,
            analysis_stage="similarity",
            n_rows=int(len(long_df)),
            seed=config.seed,
            study_id=config.study_id,
            note="Similarity rows are measurements. The hierarchical H4 estimate is a later stage.",
        ),
        destination / "similarity_metadata.json",
    )
    logger.info("Wrote %s similarity rows", len(long_df))
    return long_df


def _metric_validation(config: Config, items: pd.DataFrame, embedders: dict[str, Embedder]) -> None:
    flags = config.validation_flags
    pairs = build_validation_pairs(
        items,
        config.seed,
        include_identical=flags["identical_pairs"],
        include_within=flags["within_scale_permutation"],
        include_between=flags["between_scale_permutation"],
        include_negation=flags["negation_test"],
    )
    diagnostic_path = config.paths.get("diagnostic_pairs")
    diagnostic = load_diagnostic_pairs(diagnostic_path)
    if not diagnostic.empty:
        pairs = pd.concat([pairs, diagnostic], ignore_index=True)
    scored_frames = []
    summaries = []
    for spec in config.models:
        scored = score_control_pairs(pairs, embedders[spec.name], config.primary_metric.name)
        scored_frames.append(scored)
        summaries.append(summarize_controls(scored))
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    destination = config.paths["results"] / "validation"
    write_table(scored_all, destination / "control_pairs.csv")
    write_table(summary, destination / "metric_validation.csv")
    write_json(
        collect_metadata(
            config.root,
            stage="metric_validation",
            seed=config.seed,
            study_id=config.study_id,
            model_selection="frozen",
            action_on_failure="report_only",
            note="Control scoring does not choose the confirmatory embedder and does not read translation labels.",
        ),
        destination / "metric_validation_metadata.json",
    )
    logger.info("Wrote metric-validation controls to %s", destination)


def _confirmatory(config: Config, items: pd.DataFrame, wide: pd.DataFrame) -> dict:
    primary = wide.loc[wide["analysis_tier"] == "confirmatory"].copy()
    if primary.empty:
        raise SchemaError("The similarity table has no confirmatory rows.")
    if set(primary["metric"]) != {config.primary_metric.name}:
        raise SchemaError("Confirmatory rows must use the preregistered metric only.")
    if set(primary["comparison"]) != {"back_translation"}:
        raise SchemaError("Confirmatory rows must use the back-translation comparison only.")
    if set(primary["embedding_model"]) != {config.primary.name}:
        raise SchemaError("Confirmatory rows must use the preregistered embedder only.")
    fitted = fit_hierarchical(
        primary,
        confidence=config.confidence_level,
        engine=config.engine,
        seed=config.seed,
    )
    scale_table = _with_scale_names(fitted.pop("scale_table"), items)
    scale_table["analysis_tier"] = "confirmatory"
    scale_table["embedding_model"] = config.primary.name
    scale_table["metric"] = config.primary_metric.name
    scale_table["comparison"] = "back_translation"
    row = {
        "analysis_tier": "confirmatory",
        "embedding_model": config.primary.name,
        "metric": config.primary_metric.name,
        "comparison": "back_translation",
        "estimand": "human_minus_llm",
        "expected_direction": config.expected_direction,
        "seed": config.seed,
        **{key: value for key, value in fitted.items() if key != "design_matrix_columns"},
        "design_matrix_columns": fitted["design_matrix_columns"],
        "decision_rule": "H4 is evaluated only on this preregistered specification.",
    }
    tables = config.paths["results"] / "tables"
    models = config.paths["results"] / "models"
    write_table(pd.DataFrame([row]), tables / "confirmatory_estimate.csv")
    write_table(scale_table, tables / "scale_estimates.csv")
    write_json(
        collect_metadata(
            config.root,
            **{key: value for key, value in row.items() if key != "convergence_warning"},
            convergence_warning=row.get("convergence_warning", ""),
            items_sha256=file_sha256(config.paths["items"]),
            input_rows=int(len(primary)),
        ),
        models / "confirmatory_metadata.json",
    )
    logger.info(
        "Confirmatory hierarchical estimate %.4f (%.4f, %.4f)",
        row["estimate"],
        row["ci_low"],
        row["ci_high"],
    )
    return row


def _robustness(config: Config, items: pd.DataFrame, wide: pd.DataFrame) -> None:
    primary = wide.loc[wide["analysis_tier"] == "confirmatory"]
    boot = hierarchical_bootstrap(
        primary,
        iterations=config.bootstrap_iterations,
        seed=config.seed,
        confidence=config.confidence_level,
    )
    draws = boot.pop("draws")
    models = config.paths["results"] / "models"
    tables = config.paths["results"] / "tables"
    models.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(models / "bootstrap_draws.npy", draws)
    summary = pd.DataFrame([{**boot, "analysis_tier": "robustness"}])
    write_table(summary, tables / "bootstrap_summary.csv")
    write_json(
        collect_metadata(
            config.root,
            analysis_tier="robustness",
            **boot,
            study_id=config.study_id,
        ),
        models / "bootstrap_metadata.json",
    )
    curve, curve_summary = specification_curve(
        wide,
        confidence=config.confidence_level,
        engine=config.engine,
        seed=config.seed,
    )
    write_table(curve, tables / "specification_curve.csv")
    sensitivity = curve.loc[~curve["independent_geometry"].astype(bool)].copy()
    write_table(sensitivity, tables / "sensitivity_transforms.csv")
    write_json(
        collect_metadata(config.root, analysis_tier="robustness", **curve_summary, seed=config.seed),
        models / "specification_summary.json",
    )
    write_table(_lexical_table(items), tables / "lexical_overlap.csv")
    _direct_contrast(config, items, wide)
    _beta_contrasts(config, wide)
    logger.info(
        "Robustness sign consistency across cosine specifications: %.2f",
        curve_summary["sign_consistency"],
    )


def _beta_contrasts(config: Config, wide: pd.DataFrame) -> None:
    """Beta regression of the paired cosine scores for the primary embedder.

    Each comparison is fit separately. The result is a robustness check
    for a bounded, non-Gaussian similarity, and it is not the confirmatory
    H4 estimate.
    """

    from h4.statistics.beta import fit_beta_contrast

    primary = wide.loc[
        (wide["metric"] == config.primary_metric.name)
        & (wide["embedding_model"] == config.primary.name)
    ]
    rows = []
    for comparison, group in primary.groupby("comparison", sort=False):
        fitted = fit_beta_contrast(
            group,
            confidence=config.confidence_level,
            iterations=config.bootstrap_iterations,
            seed=config.seed,
        )
        rows.append(
            {
                "analysis_tier": "robustness",
                "embedding_model": config.primary.name,
                "metric": config.primary_metric.name,
                "comparison": comparison,
                "estimand": "beta_mean_human - beta_mean_llm",
                "seed": config.seed,
                **fitted,
                "decision_rule": (
                    "Beta regression is a robustness check for bounded cosine "
                    "scores. It does not replace the preregistered mixed model."
                ),
            }
        )
        logger.info(
            "Beta %s mean difference %.4f (%.4f, %.4f), cluster p %.4g",
            comparison,
            rows[-1]["mean_difference"],
            rows[-1]["ci_low"],
            rows[-1]["ci_high"],
            rows[-1]["p_two_sided_cluster"],
        )
    if rows:
        write_table(pd.DataFrame(rows), config.paths["results"] / "tables" / "beta_estimate.csv")


def _direct_contrast(config: Config, items: pd.DataFrame, wide: pd.DataFrame) -> None:
    """Hierarchical test of original English against the two Japanese texts.

    Delta is cosine(original, human Japanese) minus cosine(original, LLM
    Japanese). This is the direct comparison. It does not replace the
    preregistered back-translation estimate.
    """

    direct = wide.loc[
        (wide["comparison"] == "direct_crosslingual")
        & (wide["metric"] == config.primary_metric.name)
        & (wide["embedding_model"] == config.primary.name)
    ].copy()
    if direct.empty:
        return
    fitted = fit_hierarchical(
        direct,
        confidence=config.confidence_level,
        engine=config.engine,
        seed=config.seed,
    )
    scale_table = _with_scale_names(fitted.pop("scale_table"), items)
    scale_table["analysis_tier"] = "robustness"
    scale_table["embedding_model"] = config.primary.name
    scale_table["metric"] = config.primary_metric.name
    scale_table["comparison"] = "direct_crosslingual"
    from scipy import stats

    z_value = fitted["estimate"] / fitted["se"]
    row = {
        "analysis_tier": "robustness",
        "embedding_model": config.primary.name,
        "metric": config.primary_metric.name,
        "comparison": "direct_crosslingual",
        "estimand": "human_minus_llm",
        "contrast": "cosine(original_en, human_ja) - cosine(original_en, llm_ja)",
        "expected_direction": config.expected_direction,
        "seed": config.seed,
        **fitted,
        "mean_similarity_human": float(direct["sim_human"].mean()),
        "mean_similarity_llm": float(direct["sim_llm"].mean()),
        "z": float(z_value),
        "p_two_sided": float(2 * stats.norm.sf(abs(z_value))),
        "decision_rule": (
            "Two-sided test of a nonzero human-minus-LLM difference on the "
            "direct original-versus-Japanese comparison."
        ),
    }
    tables = config.paths["results"] / "tables"
    write_table(pd.DataFrame([row]), tables / "direct_estimate.csv")
    write_table(scale_table, tables / "direct_scale_estimates.csv")
    logger.info(
        "Direct hierarchical estimate %.4f (%.4f, %.4f), two-sided p %.4g",
        row["estimate"],
        row["ci_low"],
        row["ci_high"],
        row["p_two_sided"],
    )


def _with_scale_names(table: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    named = table.copy()
    named["scale_id"] = named["scale_id"].map(_scale_key)
    if "scale_name" not in items.columns:
        named["scale_name"] = named["scale_id"]
        return named
    names = items.loc[:, ["scale_id", "scale_name"]].copy()
    names["scale_id"] = names["scale_id"].map(_scale_key)
    names["scale_name"] = names["scale_name"].astype(str).str.strip()
    names = names.loc[names["scale_name"].ne("")].drop_duplicates("scale_id")
    named = named.merge(names, on="scale_id", how="left")
    named["scale_name"] = named["scale_name"].fillna(named["scale_id"])
    return named


def _scale_key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


def _lexical_table(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in items.itertuples(index=False):
        rows.append(
            {
                "scale_id": str(record.scale_id),
                "item_id": str(record.item_id),
                "jaccard_human_back": jaccard(record.original_en, record.human_back_en),
                "jaccard_llm_back": jaccard(record.original_en, record.llm_back_en),
                "analysis_tier": "robustness",
                "role": "lexical_description",
            }
        )
    return pd.DataFrame(rows)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    main()
