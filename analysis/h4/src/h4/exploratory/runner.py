"""Run exploratory pooling and token-distribution analyses.

Outputs stay under ``results/exploratory`` and ``results/token_embeddings``.
The confirmatory estimate is not read or rewritten.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from h4.config import SENTENCE_FIELDS, Config, load_config
from h4.embedders import build_embedder
from h4.exploratory.distributional_distance import pairwise_euclidean, wasserstein
from h4.exploratory.pooling.collapse_metric import mean_pool, socm
from h4.exploratory.pooling.paired_distances import paired_token_contrast, paired_token_summary
from h4.exploratory.pooling.token_embeddings import (
    TransformerTokenEncoder,
    save_token_matrix,
    token_filename,
)
from h4.exploratory.token_geometry import associate_disagreement, model_disagreement
from h4.io import read_item_table, write_json, write_table
from h4.metadata import collect_metadata
from h4.similarity import score_vectors
from h4.statistics.paired import wide_delta

logger = logging.getLogger(__name__)

COMPARISONS = {
    "back_translation": {"human": "human_back_en", "llm": "llm_back_en"},
    "direct_crosslingual": {"human": "human_ja", "llm": "llm_ja"},
}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exploratory representation analyses for ManyScales H4.")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--part", choices=("pooling", "geometry"), required=True)
    parser.add_argument("--results", default=None)
    args = parser.parse_args(argv)
    _configure_logging()
    config = load_config(args.config)
    if args.results:
        config = config.with_paths(results=args.results)
    if args.part == "pooling":
        run_pooling(config)
    else:
        run_geometry(config)


def run_pooling(config: Config) -> None:
    block = (config.exploratory.get("hara_pooling_analysis") or {})
    if not block.get("enabled", False):
        logger.info("Exploratory pooling analysis is disabled in the config. Nothing was estimated.")
        return
    items = read_item_table(
        config.paths["items"],
        preanalysis_dir=config.paths["processed"].parent,
    )
    results = config.paths["results"]
    pooling_dir = results / "exploratory" / "pooling"
    token_root = results / "token_embeddings"
    pair_frames = []
    pooling_frames = []
    methods = list((config.exploratory.get("alternative_pooling") or {}).get("methods") or ["mean", "cls"])
    alternative_enabled = bool((config.exploratory.get("alternative_pooling") or {}).get("enabled", False))
    for spec in config.models:
        logger.info("Pooling diagnostics for %s", spec.name)
        content, cls_map, unpadded = _token_maps(config, spec, items, token_root / spec.slug, save=bool(block.get("save_token_embeddings", True)))
        pair_frames.append(_socm_frame(items, content, spec.name, spec.short_name))
        if alternative_enabled:
            pooling_frames.append(
                _pooling_frame(items, content, cls_map, unpadded, methods, spec.name, spec.short_name)
            )
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    contrast = _socm_contrast(pairs)
    distances = paired_token_contrast(pairs)
    write_table(pairs, pooling_dir / "socm_pairs.csv")
    write_table(contrast, pooling_dir / "socm_contrast.csv")
    write_table(distances, pooling_dir / "token_distance_pairs.csv")
    write_table(paired_token_summary(distances), pooling_dir / "token_distance_summary.csv")
    if pooling_frames:
        write_table(pd.concat(pooling_frames, ignore_index=True), pooling_dir / "alternative_pooling.csv")
    write_json(
        collect_metadata(
            config.root,
            analysis_tier="exploratory",
            directional_test=False,
            reference="Hara2026",
            token_input="raw_item_text",
            note="Exploratory pooling diagnostics are not used to evaluate H4.",
            seed=config.seed,
            study_id=config.study_id,
        ),
        pooling_dir / "metadata.json",
    )
    logger.info("Wrote exploratory pooling tables to %s", pooling_dir)


def run_geometry(config: Config) -> None:
    distribution = config.exploratory.get("token_distribution") or {}
    if not distribution.get("enabled", False):
        logger.info("Exploratory token-distribution analysis is disabled in the config. Nothing was estimated.")
        return
    items = read_item_table(
        config.paths["items"],
        preanalysis_dir=config.paths["processed"].parent,
    )
    results = config.paths["results"]
    geometry_dir = results / "exploratory" / "geometry"
    frames = []
    for spec in config.models:
        logger.info("Token-distribution distances for %s", spec.name)
        token_map = _load_token_map(results / "token_embeddings" / spec.slug)
        if not token_map:
            content, _cls_map, _unpadded = _token_maps(
                config,
                spec,
                items,
                results / "token_embeddings" / spec.slug,
                save=True,
            )
            token_map = content
        frames.append(
            _wasserstein_frame(
                items,
                token_map,
                spec.name,
                spec.short_name,
                str(distribution.get("cost", "euclidean")),
            )
        )
    wasserstein_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(wasserstein_table, geometry_dir / "wasserstein.csv")
    disagreement_path = geometry_dir / "model_disagreement.csv"
    association_path = geometry_dir / "disagreement_association.json"
    similarity_path = results / "similarities" / "similarity_long.parquet"
    socm_path = results / "exploratory" / "pooling" / "socm_pairs.csv"
    if similarity_path.exists() and socm_path.exists():
        wide, _dropped = wide_delta(pd.read_parquet(similarity_path))
        cosine = wide.loc[
            (wide["metric"] == "cosine") & (wide["comparison"] == "back_translation")
        ]
        disagreement = model_disagreement(cosine)
        write_table(disagreement, disagreement_path)
        association = associate_disagreement(
            disagreement,
            pd.read_csv(socm_path),
            seed=config.seed,
        )
        write_json(association, association_path)
    else:
        logger.info(
            "Cross-model disagreement was not estimated because the similarity table or SOCM table is absent."
        )
    write_json(
        collect_metadata(
            config.root,
            analysis_tier="exploratory",
            directional_test=False,
            token_distribution_status="exploratory",
            seed=config.seed,
            study_id=config.study_id,
            note="Token-distribution distances are not a test of H4.",
        ),
        geometry_dir / "metadata.json",
    )
    logger.info("Wrote exploratory geometry tables to %s", geometry_dir)


def _token_maps(
    config: Config,
    spec,
    items: pd.DataFrame,
    directory: Path,
    *,
    save: bool,
) -> tuple[
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], np.ndarray],
]:
    content: dict[tuple[str, str, str], np.ndarray] = {}
    cls_map: dict[tuple[str, str, str], np.ndarray] = {}
    unpadded: dict[tuple[str, str, str], np.ndarray] = {}
    index_rows = []
    logger.info("Token states for %s, %s items", spec.name, len(items))
    encoder = None
    embedder = None
    if spec.family == "deterministic":
        embedder = build_embedder(spec, device="cpu", batch_size=1)
    else:
        encoder = TransformerTokenEncoder(
            spec.name,
            device=str(config.runtime.get("device", "cpu")),
            batch_size=int(config.runtime.get("batch_size", 8)),
            revision=spec.revision,
        )
    for text_field in SENTENCE_FIELDS:
        logger.info("Encoding %s / %s", spec.short_name, text_field)
        texts = items[text_field].astype(str).tolist()
        if embedder is not None:
            contents = embedder.encode_tokens(texts)
            clss = list(embedder.cls_vectors(texts))
            unpadded_rows = contents
        else:
            assert encoder is not None
            encoded = encoder.encode(texts)
            contents = [batch["content"] for batch in encoded]
            clss = [batch["cls"] for batch in encoded]
            unpadded_rows = [batch["unpadded"] for batch in encoded]
        for record, matrix, cls_vector, full in zip(
            items.itertuples(index=False),
            contents,
            clss,
            unpadded_rows,
            strict=True,
        ):
            key = (str(record.scale_id), str(record.item_id), text_field)
            content[key] = np.asarray(matrix, dtype=np.float32)
            cls_map[key] = np.asarray(cls_vector, dtype=np.float32)
            unpadded[key] = np.asarray(full, dtype=np.float32)
            if save:
                relative = token_filename(key[0], key[1], text_field)
                save_token_matrix(directory / relative, content[key])
                index_rows.append(
                    {
                        "scale_id": key[0],
                        "item_id": key[1],
                        "text_field": text_field,
                        "path": relative,
                        "n_tokens": int(len(content[key])),
                        "dimension": int(content[key].shape[1]),
                    }
                )
    if save and index_rows:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(index_rows).to_parquet(directory / "index.parquet", index=False)
    return content, cls_map, unpadded


def _load_token_map(directory: Path) -> dict[tuple[str, str, str], np.ndarray]:
    index_path = directory / "index.parquet"
    if not index_path.exists():
        return {}
    index = pd.read_parquet(index_path)
    mapping = {}
    for record in index.itertuples(index=False):
        mapping[(str(record.scale_id), str(record.item_id), str(record.text_field))] = np.load(
            directory / str(record.path)
        )
    return mapping


def _socm_frame(
    items: pd.DataFrame,
    token_map: dict[tuple[str, str, str], np.ndarray],
    model_name: str,
    short_name: str,
) -> pd.DataFrame:
    rows = []
    for comparison, fields in COMPARISONS.items():
        for condition, text_field in fields.items():
            for record in items.itertuples(index=False):
                reference = token_map[(str(record.scale_id), str(record.item_id), "original_en")]
                other = token_map[(str(record.scale_id), str(record.item_id), text_field)]
                measured = socm(reference, other)
                measured.update(
                    {
                        "scale_id": str(record.scale_id),
                        "item_id": str(record.item_id),
                        "translation": condition,
                        "comparison": comparison,
                        "embedding_model": model_name,
                        "short_name": short_name,
                        "analysis_tier": "exploratory",
                    }
                )
                rows.append(measured)
    return pd.DataFrame(rows)


def _socm_contrast(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    keys = ["scale_id", "item_id", "comparison", "embedding_model", "short_name"]
    human = pairs.loc[pairs["translation"] == "human"]
    llm = pairs.loc[pairs["translation"] == "llm"]
    merged = human.merge(llm, on=keys, suffixes=("_human", "_llm"))
    merged["delta_socm"] = merged["socm_human"] - merged["socm_llm"]
    merged["delta_d_sigma"] = merged["d_sigma_human"] - merged["d_sigma_llm"]
    merged["analysis_tier"] = "exploratory"
    merged["directional_test"] = False
    keep = keys + [
        "socm_human",
        "socm_llm",
        "delta_socm",
        "d_sigma_human",
        "d_sigma_llm",
        "delta_d_sigma",
        "cosine_human",
        "cosine_llm",
        "analysis_tier",
        "directional_test",
    ]
    return merged.loc[:, keep]


def _pooling_frame(
    items: pd.DataFrame,
    content: dict[tuple[str, str, str], np.ndarray],
    cls_map: dict[tuple[str, str, str], np.ndarray],
    unpadded: dict[tuple[str, str, str], np.ndarray],
    methods: list[str],
    model_name: str,
    short_name: str,
) -> pd.DataFrame:
    rows = []
    stores = {"mean": unpadded, "cls": cls_map, "max": content}
    for method in methods:
        source = stores[method]
        pooled = {
            key: (vector if vector.ndim == 1 else mean_pool(vector) if method == "mean" else np.max(vector, axis=0))
            for key, vector in source.items()
        }
        if method == "cls":
            pooled = {key: np.asarray(vector, dtype=np.float64) for key, vector in cls_map.items()}
        for comparison, fields in COMPARISONS.items():
            for record in items.itertuples(index=False):
                key_prefix = (str(record.scale_id), str(record.item_id))
                reference = pooled[(*key_prefix, "original_en")][None, :]
                human = pooled[(*key_prefix, fields["human"])][None, :]
                llm = pooled[(*key_prefix, fields["llm"])][None, :]
                sim_human, _raw_h = score_vectors("cosine", reference, human)
                sim_llm, _raw_l = score_vectors("cosine", reference, llm)
                rows.append(
                    {
                        "scale_id": key_prefix[0],
                        "item_id": key_prefix[1],
                        "comparison": comparison,
                        "pooling": method,
                        "embedding_model": model_name,
                        "short_name": short_name,
                        "sim_human": float(sim_human[0]),
                        "sim_llm": float(sim_llm[0]),
                        "delta": float(sim_human[0] - sim_llm[0]),
                        "analysis_tier": "exploratory",
                        "directional_test": False,
                    }
                )
    return pd.DataFrame(rows)


def _wasserstein_frame(
    items: pd.DataFrame,
    token_map: dict[tuple[str, str, str], np.ndarray],
    model_name: str,
    short_name: str,
    cost_name: str,
) -> pd.DataFrame:
    if cost_name != "euclidean":
        raise ValueError("The configured token-distribution cost is euclidean.")
    rows = []
    for comparison, fields in COMPARISONS.items():
        logger.info("Wasserstein %s / %s", short_name, comparison)
        for record in items.itertuples(index=False):
            prefix = (str(record.scale_id), str(record.item_id))
            reference = token_map[(*prefix, "original_en")]
            distances = {}
            solver = "emd"
            for condition, text_field in fields.items():
                other = token_map[(*prefix, text_field)]
                result = wasserstein(pairwise_euclidean(reference, other))
                distances[condition] = result["distance"]
                solver = result["solver"]
            rows.append(
                {
                    "scale_id": prefix[0],
                    "item_id": prefix[1],
                    "comparison": comparison,
                    "embedding_model": model_name,
                    "short_name": short_name,
                    "cost": cost_name,
                    "solver": solver,
                    "distance_human": distances["human"],
                    "distance_llm": distances["llm"],
                    "delta_distance": distances["human"] - distances["llm"],
                    "analysis_tier": "exploratory",
                    "directional_test": False,
                }
            )
    return pd.DataFrame(rows)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    main()
