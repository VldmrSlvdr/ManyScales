"""Same judgment models, every stored origin-versus-back-translation measure.

A positive predictor means the human back-translation is closer to the
original than the LLM back-translation. Cosine keeps that sign. Distance
measures are reversed so the coefficient is comparable. These fits do not
enter the confirmatory cosine model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h4.survey.mixed_judgments import (
    _fit,
    _key,
    _specifications,
    _z,
    choice_frame,
    item_flag_frame,
    rating_gap_frame,
)

_MODELS = {
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "labse": "sentence-transformers/LaBSE",
    "e5": "intfloat/multilingual-e5-large-instruct",
}
_SIMILARITY_TERMS = {
    ("naturalness_gap", "delta_z"),
    ("naturalness_level", "delta_z"),
    ("naturalness_version", "delta_z"),
    ("naturalness_version", "version_human:delta_z"),
    ("expert_semantic_gap", "delta_z"),
    ("expert_cultural_validity_gap", "delta_z"),
    ("expert_choice_fit_gap", "delta_z"),
    ("expert_naturalness_gap", "delta_z"),
    ("preference", "delta_z"),
    ("guess", "delta_z"),
    ("guess_abs_delta", "abs_delta_z"),
    ("item_flag_cultural", "delta_z"),
    ("item_flag_cultural", "version_human:delta_z"),
    ("item_flag_semantic", "delta_z"),
    ("item_flag_semantic", "version_human:delta_z"),
    ("item_flag_understandability", "delta_z"),
    ("item_flag_understandability", "version_human:delta_z"),
}


def instruments(root: Path) -> pd.DataFrame:
    """One human-closer gap per item and instrument. Positive means human is closer."""

    rows = []
    similarity = pd.read_parquet(root / "results/similarities/similarity_long.parquet")
    similarity = similarity.loc[similarity["comparison"].eq("back_translation")].copy()
    for short_name, model in _MODELS.items():
        for metric in ("cosine", "angular_distance", "euclidean_normalized"):
            part = similarity.loc[similarity["embedding_model"].eq(model) & similarity["metric"].eq(metric)]
            wide = _wide_similarity(part)
            signed = wide["human"] - wide["llm"]
            family = "sentence_cosine" if metric == "cosine" else "sentence_distance"
            rows.append(_pack(wide, signed, f"{short_name} {metric.replace('_', ' ')}", family, short_name))
    tokens = pd.read_csv(root / "results/exploratory/pooling/token_distance_pairs.csv")
    tokens = tokens.loc[tokens["comparison"].eq("back_translation")]
    for short_name in _MODELS:
        part = tokens.loc[tokens["short_name"].eq(short_name)].copy()
        part["scale_id"] = part["scale_id"].map(_key)
        part["item_id"] = part["item_id"].map(lambda value: str(value).zfill(2))
        for column, family, label in (
            ("delta_d_mu", "token_mean", "token mean distance"),
            ("delta_d_sigma", "token_spread", "token spread"),
        ):
            signed = -pd.to_numeric(part[column], errors="coerce")
            rows.append(_pack(part, signed, f"{short_name} {label}", family, short_name))
    water = pd.read_csv(root / "results/exploratory/geometry/wasserstein.csv")
    water = water.loc[water["comparison"].eq("back_translation")]
    for short_name in _MODELS:
        part = water.loc[water["short_name"].eq(short_name)].copy()
        part["scale_id"] = part["scale_id"].map(_key)
        part["item_id"] = part["item_id"].map(lambda value: str(value).zfill(2))
        signed = -pd.to_numeric(part["delta_distance"], errors="coerce")
        rows.append(_pack(part, signed, f"{short_name} Wasserstein", "wasserstein", short_name))
    return pd.concat(rows, ignore_index=True)


def run(root: Path) -> pd.DataFrame:
    """Refit the judgment models under each instrument and write the appendix."""

    out = root / "results" / "sensitivity"
    fits = out / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    table = instruments(root)
    table.to_parquet(out / "item_gaps.parquet", index=False)
    seed_items, seed_scales = _summarize(table.loc[table["instrument"].eq("mpnet cosine")])
    ratings_base = rating_gap_frame(root, seed_scales)
    choices_base = choice_frame(root, seed_scales)
    flags_base = item_flag_frame(root, seed_items)
    coef_rows = []
    status_rows = []
    for instrument, part in table.groupby("instrument", sort=False):
        items, scales = _summarize(part)
        ratings = _swap_scale(ratings_base, scales)
        choices = _swap_scale(choices_base, scales)
        flags = _swap_item(flags_base, items)
        if ratings["delta_z"].isna().any() or flags["delta_z"].isna().any():
            raise RuntimeError(f"{instrument} did not match the survey items.")
        for spec in _specifications(ratings, choices, flags):
            prefix = fits / f"{_slug(str(instrument))}__{spec['name']}"
            coef, _var, status = _fit(spec["frame"], spec["formula"], spec["family"], prefix, spec["nagq"])
            prefix.with_suffix(".csv").unlink(missing_ok=True)
            coef.insert(0, "instrument", instrument)
            coef.insert(1, "family", part["family"].iloc[0])
            coef.insert(2, "embedding", part["embedding"].iloc[0])
            status.insert(0, "instrument", instrument)
            status.insert(1, "model", spec["name"])
            coef.insert(3, "model", spec["name"])
            coef_rows.append(coef)
            status_rows.append(status)
            print(f"{instrument} {spec['name']}", flush=True)
    coefficients = pd.concat(coef_rows, ignore_index=True)
    status = pd.concat(status_rows, ignore_index=True)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    status.to_csv(out / "model_status.csv", index=False)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _figure_slopes(coefficients, figures / "cosine_embeddings.png", ["sentence_cosine"])
    _figure_families(
        coefficients,
        figures / "other_back_translation.png",
        ["token_mean", "token_spread", "wasserstein"],
    )
    report = _report(coefficients, table)
    path = root / "protocl" / "h4_sensitivity_appendix.md"
    path.write_text(report)
    _point_effects_report(root / "protocl" / "h4_effects_report.md")
    return coefficients


def _pack(frame: pd.DataFrame, signed: pd.Series, instrument: str, family: str, embedding: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "scale_id": frame["scale_id"].map(_key).to_numpy(),
            "item_id": frame["item_id"].map(lambda value: str(value).zfill(2)).to_numpy(),
            "delta_back": signed.to_numpy(),
        }
    )
    out = out.dropna(subset=["delta_back"])
    out["abs_delta_back"] = out["delta_back"].abs()
    out["delta_z"] = _z(out["delta_back"])
    out["instrument"] = instrument
    out["family"] = family
    out["embedding"] = embedding
    return out


def _wide_similarity(part: pd.DataFrame) -> pd.DataFrame:
    work = part.loc[:, ["scale_id", "item_id", "translation", "similarity"]].copy()
    work["scale_id"] = work["scale_id"].map(_key)
    work["item_id"] = work["item_id"].map(lambda value: str(value).zfill(2))
    wide = work.pivot_table(index=["scale_id", "item_id"], columns="translation", values="similarity", aggfunc="first")
    return wide.reset_index()


def _summarize(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = part.loc[:, ["scale_id", "item_id", "delta_back", "delta_z"]].copy()
    scales = part.groupby("scale_id", as_index=False).agg(
        delta_back=("delta_back", "mean"),
        abs_delta_back=("abs_delta_back", "mean"),
    )
    scales["delta_z"] = _z(scales["delta_back"])
    scales["abs_delta_z"] = _z(scales["abs_delta_back"])
    return items, scales


def _swap_scale(frame: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in frame.columns if column not in {"delta_back", "delta_z", "abs_delta_back", "abs_delta_z"}]
    return frame.loc[:, keep].merge(scales, on="scale_id", how="left")


def _swap_item(frame: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in frame.columns if column not in {"delta_back", "delta_z"}]
    return frame.loc[:, keep].merge(items, on=["scale_id", "item_id"], how="left")


def _figure_families(coefficients: pd.DataFrame, path: Path, families: list[str]) -> None:
    figure, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 8.2), sharey=True)
    for axis, family in zip(axes, families):
        _draw_slopes(axis, coefficients, family, legend=family == families[-1])
        axis.set_title(family.replace("_", " "))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_slopes(coefficients: pd.DataFrame, path: Path, families: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 8.2))
    _draw_slopes(axis, coefficients, families[0], legend=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_slopes(axis, coefficients: pd.DataFrame, family: str, legend: bool) -> None:
    terms = coefficients.loc[coefficients.apply(lambda row: (row["model"], row["term"]) in _SIMILARITY_TERMS, axis=1)].copy()
    terms = terms.loc[terms["family"].eq(family)]
    if terms.empty:
        return
    terms["label"] = terms["model"] + ": " + terms["term"]
    labels = list(dict.fromkeys(terms["label"]))
    instruments = list(dict.fromkeys(terms["instrument"]))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    colors = {
        "mpnet": "#1B1B1B",
        "minilm": "#4C78A8",
        "labse": "#F58518",
        "e5": "#54A24B",
    }
    for index, instrument in enumerate(instruments):
        part = terms.loc[terms["instrument"].eq(instrument)]
        embedding = str(part["embedding"].iloc[0])
        y = [labels.index(label) + (index - (len(instruments) - 1) / 2) * 0.12 for label in part["label"]]
        axis.errorbar(
            part["estimate"],
            y,
            xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]],
            fmt="o",
            color=colors.get(embedding, "#333333"),
            ecolor=colors.get(embedding, "#333333"),
            markersize=3.5,
            capsize=2,
            label=instrument if family == "sentence_cosine" else embedding,
            linestyle="none",
        )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xlabel("Coefficient per SD of the human-closer gap")
    if legend:
        handles, legend_labels = axis.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)


def _report(coefficients: pd.DataFrame, table: pd.DataFrame) -> str:
    cosine = coefficients.loc[coefficients["family"].eq("sentence_cosine")]
    lines = [
        "# Appendix. Sensitivity of the judgment models to the back-translation measure",
        "",
        "The formulas are the ones in `protocl/h4_mixed_judgment.md`. Only the similarity predictor changes. Each instrument is standardized on its own scale. A positive unit means the human back-translation is one standard deviation closer to the original than the LLM back-translation. In the similarity table, cosine, angular distance, and normalized Euclidean distance are already stored so that a larger value means closer: angular distance is entered as one minus the angle, and Euclidean distance is negated. Token mean distance, token spread, and Wasserstein distance are raw distances, so those human-minus-LLM gaps are reversed before standardization.",
        "",
        "The confirmatory cosine estimate is unchanged. MPNet cosine remains the primary predictor. The other rows ask whether a coefficient that uses this predictor keeps its sign and its separation from zero.",
        "",
        f"Instruments: {table['instrument'].nunique()}. Items per instrument: {int(table.groupby('instrument').size().median())}.",
        "",
        "## What persists across the four embeddings",
        "",
        "These four rows are independent geometries: MPNet, MiniLM, LaBSE, and E5, each scored by cosine on the English back-translation.",
        "",
        _persistence_table(cosine),
        "",
        "![Similarity slopes under four cosine embeddings](../results/sensitivity/figures/cosine_embeddings.png)",
        "",
        "## Distance transforms of the same embedding",
        "",
        "Angular distance and normalized Euclidean distance are monotone transforms of cosine for a fixed embedding. Agreement with that embedding's cosine slope is expected and is not a new replication.",
        "",
        _family_note(coefficients, "sentence_distance"),
        "",
        "## Token distances and Wasserstein distance",
        "",
        "These compare the token cloud of the original English with the token cloud of each back-translation. They are not sentence-cosine scores.",
        "",
        _family_note(coefficients, "token_mean"),
        "",
        _family_note(coefficients, "token_spread"),
        "",
        _family_note(coefficients, "wasserstein"),
        "",
        "![Token and Wasserstein slopes](../results/sensitivity/figures/other_back_translation.png)",
        "",
        "## Order and version effects",
        "",
        "Administration order and the human-versus-LLM version term do not use the embedding. Under the four cosine embeddings their intervals are:",
        "",
        _order_table(cosine),
        "",
        "Coefficients for every instrument are in `results/sensitivity/coefficients.csv`.",
        "",
    ]
    return "\n".join(lines)


def _persistence_table(cosine: pd.DataFrame) -> str:
    lines = [
        "| Effect | MPNet | MiniLM | LaBSE | E5 | Across the four embeddings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    focus = cosine.loc[cosine.apply(lambda row: (row["model"], row["term"]) in _SIMILARITY_TERMS, axis=1)]
    for (model, term), part in focus.groupby(["model", "term"], sort=False):
        cells = []
        signs = []
        separated = []
        for embedding in ("mpnet", "minilm", "labse", "e5"):
            row = part.loc[part["embedding"].eq(embedding)].iloc[0]
            cells.append(f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]")
            signs.append(np.sign(row.estimate))
            separated.append(_excludes(row.ci_low, row.ci_high))
        verdict = _verdict(signs, separated)
        lines.append(f"| {model}: {term} | " + " | ".join(cells) + f" | {verdict} |")
    return "\n".join(lines)


def _family_note(coefficients: pd.DataFrame, family: str) -> str:
    part = coefficients.loc[
        coefficients["family"].eq(family)
        & coefficients.apply(lambda row: (row["model"], row["term"]) in _SIMILARITY_TERMS, axis=1)
    ]
    if part.empty:
        return ""
    rows = []
    for (model, term), group in part.groupby(["model", "term"], sort=False):
        signs = [np.sign(value) for value in group["estimate"]]
        separated = [_excludes(low, high) for low, high in zip(group["ci_low"], group["ci_high"])]
        if any(separated):
            rows.append(f"{model} `{term}`: {_verdict(signs, separated)} ({family.replace('_', ' ')}).")
    if not rows:
        return f"No {family.replace('_', ' ')} slope has an interval that excludes zero."
    return " ".join(rows)


def _order_table(cosine: pd.DataFrame) -> str:
    terms = [
        ("naturalness_gap", "ai_first"),
        ("preference", "ai_first"),
        ("guess", "ai_first"),
        ("expert_cultural_validity_gap", "ai_first"),
        ("expert_choice_fit_gap", "ai_first"),
        ("expert_naturalness_gap", "ai_first"),
        ("item_flag_semantic", "version_human"),
        ("item_flag_cultural", "version_human"),
        ("item_flag_understandability", "version_human"),
    ]
    lines = [
        "| Effect | MPNet | MiniLM | LaBSE | E5 | Across the four embeddings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model, term in terms:
        part = cosine.loc[cosine["model"].eq(model) & cosine["term"].eq(term)]
        cells = []
        signs = []
        separated = []
        for embedding in ("mpnet", "minilm", "labse", "e5"):
            row = part.loc[part["embedding"].eq(embedding)].iloc[0]
            cells.append(f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]")
            signs.append(np.sign(row.estimate))
            separated.append(_excludes(row.ci_low, row.ci_high))
        lines.append(f"| {model}: {term} | " + " | ".join(cells) + f" | {_verdict(signs, separated)} |")
    return "\n".join(lines)


def _verdict(signs: list[float], separated: list[bool]) -> str:
    if all(separated) and len(set(signs)) == 1:
        return "Persists, same sign, every interval excludes 0"
    if not any(separated):
        return "The null persists: every interval includes 0"
    if any(separated) and len(set(signs)) == 1:
        return "Same sign; the interval excludes 0 for some embeddings"
    return "The sign or the separation does not persist"


def _point_effects_report(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    marker = "## Appendix"
    block = (
        "## Appendix\n\n"
        "The same formulas, with every stored origin-versus-back-translation measure in place of MPNet cosine, "
        "are in `protocl/h4_sensitivity_appendix.md`.\n"
    )
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + block)


def _excludes(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
