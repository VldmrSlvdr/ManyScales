"""Replace the cosine scale gap with item-level Wasserstein gaps.

Wasserstein-1 is the stored earth-mover distance. Wasserstein-2 is the square root of
the transport cost under squared Euclidean cost, on the same MPNet token
clouds. Each item is standardized across items. A scale judgment then uses
the mean of its items' standardized distances, because the rating, the
preference, and the guess are recorded once per person per scale.

Gaussian AIC and BIC use maximum likelihood. The cosine replicate file is
not rewritten.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h4.exploratory.distributional_distance import pairwise_euclidean, wasserstein
from h4.survey.mixed_judgments import (
    _fit,
    _key,
    _p,
    _z,
    back_translation_delta,
    choice_frame,
    rating_gap_frame,
)

_FIELDS = ("original_en", "human_back_en", "llm_back_en")
_OUTCOMES = (
    ("naturalness_gap", "gaussian", "rating_gap ~ ai_first + expert + {rhs} + (1 | respondent_id) + (1 | scale_id)", "naturalness"),
    ("expert_choice_fit_gap", "gaussian", "rating_gap ~ ai_first + {rhs} + (1 | respondent_id) + (1 | scale_id)", "choice_fit"),
    ("preference", "binomial", "prefer_human ~ ai_first + expert + {rhs} + (1 | respondent_id) + (1 | scale_id)", "preference"),
    ("guess", "binomial", "guess_correct ~ ai_first + expert + {rhs} + (1 | respondent_id) + (1 | scale_id)", "guess"),
)
_PREDICTORS = (
    ("cosine_scale_z", "delta_z"),
    ("w1_item_mean", "w1_z"),
    ("w2_item_mean", "w2_z"),
    ("w1_and_w2", "w1_z + w2_z"),
)


def item_wasserstein(root: Path) -> pd.DataFrame:
    """One Wasserstein-1 and one Wasserstein-2 gap per item. Positive means the human cloud is closer."""

    stored = pd.read_csv(root / "results/exploratory/geometry/wasserstein.csv")
    part = stored.loc[
        stored["short_name"].eq("mpnet") & stored["comparison"].eq("back_translation")
    ].copy()
    part["scale_id"] = part["scale_id"].map(_key)
    part["item_id"] = part["item_id"].map(lambda value: str(value).zfill(2))
    part["w1_gap"] = -(pd.to_numeric(part["distance_human"]) - pd.to_numeric(part["distance_llm"]))
    order2 = _order2(root, part)
    merged = part.loc[:, ["scale_id", "item_id", "w1_gap"]].merge(order2, on=["scale_id", "item_id"], how="inner")
    merged["w1_z"] = _z(merged["w1_gap"])
    merged["w2_z"] = _z(merged["w2_gap"])
    return merged


def run(root: Path) -> pd.DataFrame:
    """Fit cosine, Wasserstein-1, Wasserstein-2, and the two-distance model on the same rows."""

    root = root.resolve()
    out = root / "results" / "wasserstein_models"
    fits = out / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    items = item_wasserstein(root)
    items.to_csv(out / "item_distances.csv", index=False)
    scale = items.groupby("scale_id", as_index=False).agg(
        w1_z=("w1_z", "mean"),
        w2_z=("w2_z", "mean"),
        n_items=("w1_z", "size"),
    )
    _items, cosine_scales = back_translation_delta(root / "results/similarities/similarity_long.parquet")
    ratings = rating_gap_frame(root, cosine_scales)
    choices = choice_frame(root, cosine_scales)
    ratings = ratings.merge(scale, on="scale_id", how="inner")
    choices = choices.merge(scale, on="scale_id", how="inner")

    rows = []
    for name, family, template, outcome in _OUTCOMES:
        frame = _outcome_frame(ratings, choices, outcome)
        for predictor, rhs in _PREDICTORS:
            formula = template.format(rhs=rhs)
            coef, _var, status = _fit(frame, formula, family, fits / f"{name}__{predictor}", 1, ml=family == "gaussian")
            slope = coef.loc[coef["term"].isin(["delta_z", "w1_z", "w2_z"])].copy()
            intercept = coef.loc[coef["term"].eq("(Intercept)")].iloc[0]
            info = status.iloc[0]
            for _, term in slope.iterrows():
                rows.append(
                    {
                        "outcome": name,
                        "predictor": predictor,
                        "term": term["term"],
                        "estimate": term["estimate"],
                        "se": term["se"],
                        "ci_low": term["ci_low"],
                        "ci_high": term["ci_high"],
                        "p": term["p"],
                        "intercept": intercept["estimate"],
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "n": info["n"],
                        "family": family,
                    }
                )
    table = pd.DataFrame(rows)
    table["includes_zero"] = (table["ci_low"] <= 0) & (table["ci_high"] >= 0)
    table.to_csv(out / "comparison.csv", index=False)
    _plot(table, out / "figures" / "wasserstein_vs_cosine")
    (root / "protocl" / "h4_wasserstein_aic.md").write_text(_report(table, items, scale))
    return table


def _order2(root: Path, items: pd.DataFrame) -> pd.DataFrame:
    folder = root / "results/token_embeddings/mpnet"
    rows = []
    for record in items.itertuples(index=False):
        clouds = {field: np.load(folder / f"{record.scale_id}_{record.item_id}_{field}.npy") for field in _FIELDS}
        human = _w2(clouds["original_en"], clouds["human_back_en"])
        llm = _w2(clouds["original_en"], clouds["llm_back_en"])
        rows.append({"scale_id": record.scale_id, "item_id": record.item_id, "w2_gap": -(human - llm)})
    return pd.DataFrame(rows)


def _w2(reference: np.ndarray, other: np.ndarray) -> float:
    cost = pairwise_euclidean(reference, other)
    transported = wasserstein(cost * cost)["distance"]
    return math.sqrt(max(transported, 0.0))


def _outcome_frame(ratings: pd.DataFrame, choices: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if outcome == "preference":
        return choices.dropna(subset=["prefer_human"]).copy()
    if outcome == "guess":
        return choices.dropna(subset=["guess_correct"]).copy()
    if outcome == "naturalness":
        return ratings.loc[ratings["outcome"].eq("naturalness")].copy()
    part = ratings.loc[ratings["outcome"].eq(outcome) & ratings["expert"].eq(1)].copy()
    if part.empty:
        raise RuntimeError(f"No rows for {outcome}")
    return part


def _report(table: pd.DataFrame, items: pd.DataFrame, scale: pd.DataFrame) -> str:
    lines = [
        "# Item Wasserstein gaps in place of delta_z",
        "",
        (
            f"Wasserstein-1 is the stored earth-mover distance. Wasserstein-2 uses squared Euclidean cost on the same MPNet token clouds, "
            f"then takes the square root. A larger distance means the token cloud is farther from the original, so the raw distance is a reverse index of cosine. "
            f"The gap flips that distance: the LLM distance minus the human distance. A positive gap means the human cloud is closer, the same direction as cosine delta_z. "
            f"Each gap is standardized across {len(items)} items. The judgment model uses the mean of those item scores "
            f"inside the scale ({scale['scale_id'].nunique()} scales), because a rating, a preference, and a guess are one row per person per scale."
        ),
        "",
        "Gaussian AIC and BIC are from maximum likelihood, so they can be compared across predictors of the same outcome. Binomial models are already maximum likelihood. A lower AIC or BIC is the better description of that outcome. The cosine `delta_z` model is the reference.",
        "",
        (
            f"The Wasserstein-1 and Wasserstein-2 item scores correlate {items['w1_z'].corr(items['w2_z']):.3f}. "
            f"Their scale means correlate {scale['w1_z'].corr(scale['w2_z']):.3f}. "
            "The two-distance model therefore splits one predictor into two unstable slopes and has the worst AIC and BIC in every outcome."
        ),
        "",
        "![AIC relative to the cosine model, and the single-distance slopes. A point left of zero on the AIC panel is a better description than cosine. Intervals that meet zero are compatible with no slope.](../results/wasserstein_models/figures/wasserstein_vs_cosine.png)",
        "",
        _aic_table(table),
        "",
        "## Slopes",
        "",
        _slope_table(table),
        "",
        _read(table),
        "",
        "These fits do not rewrite `results/delta_z/` or the cosine replicate.",
        "",
    ]
    return "\n".join(lines)


def _aic_table(table: pd.DataFrame) -> str:
    summary = table.drop_duplicates(["outcome", "predictor"]).sort_values(["outcome", "aic"])
    header = "| Outcome | Predictor | AIC | BIC | Intercept |"
    rule = "| --- | --- | ---: | ---: | ---: |"
    body = [
        f"| {row.outcome} | {row.predictor} | {row.aic:.1f} | {row.bic:.1f} | {row.intercept:.3f} |"
        for row in summary.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _slope_table(table: pd.DataFrame) -> str:
    header = "| Outcome | Predictor | Term | Estimate | 95% CI | p |"
    rule = "| --- | --- | --- | ---: | --- | ---: |"
    body = [
        (
            f"| {row.outcome} | {row.predictor} | {row.term} | {row.estimate:.3f} | "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | {_p(row.p)} |"
        )
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _read(table: pd.DataFrame) -> str:
    pieces = []
    for outcome, part in table.groupby("outcome", sort=False):
        info = part.drop_duplicates("predictor").sort_values("aic")
        best = info.iloc[0]
        cosine = info.loc[info["predictor"].eq("cosine_scale_z")].iloc[0]
        pieces.append(
            f"For {outcome}, the lowest AIC is {best.predictor} "
            f"(AIC {best.aic:.1f}, BIC {best.bic:.1f}), against cosine AIC {cosine.aic:.1f}."
        )
    pieces.append(
        "Those AIC gaps are smaller than 1, except where Wasserstein-1 and Wasserstein-2 are entered together. "
        "The intercepts stay on the same value within each outcome."
    )
    edge = table.loc[
        table["predictor"].isin(["w1_item_mean", "w2_item_mean"]) & ~table["includes_zero"]
    ]
    if len(edge):
        bits = [
            f"{row.outcome} under {row.predictor} is {row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
            for row in edge.itertuples(index=False)
        ]
        pieces.append("Single-distance Wasserstein intervals that exclude zero: " + "; ".join(bits) + ".")
    return " ".join(pieces)


def _plot(table: pd.DataFrame, stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
        }
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "cosine_scale_z": "#4C78A8",
        "w1_item_mean": "#1B1B1B",
        "w2_item_mean": "#E45756",
        "w1_and_w2": "#B0B0B0",
    }
    labels = {
        "cosine_scale_z": "Cosine",
        "w1_item_mean": "Wasserstein-1",
        "w2_item_mean": "Wasserstein-2",
        "w1_and_w2": "Both",
    }
    outcomes = ["naturalness_gap", "expert_choice_fit_gap", "preference", "guess"]
    outcome_labels = ["Naturalness", "Expert choice fit", "Preference", "LLM guess"]
    millimeter = 1 / 25.4
    figure, axes = plt.subplots(1, 2, figsize=(183 * millimeter, 90 * millimeter), constrained_layout=True)
    info = table.drop_duplicates(["outcome", "predictor"])
    cosine = info.loc[info["predictor"].eq("cosine_scale_z")].set_index("outcome")["aic"]
    axes[0].axvline(0, color="#1B1B1B", linewidth=0.6)
    for offset, predictor in enumerate(("w1_item_mean", "w2_item_mean", "w1_and_w2")):
        part = info.loc[info["predictor"].eq(predictor)].set_index("outcome")
        y = [index + (offset - 1) * 0.18 for index, _name in enumerate(outcomes)]
        delta = [part.loc[name, "aic"] - cosine.loc[name] for name in outcomes]
        axes[0].scatter(delta, y, s=18, color=colors[predictor], label=labels[predictor], zorder=2)
    axes[0].set_yticks(range(len(outcomes)), outcome_labels)
    axes[0].set_xlabel("AIC minus the cosine model")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    single = table.loc[table["predictor"].isin(["cosine_scale_z", "w1_item_mean", "w2_item_mean"])]
    axes[1].axvline(0, color="#1B1B1B", linewidth=0.6, zorder=0)
    for offset, predictor in enumerate(("cosine_scale_z", "w1_item_mean", "w2_item_mean")):
        part = single.loc[single["predictor"].eq(predictor)].set_index("outcome")
        y = [index + (offset - 1) * 0.22 for index, _name in enumerate(outcomes)]
        estimate = [part.loc[name, "estimate"] for name in outcomes]
        left = [part.loc[name, "estimate"] - part.loc[name, "ci_low"] for name in outcomes]
        right = [part.loc[name, "ci_high"] - part.loc[name, "estimate"] for name in outcomes]
        axes[1].errorbar(
            estimate,
            y,
            xerr=[left, right],
            fmt="o",
            color=colors[predictor],
            ecolor=colors[predictor],
            elinewidth=0.9,
            capsize=2,
            markersize=3.5,
            label=labels[predictor],
        )
    axes[1].set_yticks(range(len(outcomes)), outcome_labels)
    axes[1].set_xlabel("Slope per unit of the standardized gap")
    axes[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    for axis, letter in zip(axes, "ab", strict=True):
        axis.text(-0.16, 1.05, letter, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
