"""Join stored NLP metrics to expert judgments and citizen item responses.

This is the exploratory three-level check in the 2026-09-29 comment:
representation, expert judgment, and within-person response behavior.
It reads the fixed tables from ``python -m h4.survey.preanalysis`` and does
not change the confirmatory cosine estimate. COMET, BERTScore, an LLM judge,
and a new perturbation lexicon are not in the stored artifacts, so they are
not scored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from h4.io import write_json
from h4.statistics.mixed import fit_hierarchical
from h4.survey.reshape import response_bounds

_ROOT = Path(__file__).resolve().parents[3]
_MODEL = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "minilm",
    "sentence-transformers/LaBSE": "labse",
    "intfloat/multilingual-e5-large-instruct": "e5",
}


def main() -> None:
    summary = run(_ROOT)
    print(json.dumps(summary, indent=2))


def run(root: Path) -> dict:
    analysis = root / "data" / "survey" / "4_analysis"
    responses = pd.read_parquet(analysis / "citizen_item_response.parquet")
    ratings = pd.read_parquet(analysis / "expert_scale_rating.parquet")
    problems = pd.read_parquet(analysis / "expert_item_problem.parquet")
    items = pd.read_parquet(analysis / "catalog_items.parquet")
    labels = pd.read_csv(root / "data/processed/labels.csv", dtype=str)
    metrics = _item_metrics(root)
    expert = _expert_scale_delta(ratings)
    expert_models = _expert_models(ratings)
    flags = _problem_rates(problems)
    behavior, scores = _behavior(responses, items, response_bounds(labels))
    scale = _scale_frame(expert, behavior, metrics)
    associations = _associations(scale, behavior, metrics, flags)
    loso = _loso(behavior, metrics)
    out = root / "results" / "survey_meta"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    expert.to_csv(table_dir / "expert_scale_delta.csv", index=False)
    expert_models.to_csv(table_dir / "expert_dimension_models.csv", index=False)
    behavior.to_csv(table_dir / "behavior_item.csv", index=False)
    scores.to_csv(table_dir / "scale_scores.csv", index=False)
    scale.to_csv(table_dir / "scale_triangle.csv", index=False)
    associations.to_csv(table_dir / "associations.csv", index=False)
    flags.to_csv(table_dir / "expert_item_problem_rate.csv", index=False)
    loso.to_csv(table_dir / "loso_behavior.csv", index=False)
    _plot_expert(scale, figure_dir / "expert_vs_cosine.png")
    _plot_behavior(behavior, metrics, figure_dir / "behavior_vs_cosine.png")
    _plot_scores(scores, figure_dir / "scale_score_correlation.png")
    report = _report(expert, expert_models, scores, associations, loso, scale)
    path = root / "protocl" / "h4_metaeval_report.md"
    path.write_text(report)
    write_json(
        {
            "n_behavior_items": int(behavior["scale_id"].nunique() and len(behavior)),
            "n_scales": int(scale["scale_id"].nunique()),
            "report": str(path.relative_to(root)),
        },
        table_dir / "metaeval_summary.json",
    )
    return {"report": str(path.relative_to(root)), "n_associations": int(len(associations))}


def _item_metrics(root: Path) -> pd.DataFrame:
    similarity = pd.read_parquet(root / "results/similarities/similarity_long.parquet")
    similarity = similarity.loc[similarity["metric"].eq("cosine")].copy()
    similarity["scale_id"] = similarity["scale_id"].map(_key)
    similarity["item_id"] = similarity["item_id"].map(lambda value: str(value).zfill(2))
    similarity["model"] = similarity["embedding_model"].map(_MODEL)
    wide = similarity.pivot_table(
        index=["scale_id", "item_id", "model", "comparison"],
        columns="translation",
        values="similarity",
        aggfunc="first",
    ).reset_index()
    wide["delta_cosine"] = wide["human"] - wide["llm"]
    wide["abs_delta_cosine"] = wide["delta_cosine"].abs()
    tokens = pd.read_csv(root / "results/exploratory/pooling/token_distance_pairs.csv")
    tokens["scale_id"] = tokens["scale_id"].map(_key)
    tokens["item_id"] = tokens["item_id"].map(lambda value: str(value).zfill(2))
    tokens = tokens.loc[tokens["short_name"].eq("mpnet"), ["scale_id", "item_id", "comparison", "delta_d_mu", "delta_d_sigma"]]
    water = pd.read_csv(root / "results/exploratory/geometry/wasserstein.csv")
    water["scale_id"] = water["scale_id"].map(_key)
    water["item_id"] = water["item_id"].map(lambda value: str(value).zfill(2))
    water = water.loc[
        water["embedding_model"].str.contains("mpnet") & water["cost"].eq("euclidean"),
        ["scale_id", "item_id", "comparison", "delta_distance"],
    ]
    cosine = wide.merge(tokens, on=["scale_id", "item_id", "comparison"], how="left")
    return cosine.merge(water, on=["scale_id", "item_id", "comparison"], how="left")


def _expert_scale_delta(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings["rating"] = pd.to_numeric(ratings["raw_value"], errors="coerce")
    grouped = (
        ratings.dropna(subset=["rating"])
        .groupby(["scale_id", "scale_name", "dimension", "version"], as_index=False)
        .agg(mean_rating=("rating", "mean"), n=("rating", "size"))
    )
    wide = grouped.pivot(index=["scale_id", "scale_name", "dimension"], columns="version", values="mean_rating")
    counts = grouped.pivot(index=["scale_id", "scale_name", "dimension"], columns="version", values="n")
    wide = wide.reset_index()
    counts = counts.reset_index()
    wide["delta"] = wide["human"] - wide["llm"]
    wide["n_human"] = counts["human"]
    wide["n_llm"] = counts["llm"]
    return wide


def _expert_models(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings["rating"] = pd.to_numeric(ratings["raw_value"], errors="coerce")
    wide = ratings.pivot_table(
        index=["respondent_id", "scale_id", "dimension"],
        columns="version",
        values="rating",
        aggfunc="first",
    ).reset_index()
    rows = []
    for dimension, group in wide.groupby("dimension"):
        paired = group.dropna(subset=["human", "llm"]).copy()
        paired["delta"] = paired["human"] - paired["llm"]
        fit = fit_hierarchical(paired.loc[:, ["delta", "scale_id"]])
        rows.append(
            {
                "dimension": dimension,
                "n_pairs": int(len(paired)),
                "estimate": fit["estimate"],
                "ci_low": fit["ci_low"],
                "ci_high": fit["ci_high"],
                "share_human_higher": float((paired["delta"] > 0).mean()),
                "share_tie": float((paired["delta"] == 0).mean()),
                "share_llm_higher": float((paired["delta"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _problem_rates(problems: pd.DataFrame) -> pd.DataFrame:
    problems = problems.copy()
    problems["flagged"] = problems["raw_value"].eq("1").fillna(False).astype(float)
    return (
        problems.groupby(["scale_id", "item_id", "dimension", "version"], as_index=False)
        .agg(flag_rate=("flagged", "mean"), n=("flagged", "size"))
    )


def _behavior(
    responses: pd.DataFrame,
    items: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = responses.copy()
    frame["response"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    wide = frame.pivot_table(
        index=["respondent_id", "set_id", "scale_id", "scale_name", "item_id", "trans_cond"],
        columns="version",
        values="response",
        aggfunc="first",
    ).reset_index()
    wide["delta"] = wide["human"] - wide["llm"]
    wide["abs_delta"] = wide["delta"].abs()
    paired = wide.dropna(subset=["human", "llm"])
    behavior = (
        paired.groupby(["scale_id", "scale_name", "item_id"], as_index=False)
        .agg(
            n=("delta", "size"),
            mean_human=("human", "mean"),
            mean_llm=("llm", "mean"),
            mean_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
        )
    )
    keys = items.loc[:, ["scale_id", "item_id", "reverse_keyed"]].copy()
    keys["scale_id"] = keys["scale_id"].map(_key)
    keys["item_id"] = keys["item_id"].map(lambda value: str(value).zfill(2))
    scored = wide.merge(keys, on=["scale_id", "item_id"], how="left")
    scored["score_human"] = [
        _reverse(human, reverse, bounds.get(scale))
        for human, reverse, scale in zip(scored["human"], scored["reverse_keyed"], scored["scale_id"], strict=True)
    ]
    scored["score_llm"] = [
        _reverse(llm, reverse, bounds.get(scale))
        for llm, reverse, scale in zip(scored["llm"], scored["reverse_keyed"], scored["scale_id"], strict=True)
    ]
    person = (
        scored.groupby(["respondent_id", "scale_id", "scale_name", "trans_cond"], as_index=False)
        .agg(
            n_human=("score_human", "count"),
            n_llm=("score_llm", "count"),
            sum_human=("score_human", "sum"),
            sum_llm=("score_llm", "sum"),
            n_items=("item_id", "nunique"),
        )
    )
    complete = person.loc[person["n_human"].eq(person["n_items"]) & person["n_llm"].eq(person["n_items"])].copy()
    score_rows = []
    for (scale_id, scale_name, order), group in complete.groupby(["scale_id", "scale_name", "trans_cond"], sort=True):
        score_rows.append(_score_row(scale_id, scale_name, str(order), group))
    for (scale_id, scale_name), group in complete.groupby(["scale_id", "scale_name"], sort=True):
        score_rows.append(_score_row(scale_id, scale_name, "all", group))
    alpha_rows = _alphas(scored)
    scores = pd.DataFrame(score_rows).merge(pd.DataFrame(alpha_rows), on=["scale_id", "scale_name"], how="left")
    return behavior, scores


def _score_row(scale_id: str, scale_name: str, order: str, group: pd.DataFrame) -> dict:
    rho, p_value = _corr(group["sum_human"], group["sum_llm"])
    return {
        "scale_id": scale_id,
        "scale_name": scale_name,
        "trans_cond": order,
        "n": int(len(group)),
        "mean_human": float(group["sum_human"].mean()),
        "mean_llm": float(group["sum_llm"].mean()),
        "mean_delta": float((group["sum_human"] - group["sum_llm"]).mean()),
        "score_correlation": rho,
        "score_correlation_p": p_value,
    }


def _alphas(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for (scale_id, scale_name), group in scored.groupby(["scale_id", "scale_name"], sort=True):
        row = {"scale_id": scale_id, "scale_name": scale_name}
        for version, column in (("human", "score_human"), ("llm", "score_llm")):
            wide = group.pivot(index="respondent_id", columns="item_id", values=column)
            complete = wide.dropna()
            row[f"alpha_{version}"] = _alpha(complete)
            row[f"alpha_n_{version}"] = int(len(complete))
        rows.append(row)
    return rows


def _scale_frame(expert: pd.DataFrame, behavior: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    behavior_scale = behavior.groupby(["scale_id", "scale_name"], as_index=False).agg(
        mean_abs_response=("mean_abs_delta", "mean"),
        mean_response_delta=("mean_delta", "mean"),
    )
    mpnet = metrics.loc[metrics["model"].eq("mpnet")]
    metric_scale = mpnet.groupby(["scale_id", "comparison"], as_index=False).agg(
        mean_delta_cosine=("delta_cosine", "mean"),
        mean_abs_delta_cosine=("abs_delta_cosine", "mean"),
        mean_delta_d_mu=("delta_d_mu", "mean"),
        mean_delta_d_sigma=("delta_d_sigma", "mean"),
    )
    back = metric_scale.loc[metric_scale["comparison"].eq("back_translation")].drop(columns="comparison")
    direct = metric_scale.loc[metric_scale["comparison"].eq("direct_crosslingual")].drop(columns="comparison")
    back = back.rename(columns={column: f"back_{column}" for column in back.columns if column != "scale_id"})
    direct = direct.rename(columns={column: f"direct_{column}" for column in direct.columns if column != "scale_id"})
    semantic = expert.loc[expert["dimension"].eq("semantic"), ["scale_id", "delta", "human", "llm"]].rename(
        columns={"delta": "expert_semantic_delta", "human": "expert_semantic_human", "llm": "expert_semantic_llm"}
    )
    natural = expert.loc[expert["dimension"].eq("naturalness"), ["scale_id", "delta"]].rename(
        columns={"delta": "expert_naturalness_delta"}
    )
    cultural = expert.loc[expert["dimension"].eq("cultural_validity"), ["scale_id", "delta"]].rename(
        columns={"delta": "expert_cultural_delta"}
    )
    frame = behavior_scale.merge(back, on="scale_id", how="left").merge(direct, on="scale_id", how="left")
    return frame.merge(semantic, on="scale_id", how="left").merge(natural, on="scale_id", how="left").merge(cultural, on="scale_id", how="left")


def _associations(
    scale: pd.DataFrame,
    behavior: pd.DataFrame,
    metrics: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dimension, column in (
        ("semantic", "expert_semantic_delta"),
        ("naturalness", "expert_naturalness_delta"),
        ("cultural_validity", "expert_cultural_delta"),
    ):
        for predictor, label in (
            ("back_mean_delta_cosine", "mpnet back-translation cosine delta"),
            ("direct_mean_delta_cosine", "mpnet direct cosine delta"),
        ):
            rows.append(_spearman("scale", dimension, label, scale[predictor], scale[column]))
        rows.append(
            _spearman(
                "scale",
                "behavior_abs",
                "mpnet back-translation |cosine delta|",
                scale["back_mean_abs_delta_cosine"],
                scale["mean_abs_response"],
            )
        )
    item = behavior.merge(
        metrics.loc[metrics["model"].eq("mpnet") & metrics["comparison"].eq("back_translation")],
        on=["scale_id", "item_id"],
        how="inner",
    )
    direct = metrics.loc[metrics["model"].eq("mpnet") & metrics["comparison"].eq("direct_crosslingual"), ["scale_id", "item_id", "abs_delta_cosine"]]
    direct = direct.rename(columns={"abs_delta_cosine": "abs_delta_cosine_direct"})
    item = item.merge(direct, on=["scale_id", "item_id"], how="left")
    rows.append(_spearman("item", "behavior_abs", "mpnet back-translation |cosine delta|", item["abs_delta_cosine"], item["mean_abs_delta"]))
    rows.append(_spearman("item", "behavior_abs", "mpnet direct |cosine delta|", item["abs_delta_cosine_direct"], item["mean_abs_delta"]))
    rows.append(_spearman("item", "behavior_abs", "mpnet back-translation d_mu gap", item["delta_d_mu"], item["mean_abs_delta"]))
    rows.append(_spearman("item", "behavior_abs", "mpnet back-translation d_sigma gap", item["delta_d_sigma"], item["mean_abs_delta"]))
    rows.append(_spearman("item", "behavior_abs", "mpnet back-translation Wasserstein gap", item["delta_distance"], item["mean_abs_delta"]))
    for model, part in metrics.loc[metrics["model"].ne("mpnet")].groupby("model"):
        for comparison, chunk in part.groupby("comparison"):
            joined = behavior.merge(chunk, on=["scale_id", "item_id"], how="inner")
            rows.append(
                _spearman(
                    "item",
                    "behavior_abs",
                    f"{model} {comparison} |cosine delta|",
                    joined["abs_delta_cosine"],
                    joined["mean_abs_delta"],
                )
            )
    semantic_flags = flags.loc[flags["dimension"].eq("semantic")].groupby(["scale_id", "item_id"], as_index=False)["flag_rate"].mean()
    flagged = semantic_flags.merge(item, on=["scale_id", "item_id"], how="inner")
    rows.append(_spearman("item", "expert_semantic_flag", "mpnet back-translation |cosine delta|", flagged["abs_delta_cosine"], flagged["flag_rate"]))
    return pd.DataFrame(rows).drop_duplicates(["level", "outcome", "predictor"])


def _loso(behavior: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    item = behavior.merge(
        metrics.loc[metrics["model"].eq("mpnet") & metrics["comparison"].eq("back_translation"), ["scale_id", "item_id", "abs_delta_cosine"]],
        on=["scale_id", "item_id"],
        how="inner",
    )
    rows = []
    for held in sorted(item["scale_id"].unique()):
        train = item.loc[item["scale_id"] != held]
        test = item.loc[item["scale_id"] == held]
        fit = smf.ols("mean_abs_delta ~ abs_delta_cosine", data=train).fit()
        predicted = fit.predict(test)
        rho, p_value = _corr(predicted, test["mean_abs_delta"])
        rows.append(
            {
                "held_out_scale": held,
                "n_test": int(len(test)),
                "spearman_rho": rho,
                "spearman_p": p_value,
                "mae": float(np.mean(np.abs(predicted.to_numpy() - test["mean_abs_delta"].to_numpy()))),
            }
        )
    return pd.DataFrame(rows)


def _plot_expert(scale: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, column, title in (
        (axes[0], "back_mean_delta_cosine", "Back-translation"),
        (axes[1], "direct_mean_delta_cosine", "Direct Japanese"),
    ):
        axis.axhline(0, color="#bbbbbb", linewidth=0.8)
        axis.axvline(0, color="#bbbbbb", linewidth=0.8)
        axis.scatter(scale[column], scale["expert_semantic_delta"], color="#2c7fb8", s=36)
        for _, row in scale.iterrows():
            axis.annotate(row["scale_name"], (row[column], row["expert_semantic_delta"]), fontsize=7, alpha=0.8)
        axis.set_xlabel("Mean cosine(human, original) − cosine(LLM, original)")
        axis.set_title(title)
    axes[0].set_ylabel("Expert semantic rating, human − LLM")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_behavior(behavior: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item = behavior.merge(
        metrics.loc[metrics["model"].eq("mpnet"), ["scale_id", "item_id", "comparison", "abs_delta_cosine"]],
        on=["scale_id", "item_id"],
        how="inner",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, comparison, title in (
        (axes[0], "back_translation", "Back-translation"),
        (axes[1], "direct_crosslingual", "Direct Japanese"),
    ):
        part = item.loc[item["comparison"].eq(comparison)]
        axis.scatter(part["abs_delta_cosine"], part["mean_abs_delta"], s=12, color="#d95f0e", alpha=0.75)
        axis.set_xlabel("|cosine delta|")
        axis.set_title(title)
    axes[0].set_ylabel("Mean |human response − LLM response|")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_scores(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = scores.loc[scores["trans_cond"].eq("all")].sort_values("score_correlation")
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    positions = np.arange(len(part))
    axis.axvline(0.8, color="#bbbbbb", linewidth=0.8)
    axis.scatter(part["score_correlation"], positions, color="#2c7fb8")
    axis.set_yticks(positions)
    axis.set_yticklabels(part["scale_name"])
    axis.set_xlabel("Within-person correlation of scale scores")
    axis.set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(
    expert: pd.DataFrame,
    expert_models: pd.DataFrame,
    scores: pd.DataFrame,
    associations: pd.DataFrame,
    loso: pd.DataFrame,
    scale: pd.DataFrame,
) -> str:
    overall = scores.loc[scores["trans_cond"].eq("all")]
    above = int((overall["score_correlation"] >= 0.8).sum())
    lines = [
        "# Three-level check: metrics, expert judgments, and citizen responses",
        "",
        "The long tables in `data/survey/4_analysis/` are the input. They match the Quarto processed files after the fixed reshape. Each citizen answered both translation versions; `trans_cond` records which version came first. The behavioral contrast is within person. The confirmatory cosine estimate is unchanged.",
        "",
        "Stored metrics enter this check: cosine from MPNet, MiniLM, LaBSE, and multilingual E5, plus the MPNet Hara mean distance, covariance distance, and token Wasserstein gap. COMET, BERTScore, an LLM judge, and a newly written perturbation lexicon are not in the stored artifacts, so this pass does not score them. The existing negation control remains in the metric-validation tables.",
        "",
        "## Expert judgments",
        "",
        "Experts rated each assigned scale on semantic fidelity, naturalness, cultural validity, and choice fit, separately for the human version and the LLM version. The scale summary is the mean human rating minus the mean LLM rating. Item problem checkboxes are a second expert signal: a cell equal to 1 is a flagged problem.",
        "",
        _expert_sentence(expert_models),
        "",
        "## Citizen scale scores",
        "",
        "A scale score is the sum of administered items after reverse-keying. Three SCSS attention rows are in the workbook and are absent from the citizen item export, so they are not in the sum. The memo's parallel-form reference is a correlation of 0.80.",
        "",
        f"Across 22 scales the median within-person correlation is {overall['score_correlation'].median():.3f}. {above} of 22 scales are at or above 0.80. The median absolute mean-score gap is {overall['mean_delta'].abs().median():.3f} points.",
        "",
        "![Within-person correlation of human-version and LLM-version scale scores. The gray line is 0.80.](../results/survey_meta/figures/scale_score_correlation.png)",
        "",
        "## Do the metrics agree with experts and with response gaps?",
        "",
        "A positive expert correlation means that scales where the human translation is closer to the original in the metric are also the scales experts rate higher for the human version. A positive behavior correlation means a larger metric gap goes with a larger average absolute difference in the answers a person gave to the two versions.",
        "",
        _association_table(associations),
        "",
        _association_prose(associations),
        "",
        "![Expert semantic gap against the MPNet cosine gap. Each point is a scale.](../results/survey_meta/figures/expert_vs_cosine.png)",
        "",
        "![Item-level mean absolute response gap against the absolute MPNet cosine gap.](../results/survey_meta/figures/behavior_vs_cosine.png)",
        "",
        "## Leave-one-scale-out",
        "",
        "For each held-out scale, an ordinary regression of the item's mean absolute response gap on the absolute MPNet back-translation cosine gap is fit on the other 21 scales and scored on the held-out items.",
        "",
        (
            f"The mean held-out Spearman is {loso['spearman_rho'].mean():.3f} "
            f"(median {loso['spearman_rho'].median():.3f}). "
            f"The mean absolute error on the response-gap scale is {loso['mae'].mean():.3f}."
        ),
        "",
        "Tables: `results/survey_meta/tables/`.",
        "",
    ]
    return "\n".join(lines)


def _expert_sentence(models: pd.DataFrame) -> str:
    parts = []
    for _, row in models.sort_values("dimension").iterrows():
        parts.append(
            f"{row['dimension']} {row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
            f"(human higher on {row['share_human_higher']:.0%} of pairs, tie {row['share_tie']:.0%}, LLM higher on {row['share_llm_higher']:.0%})"
        )
    return "Human minus LLM, random intercept for scale: " + "; ".join(parts) + "."


def _association_table(frame: pd.DataFrame) -> str:
    header = "| Level | Outcome | Predictor | n | Spearman rho | p |"
    rule = "| --- | --- | --- | ---: | ---: | ---: |"
    body = []
    shown = frame.drop_duplicates(["level", "outcome", "predictor"])
    for _, row in shown.iterrows():
        rho = "—" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        p_value = "—" if pd.isna(row["spearman_p"]) else ("< 0.001" if row["spearman_p"] < 0.001 else f"{row['spearman_p']:.3f}")
        body.append(f"| {row['level']} | {row['outcome']} | {row['predictor']} | {int(row['n'])} | {rho} | {p_value} |")
    return "\n".join([header, rule, *body])


def _association_prose(frame: pd.DataFrame) -> str:
    def pick(outcome: str, predictor: str) -> pd.Series:
        rows = frame.loc[frame["outcome"].eq(outcome) & frame["predictor"].eq(predictor)]
        return rows.iloc[0]

    semantic = pick("semantic", "mpnet back-translation cosine delta")
    direct = pick("semantic", "mpnet direct cosine delta")
    behavior = pick("behavior_abs", "mpnet back-translation |cosine delta|")
    # item row overwrites if both levels share the predictor text. Filter level.
    behavior = frame.loc[
        frame["level"].eq("item")
        & frame["outcome"].eq("behavior_abs")
        & frame["predictor"].eq("mpnet back-translation |cosine delta|")
    ].iloc[0]
    scale_behavior = frame.loc[
        frame["level"].eq("scale")
        & frame["outcome"].eq("behavior_abs")
        & frame["predictor"].eq("mpnet back-translation |cosine delta|")
    ].iloc[0]
    flagged = frame.loc[
        frame["outcome"].eq("expert_semantic_flag") & frame["predictor"].eq("mpnet back-translation |cosine delta|")
    ].iloc[0]
    return (
        f"At the scale level, the MPNet back-translation cosine gap and the expert semantic gap have Spearman {semantic['spearman_rho']:.3f} (p {semantic['spearman_p']:.3f}). "
        f"The direct-Japanese gap has Spearman {direct['spearman_rho']:.3f} (p {direct['spearman_p']:.3f}). "
        f"The same back-translation absolute gap and the mean absolute response gap have scale-level Spearman {scale_behavior['spearman_rho']:.3f} (p {scale_behavior['spearman_p']:.3f}) "
        f"and item-level Spearman {behavior['spearman_rho']:.3f} (p {behavior['spearman_p']:.3f}). "
        f"Items with a larger absolute back-translation gap are flagged more often as a semantic problem "
        f"(Spearman {flagged['spearman_rho']:.3f}, p {_p_text(flagged['spearman_p'])})."
    )


def _p_text(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _spearman(level: str, outcome: str, predictor: str, left: pd.Series, right: pd.Series) -> dict:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    rho, p_value = _corr(pair["left"], pair["right"])
    return {
        "level": level,
        "outcome": outcome,
        "predictor": predictor,
        "n": int(len(pair)),
        "spearman_rho": rho,
        "spearman_p": p_value,
    }


def _corr(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan"), float("nan")
    rho, p_value = stats.spearmanr(pair["left"], pair["right"])
    return float(rho), float(p_value)


def _alpha(wide: pd.DataFrame) -> float:
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        return float("nan")
    item_var = wide.var(axis=0, ddof=1).sum()
    total_var = wide.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    k = wide.shape[1]
    return float((k / (k - 1)) * (1 - item_var / total_var))


def _reverse(value: object, reverse: object, bounds: tuple[float, float] | None) -> float:
    if pd.isna(value) or bounds is None:
        return float("nan")
    number = float(value)
    if bool(reverse):
        return float(bounds[0] + bounds[1] - number)
    return number


def _key(value: object) -> str:
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() else text


if __name__ == "__main__":
    main()
