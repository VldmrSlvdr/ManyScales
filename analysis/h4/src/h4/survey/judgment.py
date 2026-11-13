"""Naturalness, preference, and LLM-guess analyses of the survey judgments.

Preference and ``llm_guess`` are stored as translation slot 1 or 2. In both
Qualtrics surveys the comparison table is fixed: column 翻訳1 is the human
Japanese text and column 翻訳2 is the LLM Japanese text. ``trans_cond`` records
which version was administered first. It does not swap those columns.

Naturalness columns are already labeled by the true version (``_human`` and
``_ai``). Citizen naturalness is the wording/clarity rating. This module does
not enter the confirmatory cosine model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from h4.catalog import load_catalog
from h4.statistics.mixed import fit_hierarchical
from h4.survey.codebook import plain_text, _norm
from h4.survey.qualtrics import read_qualtrics_tsv

_SETS = tuple("ABCDEF")
_ITEM_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def fold_text(value: str) -> str:
    return _norm(value).replace(",", "、").replace("，", "、")


def decode_preference(code: object) -> float:
    """Return 1 when the respondent preferred the human column (翻訳1)."""

    text = "" if code is None or (isinstance(code, float) and np.isnan(code)) else str(code).strip()
    if text == "1":
        return 1.0
    if text == "2":
        return 0.0
    return float("nan")


def decode_guess(code: object) -> float:
    """Return 1 when the respondent named the LLM column (翻訳2) as the AI version."""

    text = "" if code is None or (isinstance(code, float) and np.isnan(code)) else str(code).strip()
    if text == "2":
        return 1.0
    if text == "1":
        return 0.0
    return float("nan")


def comparison_column_audit(qsf_path: Path, catalog_path: Path) -> pd.DataFrame:
    """Match each comparison-table cell to the workbook Japanese texts."""

    catalog = load_catalog(catalog_path)
    items = {}
    for scale in catalog["scales"]:
        if scale.get("status") != "study":
            continue
        for item in scale["items"]:
            items[(str(scale["scale_id"]), str(item["item_id"]).zfill(2))] = item
    document = json.loads(Path(qsf_path).read_text())
    tables: dict[str, set[tuple[str, str]]] = {}
    for element in document["SurveyElements"]:
        if element.get("Element") != "SQ":
            continue
        payload = element.get("Payload") or {}
        tag = str(payload.get("DataExportTag") or "")
        if not tag.endswith("_comparison_table"):
            continue
        scale_id = tag[1:3]
        chunk = str(payload.get("QuestionText") or "").split("質問項目")[-1]
        cells = _ITEM_CELL.findall(chunk)
        pairs = tables.setdefault(scale_id, set())
        for index in range(len(cells) // 3):
            pairs.add((fold_text(plain_text(cells[3 * index + 1])), fold_text(plain_text(cells[3 * index + 2]))))
    rows = []
    for (scale_id, item_id), item in sorted(items.items()):
        human = fold_text(str(item.get("human_ja") or ""))
        llm = fold_text(str(item.get("llm_ja") or ""))
        pairs = tables.get(scale_id, set())
        human_in_slot1 = any(pair[0] == human and pair[1] == llm for pair in pairs)
        swapped = human != llm and any(pair[0] == llm and pair[1] == human for pair in pairs)
        rows.append(
            {
                "scale_id": scale_id,
                "item_id": item_id,
                "slot1_is_human": human_in_slot1,
                "swapped": swapped,
                "in_comparison_table": human_in_slot1 or swapped,
            }
        )
    return pd.DataFrame(rows)


def between_version_similarity(embedding_path: Path, keep: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Item and scale cosine similarity of the two Japanese translations."""

    frame = pd.read_parquet(embedding_path)
    frame["scale_id"] = frame["scale_id"].map(_scale_key)
    frame["item_id"] = frame["item_id"].map(lambda value: str(value).zfill(2))
    vectors = {}
    for field in ("human_ja", "llm_ja"):
        part = frame.loc[frame["text_field"] == field, ["scale_id", "item_id", "embedding"]]
        vectors[field] = part.set_index(["scale_id", "item_id"])["embedding"]
    rows = []
    for key, human in vectors["human_ja"].items():
        llm = vectors["llm_ja"].get(key)
        if llm is None:
            continue
        human_vector = np.asarray(human, dtype=float)
        llm_vector = np.asarray(llm, dtype=float)
        rows.append(
            {
                "scale_id": key[0],
                "item_id": key[1],
                "cosine_ja": float(np.dot(human_vector, llm_vector) / (np.linalg.norm(human_vector) * np.linalg.norm(llm_vector))),
            }
        )
    items = pd.DataFrame(rows)
    if keep is not None:
        items = items.merge(keep.loc[:, ["scale_id", "item_id", "in_comparison_table"]], on=["scale_id", "item_id"], how="left")
        items = items.loc[items["in_comparison_table"] == True].copy()  # noqa: E712
    scale = (
        items.groupby("scale_id", as_index=False)
        .agg(similarity_ja=("cosine_ja", "mean"), n_similarity_items=("cosine_ja", "size"))
        .sort_values("scale_id")
    )
    return items, scale


def load_judgments(processed_dir: Path, catalog_path: Path) -> pd.DataFrame:
    """One row per respondent, scale, and survey in the analysis tables."""

    names = _scale_names(catalog_path)
    frames = [_survey_judgments(processed_dir, "citizen", "wording"), _survey_judgments(processed_dir, "expert", "naturalness")]
    frame = pd.concat(frames, ignore_index=True)
    frame["scale_name"] = frame["scale_id"].map(names)
    return frame


def naturalness_contrast(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Human-minus-LLM naturalness, overall and within each administration order."""

    overall_rows = []
    scale_rows = []
    for survey, group in frame.groupby("survey", sort=True):
        paired = group.dropna(subset=["naturalness_human", "naturalness_llm"]).copy()
        paired["delta"] = paired["naturalness_human"] - paired["naturalness_llm"]
        overall_rows.append(_contrast_row(survey, "all", paired))
        for order, part in paired.groupby("trans_cond", sort=True):
            overall_rows.append(_contrast_row(survey, str(order), part))
        for scale_id, part in paired.groupby("scale_id", sort=True):
            scale_rows.append(
                {
                    "survey": survey,
                    "scale_id": scale_id,
                    "scale_name": part["scale_name"].iloc[0],
                    "n": int(len(part)),
                    "mean_human": float(part["naturalness_human"].mean()),
                    "mean_llm": float(part["naturalness_llm"].mean()),
                    "mean_delta": float(part["delta"].mean()),
                    "ci_low": _mean_ci(part["delta"])[0],
                    "ci_high": _mean_ci(part["delta"])[1],
                    "share_human_higher": float((part["delta"] > 0).mean()),
                    "share_tie": float((part["delta"] == 0).mean()),
                    "share_llm_higher": float((part["delta"] < 0).mean()),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(scale_rows)


def choice_association(frame: pd.DataFrame, similarity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Relate between-version similarity to preference and the LLM guess."""

    merged = frame.merge(similarity, on="scale_id", how="left")
    scale_rows = []
    association_rows = []
    model_rows = []
    for survey, group in merged.groupby("survey", sort=True):
        for outcome, column in (("prefer_human", "prefer_human"), ("guess_correct", "guess_correct")):
            model_rows.append(_choice_model(survey, outcome, group, column))
            for order, part in [("all", group), *group.groupby("trans_cond", sort=True)]:
                order_name = order if isinstance(order, str) else str(order)
                rates = _scale_rates(part, column)
                rates.insert(0, "survey", survey)
                rates.insert(1, "outcome", outcome)
                rates.insert(2, "trans_cond", order_name)
                scale_rows.append(rates)
                association_rows.append(_spearman_row(survey, outcome, order_name, rates))
    scales = pd.concat(scale_rows, ignore_index=True)
    return scales, pd.DataFrame(association_rows), pd.DataFrame(model_rows)


def run(
    root: Path,
    *,
    processed: Path | None = None,
    out: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Write the judgment tables, figures, and markdown report."""

    catalog = root / "data/raw/scales.json"
    processed = processed or root / "data/survey/3_processed"
    citizen_qsf = root / "data/survey/questionnaires/2026_ManyScales_一般（調査票）.qsf"
    expert_qsf = root / "data/survey/questionnaires/2026_ManyScales_専門（調査票）.qsf"
    embeddings = root / "results/sentence_embeddings/mpnet.parquet"
    out = out or root / "results/survey_judgment"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    citizen_audit = comparison_column_audit(citizen_qsf, catalog)
    expert_audit = comparison_column_audit(expert_qsf, catalog)
    citizen_audit.insert(0, "survey", "citizen")
    expert_audit.insert(0, "survey", "expert")
    audit = pd.concat([citizen_audit, expert_audit], ignore_index=True)
    keep = citizen_audit.loc[:, ["scale_id", "item_id", "in_comparison_table"]]
    _items, similarity = between_version_similarity(embeddings, keep)
    judgments = load_judgments(processed, catalog)
    naturalness, naturalness_scales = naturalness_contrast(judgments)
    choice_scales, associations, choice_models = choice_association(judgments, similarity)
    order_counts = _order_counts(root, judgments)

    audit.to_csv(table_dir / "comparison_column_audit.csv", index=False)
    similarity.to_csv(table_dir / "scale_similarity_ja.csv", index=False)
    judgments.to_csv(table_dir / "judgments_long.csv", index=False)
    naturalness.to_csv(table_dir / "naturalness_contrast.csv", index=False)
    naturalness_scales.to_csv(table_dir / "naturalness_by_scale.csv", index=False)
    choice_scales.to_csv(table_dir / "choice_by_scale_condition.csv", index=False)
    associations.to_csv(table_dir / "similarity_association.csv", index=False)
    choice_models.to_csv(table_dir / "choice_models.csv", index=False)
    pd.DataFrame(order_counts).to_csv(table_dir / "trans_cond_counts.csv", index=False)
    _plot_naturalness(naturalness_scales, figure_dir / "naturalness_by_scale.png")
    _plot_association(choice_scales, similarity, figure_dir / "similarity_and_choice.png")
    report = _report(
        audit=audit,
        similarity=similarity,
        naturalness=naturalness,
        naturalness_scales=naturalness_scales,
        associations=associations,
        choice_models=choice_models,
        order_counts=order_counts,
        judgments=judgments,
    )
    report_path = report_path or root / "protocl/h4_judgment_report.md"
    report_path.write_text(report)
    return {
        "naturalness": naturalness,
        "associations": associations,
        "report": report_path,
    }


def _survey_judgments(processed_dir: Path, survey: str, rating_stem: str) -> pd.DataFrame:
    from h4.survey.eligibility import ocq_nonconsumer
    prefix = "ManyScales_市民_processed_" if survey == "citizen" else "ManyScales_専門家_processed_"
    rows = []
    for set_id in _SETS:
        frame = pd.read_csv(processed_dir / f"{prefix}{set_id}.csv", dtype=str, keep_default_na=False)
        frame = frame.replace({"": pd.NA, "NA": pd.NA})
        for column in frame.columns:
            match = re.fullmatch(rf"{set_id}(\d{{2}})_{rating_stem}_human", column)
            if match is None:
                continue
            scale_id = match.group(1)
            stub = f"{set_id}{scale_id}"
            for index, record in frame.iterrows():
                if survey == "citizen" and scale_id == "08" and ocq_nonconsumer(frame).loc[index]:
                    continue
                rows.append(
                    {
                        "survey": survey,
                        "respondent_id": f"{survey}:{set_id}:{index}",
                        "set_id": set_id,
                        "scale_id": scale_id,
                        "trans_cond": record.get("trans_cond"),
                        "naturalness_human": _score(record.get(f"{stub}_{rating_stem}_human")),
                        "naturalness_llm": _score(record.get(f"{stub}_{rating_stem}_ai")),
                        "prefer_human": decode_preference(record.get(f"{stub}_preference")),
                        "guess_correct": decode_guess(record.get(f"{stub}_llm_guess")),
                    }
                )
    return pd.DataFrame(rows)


def _contrast_row(survey: str, order: str, paired: pd.DataFrame) -> dict[str, Any]:
    fit = fit_hierarchical(paired.loc[:, ["delta", "scale_id"]])
    person = smf.ols("delta ~ 1", data=paired).fit(
        cov_type="cluster",
        cov_kwds={"groups": paired["respondent_id"]},
    )
    person_ci = person.conf_int().iloc[0].tolist()
    order_effect = {}
    if order == "all" and paired["trans_cond"].nunique() > 1:
        coded = paired.copy()
        coded["ai_first"] = (coded["trans_cond"] == "ai").astype(float)
        mixed = smf.mixedlm("delta ~ ai_first", coded, groups=coded["scale_id"]).fit(reml=True, method="lbfgs", disp=False)
        order_effect = {
            "order_estimate": float(np.asarray(mixed.fe_params)[1]),
            "order_se": float(np.asarray(mixed.bse_fe)[1]),
            "order_ci_low": float(mixed.conf_int().iloc[1, 0]),
            "order_ci_high": float(mixed.conf_int().iloc[1, 1]),
            "order_p": float(np.asarray(mixed.pvalues)[1]),
        }
    return {
        "survey": survey,
        "trans_cond": order,
        "n_pairs": int(len(paired)),
        "n_respondents": int(paired["respondent_id"].nunique()),
        "n_scales": int(paired["scale_id"].nunique()),
        "mean_human": float(paired["naturalness_human"].mean()),
        "mean_llm": float(paired["naturalness_llm"].mean()),
        "item_mean_delta": fit["item_mean"],
        "scale_mean_delta": fit["unweighted_scale_mean"],
        "estimate": fit["estimate"],
        "se": fit["se"],
        "ci_low": fit["ci_low"],
        "ci_high": fit["ci_high"],
        "icc": fit["icc"],
        "person_cluster_estimate": float(person.params.iloc[0]),
        "person_cluster_ci_low": float(person_ci[0]),
        "person_cluster_ci_high": float(person_ci[1]),
        "share_human_higher": float((paired["delta"] > 0).mean()),
        "share_tie": float((paired["delta"] == 0).mean()),
        "share_llm_higher": float((paired["delta"] < 0).mean()),
        "convergence_warning": fit["convergence_warning"],
        **order_effect,
    }


def _choice_model(survey: str, outcome: str, group: pd.DataFrame, column: str) -> dict[str, Any]:
    data = group.dropna(subset=[column, "similarity_ja", "trans_cond"]).copy()
    data["y"] = data[column]
    data["ai_first"] = (data["trans_cond"] == "ai").astype(float)
    center = float(data.groupby("scale_id")["similarity_ja"].first().mean())
    spread = float(data.groupby("scale_id")["similarity_ja"].first().std(ddof=1))
    data["sim_z"] = (data["similarity_ja"] - center) / spread
    fit = smf.glm("y ~ sim_z + ai_first", data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["scale_id"]},
    )
    ci = fit.conf_int()
    return {
        "survey": survey,
        "outcome": outcome,
        "n": int(len(data)),
        "n_scales": int(data["scale_id"].nunique()),
        "similarity_mean": center,
        "similarity_sd": spread,
        "sim_logit": float(fit.params["sim_z"]),
        "sim_se": float(fit.bse["sim_z"]),
        "sim_ci_low": float(ci.loc["sim_z", 0]),
        "sim_ci_high": float(ci.loc["sim_z", 1]),
        "sim_p": float(fit.pvalues["sim_z"]),
        "ai_first_logit": float(fit.params["ai_first"]),
        "ai_first_se": float(fit.bse["ai_first"]),
        "ai_first_p": float(fit.pvalues["ai_first"]),
        "rate": float(data["y"].mean()),
    }


def _scale_rates(part: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for scale_id, group in part.groupby("scale_id", sort=True):
        observed = group.dropna(subset=[column])
        successes = float(observed[column].sum()) if len(observed) else float("nan")
        n = int(len(observed))
        low, high = (float("nan"), float("nan")) if n == 0 else proportion_confint(successes, n, method="wilson")
        rows.append(
            {
                "scale_id": scale_id,
                "scale_name": group["scale_name"].iloc[0],
                "similarity_ja": float(group["similarity_ja"].iloc[0]),
                "n": n,
                "rate": float(observed[column].mean()) if n else float("nan"),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def _spearman_row(survey: str, outcome: str, order: str, rates: pd.DataFrame) -> dict[str, Any]:
    used = rates.dropna(subset=["similarity_ja", "rate"])
    used = used.loc[used["n"] > 0]
    if len(used) < 3 or used["similarity_ja"].nunique() < 2:
        rho, p_value = float("nan"), float("nan")
    else:
        rho, p_value = stats.spearmanr(used["similarity_ja"], used["rate"])
    return {
        "survey": survey,
        "outcome": outcome,
        "trans_cond": order,
        "n_scales": int(len(used)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "mean_rate": float(np.average(used["rate"], weights=used["n"])) if len(used) else float("nan"),
        "unweighted_mean_rate": float(used["rate"].mean()) if len(used) else float("nan"),
    }


def _order_counts(root: Path, judgments: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    raw = {
        "citizen": root.parent / "ratings/primary/data/0_raw_data/2026_ManyScales_一般_2026年7月29日_06.39.tsv",
        "expert": root.parent / "ratings/primary/data/0_raw_data/2026_ManyScales_専門_2026年9月16日_08.46.tsv",
    }
    for survey, path in raw.items():
        frame, _header = read_qualtrics_tsv(path)
        counts = frame["trans_cond"].value_counts(dropna=False).to_dict()
        rows.append({"survey": survey, "table": "raw", "n": int(len(frame)), **{f"n_{key}": int(value) for key, value in counts.items()}})
    people = judgments.drop_duplicates(["survey", "respondent_id"])
    for survey, group in people.groupby("survey"):
        counts = group["trans_cond"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "survey": survey,
                "table": "processed",
                "n": int(len(group)),
                **{f"n_{key}": int(value) for key, value in counts.items()},
            }
        )
    return rows


def _plot_naturalness(scales: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 8.2), sharex=True)
    for axis, survey, title in zip(axes, ("citizen", "expert"), ("Citizens, wording", "Experts, naturalness"), strict=True):
        part = scales.loc[scales["survey"] == survey].sort_values("mean_delta")
        positions = np.arange(len(part))
        axis.axvline(0, color="#666666", linewidth=0.8)
        axis.errorbar(
            part["mean_delta"],
            positions,
            xerr=[part["mean_delta"] - part["ci_low"], part["ci_high"] - part["mean_delta"]],
            fmt="o",
            color="#2c7fb8",
            ecolor="#9ecae1",
            elinewidth=1.2,
            markersize=4.5,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(part["scale_name"])
        axis.set_title(title)
        axis.set_xlabel("Human rating − LLM rating")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_association(choice_scales: pd.DataFrame, similarity: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True)
    colors = {"human": "#2c7fb8", "ai": "#d95f0e"}
    titles = {
        ("citizen", "prefer_human"): "Citizens, prefer human",
        ("citizen", "guess_correct"): "Citizens, identify the LLM version",
        ("expert", "prefer_human"): "Experts, prefer human",
        ("expert", "guess_correct"): "Experts, identify the LLM version",
    }
    for row, survey in enumerate(("citizen", "expert")):
        for column, outcome in enumerate(("prefer_human", "guess_correct")):
            axis = axes[row, column]
            part = choice_scales.loc[
                (choice_scales["survey"] == survey)
                & (choice_scales["outcome"] == outcome)
                & (choice_scales["trans_cond"].isin(["human", "ai"]))
            ]
            for order, color in colors.items():
                points = part.loc[part["trans_cond"] == order]
                axis.scatter(points["similarity_ja"], points["rate"], s=28 + points["n"] * 0.15, color=color, label=f"{order} first", alpha=0.9)
            axis.set_ylim(-0.05, 1.05)
            axis.axhline(0.5, color="#bbbbbb", linewidth=0.8)
            axis.set_title(titles[(survey, outcome)])
            if row == 1:
                axis.set_xlabel("Mean cosine of the two Japanese versions")
            if column == 0:
                axis.set_ylabel("Proportion")
            if row == 0 and column == 1:
                axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(
    *,
    audit: pd.DataFrame,
    similarity: pd.DataFrame,
    naturalness: pd.DataFrame,
    naturalness_scales: pd.DataFrame,
    associations: pd.DataFrame,
    choice_models: pd.DataFrame,
    order_counts: list[dict[str, Any]],
    judgments: pd.DataFrame,
) -> str:
    swapped = int(audit["swapped"].sum())
    matched = int(audit.loc[audit["survey"] == "citizen", "slot1_is_human"].sum())
    omitted = int((~audit.loc[audit["survey"] == "citizen", "in_comparison_table"]).sum())
    lines = [
        "# Survey judgments: naturalness, preference, and the LLM guess",
        "",
        "This note uses the two-rule citizen analysis tables: assigned items completed, attention passed, and speeding passed. Straight-lining is not applied because presentation order was randomized and was not stored. The earlier three-rule cut is a sensitivity analysis. Experts are unchanged. The confirmatory cosine estimate is unchanged.",
        "",
        "## Which label records the translation the respondent saw",
        "",
        "Two labels are stored, and they answer different questions.",
        "",
        "`trans_cond` is embedded data written by the survey flow before the items. Its values are `human` and `ai`. It names the version administered first. The other version is administered second. The same column is in the raw Qualtrics export and in every processed set table.",
        "",
        _count_sentence(order_counts),
        "",
        "Preference and the LLM guess are coded `1` or `2`. Those codes name columns in the comparison table, headed 翻訳1 and 翻訳2. In both the citizen survey and the expert survey that table is written into the questionnaire. The survey script does not swap the columns. Across the study items, 翻訳1 matches the human Japanese workbook text and 翻訳2 matches the LLM Japanese workbook text "
        f"({matched} items in each survey; swapped columns: {swapped}). "
        f"Three SCSS attention items are absent from the comparison table ({omitted} citizen items), so the similarity below uses the items that table shows.",
        "",
        "The coding used here is therefore: preference `1` prefers the human version and `2` prefers the LLM version; LLM-guess `2` correctly names the LLM column and `1` names the human column. Naturalness does not use this slot coding. Citizen `wording_human` / `wording_ai` and expert `naturalness_human` / `naturalness_ai` are already labeled with the true version.",
        "",
        "## Is the LLM version as natural as the human version?",
        "",
        "Each respondent rated both versions of each assigned scale. The contrast is the human rating minus the LLM rating on the shared 1–5 scale (higher means clearer or more natural). The estimate is a random-intercept model, `delta ~ 1`, groups = `scale_id`, REML. A second interval clusters respondents, because one person rates every scale in the assigned set.",
        "",
        _naturalness_table(naturalness),
        "",
        _naturalness_prose(naturalness),
        "",
        "A positive estimate means the human version was rated higher. The order term is the AI-first minus human-first difference in that gap, from `delta ~ ai_first` with the same scale intercept. The random-intercept variance sits on the boundary in these fits, so the person-cluster interval is the check that keeps respondents who rated several scales from being treated as independent.",
        "",
        "![Human minus LLM naturalness by scale. Points are respondent means and bars are 95% intervals of those means.](../results/survey_judgment_n1859/figures/naturalness_by_scale.png)",
        "",
        "## Does similarity of the two versions shift preference and the LLM guess?",
        "",
        "Similarity is the mean item cosine, under the frozen MPNet model, between `human_ja` and `llm_ja` for the items shown in the comparison table. It is a property of the scale, so the association is across the 22 scales. Within each administration order the scale proportion is correlated with that similarity (Spearman). A binomial model then pools respondents, with the similarity standardized across scales and a cluster-robust standard error by `scale_id`. The order covariate is AI presented first.",
        "",
        f"Scale cosine ranges from {similarity['similarity_ja'].min():.3f} to {similarity['similarity_ja'].max():.3f} (mean {similarity['similarity_ja'].mean():.3f}).",
        "",
        _association_table(associations),
        "",
        _model_table(choice_models),
        "",
        _choice_prose(associations, choice_models),
        "",
        "![Scale similarity of the two Japanese versions against the proportion who prefer the human version and the proportion who identify the LLM version. Blue is human-first; orange is AI-first. Marker area follows the number of respondents.](../results/survey_judgment_n1859/figures/similarity_and_choice.png)",
        "",
        "## Sample entering these tests",
        "",
        _sample_sentence(judgments),
        "",
        "Scale 10 is pooled across sets C and D. Rows with a blank rating or a blank choice are omitted from that contrast only.",
        "",
        "Tables: `results/survey_judgment_n1859/tables/`. The earlier three-rule tables remain in `results/survey_judgment/tables/`.",
        "",
    ]
    return "\n".join(lines)


def _naturalness_prose(frame: pd.DataFrame) -> str:
    def row(survey: str, order: str) -> pd.Series:
        return frame.loc[(frame["survey"] == survey) & (frame["trans_cond"] == order)].iloc[0]

    citizen = row("citizen", "all")
    expert = row("expert", "all")
    return (
        f"Citizens rate the human wording higher by {citizen['estimate']:.3f} points "
        f"(95% CI [{citizen['ci_low']:.3f}, {citizen['ci_high']:.3f}]; person-cluster "
        f"[{citizen['person_cluster_ci_low']:.3f}, {citizen['person_cluster_ci_high']:.3f}]). "
        f"The means are {citizen['mean_human']:.3f} and {citizen['mean_llm']:.3f}, and "
        f"{citizen['share_tie']:.0%} of pairs are ties. "
        f"The gap is {row('citizen', 'ai')['estimate']:.3f} when the LLM version is answered first and "
        f"{row('citizen', 'human')['estimate']:.3f} when the human version is answered first "
        f"(AI-first shift {citizen['order_estimate']:.3f}, 95% CI [{citizen['order_ci_low']:.3f}, {citizen['order_ci_high']:.3f}]). "
        f"Experts show an overall gap of {expert['estimate']:.3f} "
        f"(95% CI [{expert['ci_low']:.3f}, {expert['ci_high']:.3f}]; person-cluster "
        f"[{expert['person_cluster_ci_low']:.3f}, {expert['person_cluster_ci_high']:.3f}]), "
        f"with means {expert['mean_human']:.3f} and {expert['mean_llm']:.3f}. "
        f"That overall interval includes 0. Split by order, the expert gap is "
        f"{row('expert', 'ai')['estimate']:.3f} after the LLM version and "
        f"{row('expert', 'human')['estimate']:.3f} after the human version "
        f"(AI-first shift {expert['order_estimate']:.3f}, 95% CI [{expert['order_ci_low']:.3f}, {expert['order_ci_high']:.3f}])."
    )


def _choice_prose(associations: pd.DataFrame, models: pd.DataFrame) -> str:
    def rho(survey: str, outcome: str, order: str) -> pd.Series:
        return associations.loc[
            (associations["survey"] == survey) & (associations["outcome"] == outcome) & (associations["trans_cond"] == order)
        ].iloc[0]

    def model(survey: str, outcome: str) -> pd.Series:
        return models.loc[(models["survey"] == survey) & (models["outcome"] == outcome)].iloc[0]

    citizen_pref = rho("citizen", "prefer_human", "all")
    citizen_guess = rho("citizen", "guess_correct", "all")
    expert_pref = model("expert", "prefer_human")
    citizen_pref_model = model("citizen", "prefer_human")
    citizen_guess_model = model("citizen", "guess_correct")
    human_first = rho("expert", "prefer_human", "human")
    ai_first = rho("expert", "prefer_human", "ai")
    return (
        f"Citizens prefer the human column on {citizen_pref['mean_rate']:.1%} of scale judgments and name the LLM column on {citizen_guess['mean_rate']:.1%}. "
        f"Across the 22 scales, higher cosine between the two Japanese versions goes with a lower human-preference rate "
        f"(Spearman {citizen_pref['spearman_rho']:.3f}, p {citizen_pref['spearman_p']:.3f}) and a lower rate of identifying the LLM version "
        f"(Spearman {citizen_guess['spearman_rho']:.3f}, p {citizen_guess['spearman_p']:.3f}). "
        f"The same negative sign appears inside each administration order. "
        f"The cluster-robust logit, which gives the similarity coefficient a standard error based on the 22 scales, excludes 0 for both outcomes "
        f"(preference {citizen_pref_model['sim_logit']:.3f} [{citizen_pref_model['sim_ci_low']:.3f}, {citizen_pref_model['sim_ci_high']:.3f}], p {citizen_pref_model['sim_p']:.3f}; "
        f"identification {citizen_guess_model['sim_logit']:.3f} [{citizen_guess_model['sim_ci_low']:.3f}, {citizen_guess_model['sim_ci_high']:.3f}], p {citizen_guess_model['sim_p']:.3f}). "
        f"Among experts the similarity correlations are weak in both orders. "
        f"Administration order does shift expert preference: the human column is chosen on {human_first['mean_rate']:.1%} of human-first judgments and {ai_first['mean_rate']:.1%} of AI-first judgments "
        f"(AI-first logit {expert_pref['ai_first_logit']:.3f}, p {expert_pref['ai_first_p']:.3f}). "
        f"Experts name the LLM column on {model('expert', 'guess_correct')['rate']:.1%} of judgments."
    )


def _count_sentence(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        human = int(row.get("n_human", 0) or 0)
        ai = int(row.get("n_ai", 0) or 0)
        missing = int(row.get("n_nan", 0) or 0)
        extra = f", unlabeled {missing}" if missing else ""
        parts.append(f"{row['survey']} {row['table']}: {row['n']} respondents, human-first {human}, AI-first {ai}{extra}")
    return "Counts of `trans_cond`: " + "; ".join(parts) + "."


def _naturalness_table(frame: pd.DataFrame) -> str:
    header = "| Survey | Order | Pairs | Mean human | Mean LLM | Estimate | 95% CI | Person-cluster CI | Human higher | Tie | LLM higher |"
    rule = "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |"
    body = []
    for _, row in frame.sort_values(["survey", "trans_cond"]).iterrows():
        body.append(
            "| {survey} | {trans_cond} | {n_pairs} | {mean_human:.3f} | {mean_llm:.3f} | {estimate:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | [{person_cluster_ci_low:.3f}, {person_cluster_ci_high:.3f}] | {share_human_higher:.3f} | {share_tie:.3f} | {share_llm_higher:.3f} |".format(
                **row
            )
        )
    order_lines = []
    for _, row in frame.loc[frame["trans_cond"] == "all"].iterrows():
        if "order_estimate" not in row or pd.isna(row.get("order_estimate")):
            continue
        order_lines.append(
            f"{row['survey']}: AI-first shift {row['order_estimate']:.3f} [{row['order_ci_low']:.3f}, {row['order_ci_high']:.3f}], p {_p(row['order_p'])}."
        )
    return "\n".join([header, rule, *body, "", *order_lines])


def _association_table(frame: pd.DataFrame) -> str:
    header = "| Survey | Outcome | Order | Scales | Spearman rho | p | Weighted rate |"
    rule = "| --- | --- | --- | ---: | ---: | ---: | ---: |"
    body = []
    order = {"all": 0, "human": 1, "ai": 2}
    shown = frame.sort_values(["survey", "outcome", "trans_cond"], key=lambda series: series.map(order) if series.name == "trans_cond" else series)
    for _, row in shown.iterrows():
        body.append(
            f"| {row['survey']} | {row['outcome']} | {row['trans_cond']} | {row['n_scales']} | {row['spearman_rho']:.3f} | {row['spearman_p']:.3f} | {row['mean_rate']:.3f} |"
        )
    return "\n".join([header, rule, *body])


def _model_table(frame: pd.DataFrame) -> str:
    header = "| Survey | Outcome | Rate | Similarity logit per SD | 95% CI | p | AI-first logit | p |"
    rule = "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"
    body = []
    for _, row in frame.sort_values(["survey", "outcome"]).iterrows():
        body.append(
            f"| {row['survey']} | {row['outcome']} | {row['rate']:.3f} | {row['sim_logit']:.3f} | [{row['sim_ci_low']:.3f}, {row['sim_ci_high']:.3f}] | {row['sim_p']:.3f} | {row['ai_first_logit']:.3f} | {row['ai_first_p']:.3f} |"
        )
    return "\n".join([header, rule, *body])


def _sample_sentence(frame: pd.DataFrame) -> str:
    parts = []
    people = frame.drop_duplicates(["survey", "respondent_id"])
    for survey, group in people.groupby("survey"):
        parts.append(f"{survey} {len(group)} respondents")
    pairs = frame.dropna(subset=["naturalness_human", "naturalness_llm"])
    pair_bits = [f"{survey} {len(group)} paired ratings" for survey, group in pairs.groupby("survey")]
    return "Analysis tables contain " + " and ".join(parts) + ". Paired naturalness ratings: " + " and ".join(pair_bits) + "."


def _scale_names(catalog_path: Path) -> dict[str, str]:
    catalog = load_catalog(catalog_path)
    return {
        str(scale["scale_id"]): str(scale.get("short_name") or scale["scale_id"])
        for scale in catalog["scales"]
        if scale.get("status") == "study"
    }


def _scale_key(value: object) -> str:
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() else text


def _score(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if text not in {"1", "2", "3", "4", "5"}:
        return float("nan")
    return float(text)


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    n = len(clean)
    if n < 2:
        mean = float(clean.mean()) if n else float("nan")
        return mean, mean
    mean = float(clean.mean())
    half = float(stats.t.ppf(0.975, n - 1) * clean.std(ddof=1) / np.sqrt(n))
    return mean - half, mean + half


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
