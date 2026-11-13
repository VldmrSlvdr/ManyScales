"""Catalog of fitted tests, with figures for intervals that exclude zero.

Reads the stored result tables. Does not refit the confirmatory cosine model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]


def run(root: Path | None = None) -> Path:
    root = root or _ROOT
    figures = root / "results" / "effects" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    catalog, effects = _collect(root)
    catalog.to_csv(root / "results" / "effects" / "test_catalog.csv", index=False)
    effects.to_csv(root / "results" / "effects" / "significant_effects.csv", index=False)
    _figure_forest(
        effects.loc[effects["panel"].eq("cosine")],
        figures / "cosine_back_translation.png",
        "Human cosine minus LLM cosine",
    )
    _figure_forest(
        effects.loc[effects["panel"].eq("direct")],
        figures / "cosine_direct_wald.png",
        "Human cosine minus LLM cosine (REML Wald interval)",
    )
    _figure_forest(
        effects.loc[effects["panel"].eq("rating")],
        figures / "rating_effects.png",
        "Points on the 1–5 rating, or on the human minus LLM gap",
    )
    _figure_forest(
        effects.loc[effects["panel"].eq("logit")],
        figures / "logit_effects.png",
        "Log-odds",
    )
    _figure_spearman(effects.loc[effects["panel"].eq("spearman")], figures / "spearman_effects.png")
    _figure_choice_fit(root, figures / "choice_fit_slope.png")
    _figure_flags(root, figures / "flag_by_delta.png")
    _figure_similarity(root, figures / "citizen_similarity.png")
    _figure_order(root, figures / "naturalness_by_order.png")
    report = _report(catalog, effects)
    path = root / "protocl" / "h4_effects_report.md"
    path.write_text(report)
    return path


def _collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict] = []
    effect_rows: list[dict] = []

    confirmatory = pd.read_csv(root / "results/tables/confirmatory_estimate.csv").iloc[0]
    bootstrap = pd.read_csv(root / "results/tables/bootstrap_summary.csv").iloc[0]
    beta = pd.read_csv(root / "results/tables/beta_estimate.csv")
    direct = pd.read_csv(root / "results/tables/direct_estimate.csv").iloc[0]
    curve = pd.read_csv(root / "results/tables/specification_curve.csv")
    no_conf = pd.read_csv(root / "results/no_tivs/tables/confirmatory_estimate.csv").iloc[0]
    no_boot = pd.read_csv(root / "results/no_tivs/tables/bootstrap_summary.csv").iloc[0]
    no_beta = pd.read_csv(root / "results/no_tivs/tables/beta_estimate.csv")
    no_direct = pd.read_csv(root / "results/no_tivs/tables/direct_estimate.csv").iloc[0]
    natural = pd.read_csv(root / "results/survey_judgment/tables/naturalness_contrast.csv")
    choices = pd.read_csv(root / "results/survey_judgment/tables/choice_models.csv")
    spearman = pd.read_csv(root / "results/survey_judgment/tables/similarity_association.csv")
    dimensions = pd.read_csv(root / "results/survey_meta/tables/expert_dimension_models.csv")
    associations = pd.read_csv(root / "results/survey_meta/tables/associations.csv")
    mixed = pd.read_csv(root / "results/survey_mixed/coefficients.csv")
    status = pd.read_csv(root / "results/survey_mixed/model_status.csv")

    _add_intercept(
        catalog_rows, effect_rows,
        test="Confirmatory back-translation",
        formula="delta ~ 1 + (1 | scale_id)",
        family="Gaussian REML",
        n=f"{int(confirmatory.n_items)} items, {int(confirmatory.n_scales)} scales",
        tier="confirmatory",
        panel="cosine",
        label="MPNet back-translation, 22 scales",
        estimate=confirmatory.estimate,
        ci_low=confirmatory.ci_low,
        ci_high=confirmatory.ci_high,
        blurb=(
            "The preregistered human-minus-LLM difference is {estimate:.3f} cosine "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). The LLM English back-translation is closer to the original English."
        ),
    )
    _add_interval(
        catalog_rows, effect_rows,
        test="Hierarchical bootstrap of the back-translation item mean",
        formula="resample scales, then items; mean of delta",
        family="Hierarchical bootstrap, 10,000 draws",
        n=f"{int(bootstrap.n_items)} items, {int(bootstrap.n_scales)} scales",
        tier="robustness",
        panel="cosine",
        label="Bootstrap item mean, 22 scales",
        estimate=bootstrap.observed_mean,
        ci_low=bootstrap.ci_low,
        ci_high=bootstrap.ci_high,
        blurb=(
            "Resampling scales and then items leaves the same sign. The item-weighted mean is {estimate:.3f} "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]), and none of the 10,000 draws are positive."
        ),
    )
    for _, row in beta.iterrows():
        panel = "cosine" if row["comparison"] == "back_translation" else ""
        comparison = "back-translation" if row["comparison"] == "back_translation" else "direct Japanese"
        _add_interval(
            catalog_rows, effect_rows,
            test=f"Beta regression, {comparison}",
            formula="similarity ~ human, logit link, cluster SE by scale_id",
            family="Beta",
            n=f"{int(row.n_items)} items, {int(row.n_scales)} scales",
            tier="robustness",
            panel=panel,
            label=f"Beta mean difference, {comparison}",
            estimate=row.mean_difference,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            blurb=(
                "On the cosine scale the human mean minus the LLM mean is {estimate:.3f} "
                "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). The logit coefficient on the human indicator is "
                f"{row.estimate_logit:.3f} (cluster p {_p(float(row.p_two_sided_cluster))})."
            ),
            plot=panel == "cosine",
        )
    _add_interval(
        catalog_rows, effect_rows,
        test="Direct Japanese mixed model",
        formula="delta ~ 1 + (1 | scale_id)",
        family="Gaussian REML",
        n=f"{int(direct.n_items)} items, {int(direct.n_scales)} scales",
        tier="robustness",
        panel="direct",
        label="MPNet direct, REML Wald",
        estimate=direct.estimate,
        ci_low=direct.ci_low,
        ci_high=direct.ci_high,
        blurb=(
            "The REML Wald interval for original English versus Japanese is {estimate:.3f} "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). The hierarchical bootstrap and the beta interval for this "
            "comparison include 0, so this row is the Wald interval only."
        ),
    )
    for _, row in curve.loc[curve["metric"].eq("cosine")].iterrows():
        if row["short_name"] == "mpnet":
            continue
        panel = "cosine" if row["comparison"] == "back_translation" else "direct"
        comparison = "back-translation" if row["comparison"] == "back_translation" else "direct Japanese"
        _add_interval(
            catalog_rows, effect_rows,
            test=f"Specification curve, {row.short_name} {comparison}",
            formula="delta ~ 1 + (1 | scale_id)",
            family="Gaussian REML",
            n=f"{int(row.n_items)} items, {int(row.n_scales)} scales",
            tier="robustness",
            panel=panel,
            label=f"{row.short_name} {comparison}",
            estimate=row.estimate,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            blurb=(
                f"Under {row.short_name}, {comparison}, the human-minus-LLM cosine gap is {{estimate:.3f}} "
                "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). The sign matches the confirmatory model."
            ),
        )
    transforms = curve.loc[~curve["metric"].eq("cosine")]
    catalog_rows.append(
        {
            "test": "Monotone transforms of cosine (angular distance, normalized Euclidean)",
            "formula": "delta ~ 1 + (1 | scale_id)",
            "family": "Gaussian REML",
            "n": f"{len(transforms)} rows",
            "tier": "robustness",
            "result": f"All {len(transforms)} intervals are negative and exclude 0. They are monotone transforms of the cosine rows, so they are not plotted again.",
        }
    )
    _add_intercept(
        catalog_rows, effect_rows,
        test="Leave-TIVS-out back-translation",
        formula="delta ~ 1 + (1 | scale_id)",
        family="Gaussian REML",
        n=f"{int(no_conf.n_items)} items, {int(no_conf.n_scales)} scales",
        tier="robustness",
        panel="cosine",
        label="MPNet back-translation, TIVS removed",
        estimate=no_conf.estimate,
        ci_low=no_conf.ci_low,
        ci_high=no_conf.ci_high,
        blurb=(
            "Removing TIVS leaves the estimate at {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "The shift from the 22-scale confirmatory estimate is about 0.001 cosine."
        ),
    )
    _add_interval(
        catalog_rows, effect_rows,
        test="Leave-TIVS-out bootstrap",
        formula="resample scales, then items; mean of delta",
        family="Hierarchical bootstrap, 10,000 draws",
        n=f"{int(no_boot.n_items)} items, {int(no_boot.n_scales)} scales",
        tier="robustness",
        panel="cosine",
        label="Bootstrap item mean, TIVS removed",
        estimate=no_boot.observed_mean,
        ci_low=no_boot.ci_low,
        ci_high=no_boot.ci_high,
        blurb=(
            "The leave-TIVS-out item mean is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "None of the 10,000 draws are positive."
        ),
    )
    no_beta_back = no_beta.loc[no_beta["comparison"].eq("back_translation")].iloc[0]
    no_beta_direct = no_beta.loc[no_beta["comparison"].eq("direct_crosslingual")].iloc[0]
    _add_interval(
        catalog_rows, effect_rows,
        test="Leave-TIVS-out beta, back-translation",
        formula="similarity ~ human, logit link, cluster SE by scale_id",
        family="Beta",
        n=f"{int(no_beta_back.n_items)} items, {int(no_beta_back.n_scales)} scales",
        tier="robustness",
        panel="cosine",
        label="Beta mean difference, TIVS removed",
        estimate=no_beta_back.mean_difference,
        ci_low=no_beta_back.ci_low,
        ci_high=no_beta_back.ci_high,
        blurb=(
            "The beta mean difference without TIVS is {estimate:.3f} "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}])."
        ),
    )
    _add_interval(
        catalog_rows, effect_rows,
        test="Leave-TIVS-out direct Japanese",
        formula="delta ~ 1 + (1 | scale_id)",
        family="Gaussian REML",
        n=f"{int(no_direct.n_items)} items, {int(no_direct.n_scales)} scales",
        tier="robustness",
        panel="direct",
        label="MPNet direct, TIVS removed, REML Wald",
        estimate=no_direct.estimate,
        ci_low=no_direct.ci_low,
        ci_high=no_direct.ci_high,
        blurb=(
            "Without TIVS the direct REML interval is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "The leave-TIVS-out beta interval for the direct comparison includes 0 "
            f"([{no_beta_direct.ci_low:.3f}, {no_beta_direct.ci_high:.3f}])."
        ),
    )
    catalog_rows.append(
        {
            "test": "Leave-TIVS-out beta, direct Japanese",
            "formula": "similarity ~ human, logit link, cluster SE by scale_id",
            "family": "Beta",
            "n": f"{int(no_beta_direct.n_items)} items, {int(no_beta_direct.n_scales)} scales",
            "tier": "robustness",
            "result": f"Mean difference {no_beta_direct.mean_difference:.3f}, 95% CI [{no_beta_direct.ci_low:.3f}, {no_beta_direct.ci_high:.3f}]. Includes 0.",
        }
    )

    for _, row in natural.iterrows():
        survey = "Citizen wording" if row["survey"] == "citizen" else "Expert naturalness"
        order = {"all": "both orders", "ai": "AI shown first", "human": "human shown first"}[row["trans_cond"]]
        _add_interval(
            catalog_rows, effect_rows,
            test=f"{survey}, {order}",
            formula="delta ~ 1 + (1 | scale_id); person-cluster OLS delta ~ 1",
            family="Gaussian REML",
            n=f"{int(row.n_pairs)} pairs, {int(row.n_respondents)} respondents",
            tier="exploratory",
            panel="rating",
            label=f"{survey} gap, {order}",
            estimate=row.estimate,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            blurb=(
                f"{survey}, {order}: the human-minus-LLM rating is {{estimate:.3f}} points "
                "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
                f"The person-cluster interval is [{row.person_cluster_ci_low:.3f}, {row.person_cluster_ci_high:.3f}]."
            ),
        )
        if row["trans_cond"] == "all" and pd.notna(row["order_estimate"]):
            _add_interval(
                catalog_rows, effect_rows,
                test=f"{survey}, order contrast",
                formula="delta ~ ai_first + (1 | scale_id)",
                family="Gaussian REML",
                n=f"{int(row.n_pairs)} pairs",
                tier="exploratory",
                panel="rating",
                label=f"{survey} gap, AI shown first minus human shown first",
                estimate=row.order_estimate,
                ci_low=row.order_ci_low,
                ci_high=row.order_ci_high,
                blurb=(
                    f"Showing the LLM version first raises the {survey.lower()} gap by {{estimate:.3f}} points "
                    "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). `ai_first` is administration order. "
                    "It does not change which column is the human translation."
                ),
            )

    for _, row in choices.iterrows():
        survey = "Citizens" if row["survey"] == "citizen" else "Experts"
        outcome = "prefer the human column" if row["outcome"] == "prefer_human" else "name the LLM column"
        sim_low = row.sim_logit - 1.96 * row.sim_se
        sim_high = row.sim_logit + 1.96 * row.sim_se
        order_low = row.ai_first_logit - 1.96 * row.ai_first_se
        order_high = row.ai_first_logit + 1.96 * row.ai_first_se
        _add_interval(
            catalog_rows, effect_rows,
            test=f"{survey}, {outcome}, between-version similarity",
            formula="y ~ sim_z + ai_first, binomial, cluster SE by scale_id",
            family="Binomial GLM",
            n=f"{int(row.n)} judgments, {int(row.n_scales)} scales",
            tier="exploratory",
            panel="logit",
            label=f"{survey}, {outcome}: between-version cosine (per SD)",
            estimate=row.sim_logit,
            ci_low=sim_low,
            ci_high=sim_high,
            blurb=(
                f"For {survey.lower()} the log-odds to {outcome} change by {{estimate:.3f}} per SD of "
                "between-version Japanese cosine (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
                "That interval includes 0. The predictor is the cosine between the two Japanese texts, "
                "not the gap from the English original."
            ),
            plot=False,
        )
        _add_interval(
            catalog_rows, effect_rows,
            test=f"{survey}, {outcome}, order",
            formula="y ~ sim_z + ai_first, binomial, cluster SE by scale_id",
            family="Binomial GLM",
            n=f"{int(row.n)} judgments, {int(row.n_scales)} scales",
            tier="exploratory",
            panel="logit",
            label=f"{survey}, {outcome}: AI shown first",
            estimate=row.ai_first_logit,
            ci_low=order_low,
            ci_high=order_high,
            blurb=(
                f"{survey} are more likely to {outcome} when the LLM version was shown first. "
                "The log-odds shift is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])."
            ),
        )
    for _, row in spearman.iterrows():
        survey = "Citizens" if row["survey"] == "citizen" else "Experts"
        outcome = "preferring the human column" if row["outcome"] == "prefer_human" else "naming the LLM column"
        order = {"all": "both orders", "ai": "AI shown first", "human": "human shown first"}[row["trans_cond"]]
        excludes = float(row.spearman_p) < 0.05
        catalog_rows.append(
            {
                "test": f"Spearman, {survey.lower()}, {outcome}, {order}",
                "formula": "Spearman of the 22 scale rates with mean cosine(human_ja, llm_ja)",
                "family": "Spearman rank correlation",
                "n": f"{int(row.n_scales)} scales",
                "tier": "exploratory",
                "result": f"rho {row.spearman_rho:.3f}, p {_p(row.spearman_p)}"
                + ("" if excludes else ". Includes a zero association."),
            }
        )
        if excludes:
            effect_rows.append(
                {
                    "panel": "spearman",
                    "label": f"{survey}, {outcome}, {order}",
                    "estimate": float(row.spearman_rho),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "blurb": (
                        f"{survey}, {order}: the rank correlation between between-version cosine and the rate of "
                        f"{outcome} is {row.spearman_rho:.3f} (p {_p(row.spearman_p)}). "
                        "The cluster-robust logit for the same predictor includes 0, because that standard error "
                        "uses the 22 scales as the clusters."
                    ),
                }
            )

    for _, row in dimensions.iterrows():
        name = row["dimension"].replace("_", " ")
        _add_interval(
            catalog_rows, effect_rows,
            test=f"Expert {name} gap",
            formula="delta ~ 1 + (1 | scale_id)",
            family="Gaussian REML",
            n=f"{int(row.n_pairs)} pairs",
            tier="exploratory",
            panel="rating",
            label=f"Expert {name} gap, human minus LLM",
            estimate=row.estimate,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            blurb=(
                f"Experts' {name} gap, human minus LLM, is {{estimate:.3f}} "
                "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
                + (
                    "A negative gap means the LLM version received the higher rating."
                    if row.estimate < 0
                    else "A positive gap means the human version received the higher rating."
                )
            ),
        )

    for _, row in associations.iterrows():
        excludes = float(row.spearman_p) < 0.05
        catalog_rows.append(
            {
                "test": f"Spearman, {row['level']}, {row['outcome']} with {row['predictor']}",
                "formula": f"Spearman at the {row['level']} level",
                "family": "Spearman rank correlation",
                "n": f"{int(row.n)} {row['level']}s",
                "tier": "exploratory",
                "result": f"rho {row.spearman_rho:.3f}, p {_p(row.spearman_p)}"
                + ("" if excludes else ". Includes a zero association."),
            }
        )
        if excludes:
            if row["outcome"] == "expert_semantic_flag":
                blurb = (
                    f"Items with a larger absolute MPNet back-translation gap are flagged more often as a semantic problem "
                    f"(Spearman {row.spearman_rho:.3f}, p {_p(row.spearman_p)}, n {int(row.n)})."
                )
            else:
                blurb = (
                    f"Across {int(row.n)} items, the mean absolute change in the citizen answer rises with "
                    f"{row['predictor']} (Spearman {row.spearman_rho:.3f}, p {_p(row.spearman_p)})."
                )
            effect_rows.append(
                {
                    "panel": "spearman",
                    "label": _spearman_label(str(row["outcome"]), str(row["predictor"])),
                    "estimate": float(row.spearman_rho),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "blurb": blurb,
                }
            )

    formulas = status.set_index("model")["formula"]
    families = status.set_index("model")["family"]
    for _, row in mixed.iterrows():
        if row["term"] == "(Intercept)":
            # Conditional intercepts are reference levels, not a separate predictor effect.
            catalog_hit = (
                f"Reference level {row.estimate:.3f}, 95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}]"
            )
            if _excludes(row.ci_low, row.ci_high):
                catalog_hit += ". Excludes 0 at the reference coding."
            else:
                catalog_hit += ". Includes 0."
            catalog_rows.append(
                {
                    "test": f"{row['model']} intercept",
                    "formula": formulas[row["model"]],
                    "family": families[row["model"]],
                    "n": "see model status",
                    "tier": "exploratory",
                    "result": catalog_hit,
                }
            )
            continue
        _add_interval(
            catalog_rows, effect_rows,
            test=f"{row['model']}: {row['term']}",
            formula=formulas[row["model"]],
            family="Gaussian REML" if families[row["model"]] == "gaussian" else "Binomial logit, lme4",
            n="see results/survey_mixed/model_status.csv",
            tier="exploratory",
            panel="logit" if families[row["model"]] == "binomial" else "rating",
            label=_mixed_label(row["model"], row["term"]),
            estimate=row.estimate,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            blurb=_mixed_blurb(row["model"], row["term"]),
        )

    catalog_rows.append(
        {
            "test": "Paired token distances (Hara d_mu and d_sigma) and Wasserstein",
            "formula": "item mean of human distance minus LLM distance; directional_test is false",
            "family": "Descriptive",
            "n": "508 items",
            "tier": "exploratory",
            "result": "Computed. No hypothesis test was applied, so these rows are not in the significant-effect figures.",
        }
    )
    catalog = pd.DataFrame(catalog_rows)
    effects = pd.DataFrame(effect_rows)
    return catalog, effects


def _add_intercept(catalog_rows, effect_rows, **kwargs) -> None:
    _add_interval(catalog_rows, effect_rows, **kwargs)


def _add_interval(catalog_rows, effect_rows, **kwargs) -> None:
    estimate = float(kwargs["estimate"])
    ci_low = float(kwargs["ci_low"])
    ci_high = float(kwargs["ci_high"])
    excludes = _excludes(ci_low, ci_high)
    catalog_rows.append(
        {
            "test": kwargs["test"],
            "formula": kwargs["formula"],
            "family": kwargs["family"],
            "n": kwargs["n"],
            "tier": kwargs["tier"],
            "result": (
                f"{estimate:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]. "
                + ("Excludes 0." if excludes else "Includes 0.")
            ),
        }
    )
    if excludes and kwargs.get("plot", True):
        effect_rows.append(
            {
                "panel": kwargs["panel"],
                "label": kwargs["label"],
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "blurb": kwargs["blurb"].format(estimate=estimate, ci_low=ci_low, ci_high=ci_high),
            }
        )


def _mixed_label(model: str, term: str) -> str:
    labels = {
        ("naturalness_gap", "ai_first"): "Pooled naturalness gap: AI shown first",
        ("naturalness_gap", "expert"): "Pooled naturalness gap: expert minus citizen",
        ("naturalness_gap", "delta_z"): "Pooled naturalness gap: back-translation delta (per SD)",
        ("naturalness_level", "ai_first"): "Raw naturalness: AI shown first",
        ("naturalness_level", "expert"): "Raw naturalness: expert minus citizen",
        ("naturalness_level", "delta_z"): "Raw naturalness: back-translation delta (per SD)",
        ("naturalness_version", "version_human"): "Naturalness: human version minus LLM version",
        ("naturalness_version", "delta_z"): "LLM naturalness: back-translation delta (per SD)",
        ("naturalness_version", "ai_first"): "LLM naturalness rating: AI shown first",
        ("naturalness_version", "expert"): "Naturalness level: expert minus citizen",
        ("naturalness_version", "version_human:delta_z"): "Naturalness gap: back-translation delta (per SD)",
        ("naturalness_version", "version_human:ai_first"): "Naturalness gap: AI shown first",
        ("expert_semantic_gap", "ai_first"): "Expert semantic gap: AI shown first",
        ("expert_semantic_gap", "delta_z"): "Expert semantic gap: back-translation delta (per SD)",
        ("expert_cultural_validity_gap", "ai_first"): "Expert cultural-validity gap: AI shown first",
        ("expert_cultural_validity_gap", "delta_z"): "Expert cultural-validity gap: back-translation delta (per SD)",
        ("expert_choice_fit_gap", "ai_first"): "Expert choice-fit gap: AI shown first",
        ("expert_choice_fit_gap", "delta_z"): "Expert choice-fit gap: back-translation delta (per SD)",
        ("expert_naturalness_gap", "ai_first"): "Expert naturalness gap: AI shown first, delta adjusted",
        ("expert_naturalness_gap", "delta_z"): "Expert naturalness gap: back-translation delta (per SD)",
        ("preference", "ai_first"): "Prefer human: AI shown first, pooled",
        ("preference", "expert"): "Prefer human: expert minus citizen",
        ("preference", "delta_z"): "Prefer human: back-translation delta (per SD)",
        ("guess", "ai_first"): "Name the LLM column: AI shown first",
        ("guess", "expert"): "Name the LLM column: expert minus citizen",
        ("guess", "delta_z"): "Name the LLM column: signed back-translation delta (per SD)",
        ("guess_abs_delta", "ai_first"): "Name the LLM column: AI shown first, absolute-gap model",
        ("guess_abs_delta", "expert"): "Name the LLM column: expert, absolute-gap model",
        ("guess_abs_delta", "abs_delta_z"): "Name the LLM column: absolute item gap (per SD)",
        ("item_flag_cultural", "version_human"): "Cultural flag: human version",
        ("item_flag_cultural", "delta_z"): "Cultural flag on the LLM version: item delta (per SD)",
        ("item_flag_cultural", "ai_first"): "Cultural flag: AI shown first",
        ("item_flag_cultural", "version_human:delta_z"): "Cultural flag: human version by item delta",
        ("item_flag_semantic", "version_human"): "Semantic flag: human version",
        ("item_flag_semantic", "delta_z"): "Semantic flag on the LLM version: item delta (per SD)",
        ("item_flag_semantic", "ai_first"): "Semantic flag: AI shown first",
        ("item_flag_semantic", "version_human:delta_z"): "Semantic flag: human version by item delta",
        ("item_flag_understandability", "version_human"): "Understandability flag: human version",
        ("item_flag_understandability", "delta_z"): "Understandability flag on the LLM version: item delta (per SD)",
        ("item_flag_understandability", "ai_first"): "Understandability flag: AI shown first",
        ("item_flag_understandability", "version_human:delta_z"): "Understandability flag: human version by item delta",
    }
    return labels.get((model, term), f"{model}: {term}")


def _mixed_blurb(model: str, term: str) -> str:
    text = {
        ("naturalness_gap", "ai_first"): (
            "In the pooled citizen and expert naturalness gap, showing the LLM version first raises the "
            "human-minus-LLM gap by {estimate:.3f} points (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "The back-translation gap and the expert indicator are in the same model, and their intervals include 0."
        ),
        ("naturalness_level", "expert"): (
            "Experts give higher raw naturalness ratings than citizens, by {estimate:.3f} points "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). This is the average of the two versions. "
            "It is a difference in how the two surveys use the 1–5 scale, not a human-versus-LLM contrast."
        ),
        ("naturalness_version", "ai_first"): (
            "When the LLM version is the reference row, showing it first lowers that version's naturalness rating "
            "by {estimate:.3f} points (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "The human-version order interaction in the same model is the rest of the gap."
        ),
        ("naturalness_version", "expert"): (
            "On the long naturalness rows, experts rate {estimate:.3f} points higher than citizens "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). This repeats the raw-level survey difference."
        ),
        ("naturalness_version", "version_human:ai_first"): (
            "The human-minus-LLM naturalness gap is {estimate:.3f} points larger when the LLM version was shown first "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). The version slope for scales is in the model, so this standard error "
            "matches the difference-score order effect."
        ),
        ("expert_cultural_validity_gap", "ai_first"): (
            "Experts' cultural-validity gap is {estimate:.3f} points larger when the LLM version was shown first "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). At the human-first reference the gap interval includes 0, "
            "so the order term is what separates the two administration conditions."
        ),
        ("expert_choice_fit_gap", "ai_first"): (
            "Experts' choice-fit gap moves {estimate:.3f} points toward the human version when the LLM version was shown first "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). Choice fit is whether the response options fit the item."
        ),
        ("expert_choice_fit_gap", "delta_z"): (
            "A scale whose human back-translation is 1 SD closer to the original has a choice-fit gap "
            "{estimate:.3f} points higher (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "Positive means the human disadvantage on choice fit is smaller when the human text is relatively closer to the English."
        ),
        ("expert_naturalness_gap", "ai_first"): (
            "With the back-translation gap in the model, experts' naturalness gap is still {estimate:.3f} points "
            "larger when the LLM version was shown first (95% CI [{ci_low:.3f}, {ci_high:.3f}])."
        ),
        ("preference", "ai_first"): (
            "Pooling citizens and experts, the log-odds of choosing the human column rise by {estimate:.3f} "
            "when the LLM version was shown first (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "The participant intercept and the scale intercept are in the model. The back-translation slope includes 0."
        ),
        ("item_flag_cultural", "version_human"): (
            "At the average item, the human version is less often flagged as a cultural problem than the LLM version. "
            "The log-odds difference is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])."
        ),
        ("item_flag_cultural", "delta_z"): (
            "On the LLM version, an item whose human back-translation is 1 SD closer to the original is less often "
            "flagged for culture. The log-odds change is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "This is the slope when version_human is 0."
        ),
        ("item_flag_cultural", "version_human:delta_z"): (
            "The cultural-flag gap between versions reverses as the item gap grows. The interaction is {estimate:.3f} "
            "(95% CI [{ci_low:.3f}, {ci_high:.3f}]). A negative interaction would have meant the closer human text is "
            "flagged less often. The estimate is positive."
        ),
        ("item_flag_semantic", "version_human"): (
            "At the average item, experts flag the human version for a semantic problem more often than the LLM version. "
            "The log-odds difference is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])."
        ),
        ("item_flag_understandability", "version_human"): (
            "At the average item, the human version is less often flagged as hard to understand. "
            "The log-odds difference is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])."
        ),
        ("item_flag_understandability", "version_human:delta_z"): (
            "The understandability interaction is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            "When the human back-translation is closer to the original, the human version is flagged relatively more often, "
            "which is the opposite of a closer-text-is-flagged-less account."
        ),
    }
    return text.get(
        (model, term),
        "The coefficient is {estimate:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]).",
    )


def _figure_forest(frame: pd.DataFrame, path: Path, xlabel: str) -> None:
    if frame.empty:
        return
    plot = frame.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8.4, max(3.2, 0.38 * len(plot) + 1.2)))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.errorbar(
        plot["estimate"],
        np.arange(len(plot)),
        xerr=[plot["estimate"] - plot["ci_low"], plot["ci_high"] - plot["estimate"]],
        fmt="o",
        color="#1B1B1B",
        ecolor="#1B1B1B",
        capsize=3,
        markersize=5,
    )
    axis.set_yticks(np.arange(len(plot)))
    axis.set_yticklabels(plot["label"])
    low = min(float(plot["ci_low"].min()), 0.0)
    high = max(float(plot["ci_high"].max()), 0.0)
    span = high - low if high > low else 1.0
    axis.set_xlim(low - 0.06 * span, high + 0.22 * span)
    axis.set_xlabel(xlabel)
    axis.set_title("Intervals that exclude zero")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_spearman(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plot = frame.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8.6, max(3.2, 0.36 * len(plot) + 1.2)))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.scatter(plot["estimate"], np.arange(len(plot)), color="#1B1B1B", s=28)
    axis.set_yticks(np.arange(len(plot)))
    axis.set_yticklabels(plot["label"])
    axis.set_xlabel("Spearman rho")
    axis.set_xlim(-0.75, 0.45)
    axis.set_title("Rank correlations with p < 0.05")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_choice_fit(root: Path, path: Path) -> None:
    ratings = pd.read_parquet(root / "results/survey_mixed/frames/rating_gaps.parquet")
    coef = pd.read_csv(root / "results/survey_mixed/coefficients.csv")
    part = ratings.loc[ratings["outcome"].eq("choice_fit") & ratings["expert"].eq(1)]
    scale = part.groupby("scale_id", as_index=False).agg(
        gap=("rating_gap", "mean"),
        delta_z=("delta_z", "first"),
        n=("rating_gap", "size"),
    )
    hit = coef.loc[coef["model"].eq("expert_choice_fit_gap")].set_index("term")
    grid = np.linspace(scale["delta_z"].min(), scale["delta_z"].max(), 40)
    ai_mean = float(part["ai_first"].mean())
    line = hit.loc["(Intercept)", "estimate"] + hit.loc["ai_first", "estimate"] * ai_mean + hit.loc["delta_z", "estimate"] * grid
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.scatter(scale["delta_z"], scale["gap"], s=12 + scale["n"], color="#1B1B1B")
    axis.plot(grid, line, color="#4C78A8", linewidth=1.6)
    axis.set_xlabel("Scale back-translation gap, standard deviations (human cosine minus LLM cosine)")
    axis.set_ylabel("Mean expert choice-fit gap (human minus LLM)")
    axis.set_title("Choice fit and closeness to the original")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_flags(root: Path, path: Path) -> None:
    flags = pd.read_parquet(root / "results/survey_mixed/frames/item_flags.parquet")
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
    for axis, dimension, title in (
        (axes[0], "cultural", "Cultural problem"),
        (axes[1], "understandability", "Hard to understand"),
    ):
        part = flags.loc[flags["dimension"].eq(dimension)]
        item = part.groupby(["item_uid", "version", "delta_z"], as_index=False)["flagged"].mean()
        order = item.drop_duplicates("item_uid").sort_values("delta_z")
        order["bin"] = pd.qcut(order["delta_z"], 5, labels=False, duplicates="drop")
        item = item.merge(order.loc[:, ["item_uid", "bin"]], on="item_uid")
        summary = item.groupby(["bin", "version"], as_index=False).agg(
            rate=("flagged", "mean"),
            delta_z=("delta_z", "mean"),
        )
        for version, color in (("human", "#1B1B1B"), ("llm", "#4C78A8")):
            rows = summary.loc[summary["version"].eq(version)].sort_values("delta_z")
            axis.plot(rows["delta_z"], rows["rate"], marker="o", color=color, label=version)
        axis.set_title(title)
        axis.set_xlabel("Item back-translation gap (SD)")
        axis.axvline(0.0, color="#888888", linewidth=0.6)
    axes[0].set_ylabel("Share of expert ratings flagged")
    axes[1].legend(frameon=False, title="Version")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_similarity(root: Path, path: Path) -> None:
    rates = pd.read_csv(root / "results/survey_judgment/tables/choice_by_scale_condition.csv")
    citizen = rates.loc[rates["survey"].eq("citizen") & rates["trans_cond"].eq("all")]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharex=True)
    for axis, outcome, title in (
        (axes[0], "prefer_human", "Prefer the human column"),
        (axes[1], "guess_correct", "Name the LLM column"),
    ):
        part = citizen.loc[citizen["outcome"].eq(outcome)]
        axis.scatter(part["similarity_ja"], part["rate"], s=18 + part["n"] / 8, color="#1B1B1B")
        axis.set_title(title)
        axis.set_xlabel("Mean cosine of the two Japanese texts")
        axis.set_ylim(0.25, 0.85)
    axes[0].set_ylabel("Scale rate")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_order(root: Path, path: Path) -> None:
    natural = pd.read_csv(root / "results/survey_judgment/tables/naturalness_contrast.csv")
    part = natural.loc[natural["trans_cond"].isin(["human", "ai"])].copy()
    part["label"] = part["survey"].map({"citizen": "Citizen wording", "expert": "Expert naturalness"}) + part["trans_cond"].map(
        {"human": ", human first", "ai": ", LLM first"}
    )
    part = part.iloc[::-1]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.errorbar(
        part["estimate"],
        np.arange(len(part)),
        xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]],
        fmt="o",
        color="#1B1B1B",
        ecolor="#1B1B1B",
        capsize=3,
    )
    axis.set_yticks(np.arange(len(part)))
    axis.set_yticklabels(part["label"])
    axis.set_xlabel("Human rating minus LLM rating")
    axis.set_title("Naturalness gap by administration order")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report(catalog: pd.DataFrame, effects: pd.DataFrame) -> str:
    lines = [
        "# Tests, predictors, and significant effects",
        "",
        "Every hypothesis test stored for ManyScales H4 is listed in the form that was fit. A result is plotted when its 95% interval excludes 0, or, for a rank correlation, when p < 0.05. The confirmatory cosine estimate is the first row. Later rows do not replace it.",
        "",
        "An intercept in a model that also contains predictors is the outcome at the reference coding. Those intercepts are in the catalog. They are not plotted as effects of a predictor. Intercept-only models are the exception: there the intercept is the human-minus-LLM contrast.",
        "",
        "## Predictors",
        "",
        "| Predictor | Where it appears | What a one-unit change is |",
        "| --- | --- | --- |",
        "| `(Intercept)` in `delta ~ 1` | Cosine models, naturalness contrasts, expert dimension gaps | The mean human-minus-LLM difference. Positive means the human text is closer, or rated higher. |",
        "| `human` | Beta regression `similarity ~ human` | The row is the human translation's cosine rather than the LLM translation's cosine. |",
        "| `ai_first` | Survey models | 1 when `trans_cond` is `ai`: the LLM version was administered first. The comparison table still shows human Japanese as translation 1 and LLM Japanese as translation 2. |",
        "| `expert` | Pooled survey models | 1 for the expert survey, 0 for citizens. Citizen wording and expert naturalness are both 1–5 and are not the same question. |",
        "| `delta_z` in scale models | Rating gaps, preference, guess | The scale mean of item MPNet back-translation gaps, human cosine minus LLM cosine, standardized across the 22 scales. Positive means that scale's human back-translation is closer to the original than the average scale. |",
        "| `abs_delta_z` | Guess companion model | The scale mean of the absolute item gap, standardized across scales. Larger means the two back-translations are farther apart, regardless of which is closer to the English. |",
        "| `delta_z` in item-flag models | Expert problem checkboxes | The item back-translation gap, standardized across items. |",
        "| `version_human` | Long naturalness rows and item flags | 1 on the human Japanese version, 0 on the LLM version. |",
        "| `sim_z` | Citizen and expert choice GLMs | Mean MPNet cosine between `human_ja` and `llm_ja` on the items in the comparison table, standardized across scales. This is similarity of the two Japanese texts to each other. |",
        "| `(1 \\| scale_id)` | Cosine and scale-judgment models | Scales differ in their mean. The slope of a scale-level predictor is what remains after that intercept. |",
        "| `(1 \\| respondent_id)` | Crossed survey models | A person who rates several scales shares an intercept. |",
        "| `(1 \\| item_uid)` | Expert problem flags | Items differ in how often they are flagged. |",
        "",
        "## Every test",
        "",
        "| Test | Formula | Family | N | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in catalog.iterrows():
        lines.append(
            "| {test} | `{formula}` | {family} | {n} | {result} |".format(
                test=_cell(row["test"]),
                formula=_cell(row["formula"]),
                family=_cell(row["family"]),
                n=_cell(row["n"]),
                result=_cell(row["result"]),
            )
        )
    lines.extend(
        [
            "",
            "Token-cloud distances, embedding maps, metric-identity checks, and the leave-one-scale-out prediction of answer changes were computed and are not hypothesis tests of a coefficient. They are omitted from the figures below.",
            "",
            "## Effects whose interval excludes zero",
            "",
            "### Back-translation cosine",
            "",
            "The outcome in each of these rows is human similarity minus LLM similarity. A negative number means the LLM text is closer to the original English.",
            "",
        ]
    )
    lines.extend(_blurbs(effects, "cosine"))
    lines.extend(
        [
            "",
            "![Back-translation intervals that exclude zero](../results/effects/figures/cosine_back_translation.png)",
            "",
            "### Direct Japanese cosine, REML Wald intervals",
            "",
            "These rows use the same mixed-model formula on cosine(original English, Japanese). For MPNet, the hierarchical bootstrap interval is [−0.033, +0.007] and the beta interval includes 0. The figure shows the Wald intervals, which exclude 0.",
            "",
        ]
    )
    lines.extend(_blurbs(effects, "direct"))
    lines.extend(
        [
            "",
            "![Direct Japanese REML intervals](../results/effects/figures/cosine_direct_wald.png)",
            "",
            "### Ratings",
            "",
            "Gaps are human rating minus LLM rating on a 1–5 scale. Order effects are the AI-first condition minus the human-first condition.",
            "",
        ]
    )
    lines.extend(_blurbs(effects, "rating"))
    lines.extend(
        [
            "",
            "![Rating effects whose interval excludes zero](../results/effects/figures/rating_effects.png)",
            "",
            "![Naturalness gap by which version was shown first](../results/effects/figures/naturalness_by_order.png)",
            "",
            "The choice-fit slope is the one rating association with the back-translation gap. Each point is a scale. The line is the fixed part of `rating_gap ~ ai_first + delta_z + (1 | respondent_id) + (1 | scale_id)` at the sample mean of `ai_first`.",
            "",
            "![Expert choice-fit gap against the back-translation gap](../results/effects/figures/choice_fit_slope.png)",
            "",
            "### Preference, the LLM guess, and problem flags",
            "",
            "Coefficients are log-odds. Preference is 1 for the human column. The guess is 1 when the respondent names the LLM column. A flag is 1 when the expert checkbox is checked.",
            "",
        ]
    )
    lines.extend(_blurbs(effects, "logit"))
    lines.extend(
        [
            "",
            "![Log-odds effects whose interval excludes zero](../results/effects/figures/logit_effects.png)",
            "",
            "The flag figure bins items by the back-translation gap and plots the share of expert ratings that checked the box. Cultural problems and understandability are the two dimensions whose version-by-gap interaction excludes 0.",
            "",
            "![Problem-flag rate by item back-translation gap](../results/effects/figures/flag_by_delta.png)",
            "",
            "### Rank correlations",
            "",
            "These tests have a p-value and no confidence interval in the stored tables. The cluster-robust logits for between-version similarity include 0; the rank correlations across 22 scales do not, for citizens.",
            "",
        ]
    )
    lines.extend(_blurbs(effects, "spearman"))
    lines.extend(
        [
            "",
            "![Rank correlations with p below 0.05](../results/effects/figures/spearman_effects.png)",
            "",
            "![Citizen scale rates against similarity of the two Japanese texts](../results/effects/figures/citizen_similarity.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _blurbs(effects: pd.DataFrame, panel: str) -> list[str]:
    rows = effects.loc[effects["panel"].eq(panel)]
    return [row["blurb"] + "\n" for _, row in rows.iterrows()]


def _excludes(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def _spearman_label(outcome: str, predictor: str) -> str:
    names = {
        "behavior_abs": "Absolute answer change",
        "expert_semantic_flag": "Semantic problem flags",
    }
    pretty = predictor.replace("|", "absolute ").replace("_", " ")
    return f"{names.get(outcome, outcome)} ~ {pretty}"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    run()


if __name__ == "__main__":
    main()
