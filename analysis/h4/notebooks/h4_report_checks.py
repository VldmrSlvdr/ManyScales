"""Called by the Hypothesis 4 Quarto reports.

Reads the published tables, refits the confirmatory REML intercept, and prints
the same decision numbers the report quotes. It does not print a local path.
"""

import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings(
    "ignore",
    message="The MLE may be on the boundary of the parameter space.",
)

repo = Path(__file__).resolve().parents[1]
published = repo / "data" / "published"


def main() -> None:
    gaps = pd.read_csv(published / "mpnet_item_delta.csv", dtype={"scale_id": str})
    primary = gaps.loc[
        gaps["comparison"].eq("back_translation") & gaps["in_primary"].astype(bool)
    ].copy()
    primary["scale_id"] = primary["scale_id"].astype(str).str.zfill(2)
    fit = smf.mixedlm("delta ~ 1", primary, groups=primary["scale_id"]).fit(reml=True)
    estimate = float(fit.fe_params["Intercept"])
    se = float(fit.bse_fe["Intercept"])
    low, high = estimate - 1.96 * se, estimate + 1.96 * se
    print(f"items {primary.shape[0]}")
    print(f"scales {primary['scale_id'].nunique()}")
    print(f"reml {estimate:.6f} {se:.6f} {low:.6f} {high:.6f}")
    assert primary.shape[0] == 486
    assert primary["scale_id"].nunique() == 21
    assert abs(estimate - (-0.041283224127389495)) < 1e-6

    ratings = pd.read_csv(published / "raw_rating_ordinal_n1859.csv")
    cultural_gap = ratings.loc[
        ratings["outcome"].eq("cultural_validity")
        & ratings["audience"].eq("expert")
        & ratings["term"].eq("x:sim")
    ].iloc[0]
    citizen_naturalness = ratings.loc[
        ratings["outcome"].eq("naturalness")
        & ratings["audience"].eq("citizen")
        & ratings["term"].eq("x:sim")
    ].iloc[0]
    print(
        "cultural_validity_slope_difference "
        f"{cultural_gap.estimate:.6f} {cultural_gap.ci_low:.6f} "
        f"{cultural_gap.ci_high:.6f}"
    )
    assert cultural_gap.ci_low > 0
    assert citizen_naturalness.ci_low < 0 < citizen_naturalness.ci_high
    cultural_simple = ratings.loc[
        ratings["outcome"].eq("cultural_validity")
        & ratings["audience"].eq("expert")
        & ratings["term"].isin(["sim_human", "sim_llm"])
    ]
    assert len(cultural_simple) == 2
    assert (cultural_simple["ci_low"] < 0).all()
    assert (cultural_simple["ci_high"] > 0).all()

    codebook = pd.read_csv(published / "cultural_rating_codebook.csv")
    assert codebook.loc[codebook["display_position"].eq(1), "export_score"].eq(5).all()
    assert codebook.loc[codebook["display_position"].eq(5), "export_score"].eq(1).all()
    coding_audit = pd.read_csv(published / "cultural_rating_coding_audit.csv")
    assert coding_audit["export_to_h4_equal"].eq(True).all()
    assert coding_audit["additional_reversals_in_h4"].eq(0).all()
    print("cultural_rating_direction higher = better; no additional reversal")

    choice_fit = ratings.loc[
        ratings["outcome"].eq("choice_fit") & ratings["audience"].eq("expert")
    ].set_index("term")
    human_choice = choice_fit.loc["sim_human"]
    choice_difference = choice_fit.loc["x:sim"]
    assert human_choice.ci_low > 0
    assert choice_difference.ci_low < 0 < choice_difference.ci_high
    assert abs(
        human_choice.estimate
        - (choice_fit.loc["sim", "estimate"] + 0.5 * choice_difference.estimate)
    ) < 1e-12
    print(
        "choice_fit human similarity slope positive; "
        "human-minus-LLM slope difference includes zero"
    )

    flags = pd.read_csv(published / "raw_flag_logit_n1859.csv")
    flag_differences = flags.loc[flags["term"].eq("x:sim")]
    assert (flag_differences["ci_low"] < 0).all()
    assert (flag_differences["ci_high"] > 0).all()

    own = pd.read_csv(published / "own_similarity_coef.csv")
    intercepts = own.loc[own["term"].eq("(Intercept)")]
    for _, row in intercepts.iterrows():
        print(
            f"intercept {row.outcome} {row.estimate:.6f} "
            f"{row.ci_low:.6f} {row.ci_high:.6f} {row.p:.6g}"
        )
    natural = intercepts.loc[intercepts["outcome"].eq("naturalness")].iloc[0]
    assert natural.ci_low < 0 < natural.ci_high


if __name__ == "__main__":
    main()
