"""Paired human-minus-LLM contrasts.

The confirmatory estimand is the hierarchical intercept of these item
deltas, not a paired t-test that treats items as independent.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from h4.errors import SchemaError

PAIR_KEYS = (
    "scale_id",
    "item_id",
    "embedding_model",
    "short_name",
    "metric",
    "comparison",
    "analysis_tier",
    "independent_geometry",
)


def wide_delta(long_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Pivot the tidy similarity table to one row per item specification.

    ``delta`` is oriented similarity for the human text minus oriented
    similarity for the LLM text. Incomplete pairs are counted and omitted.
    """

    frame = long_df.copy()
    for column in ("scale_id", "item_id"):
        frame[column] = frame[column].astype(str)
    if frame.duplicated(list(PAIR_KEYS) + ["translation"]).any():
        raise SchemaError("The similarity table has duplicate item-condition rows.")
    unknown = set(frame["translation"].unique()) - {"human", "llm"}
    if unknown:
        raise SchemaError(f"Unexpected translation labels: {sorted(unknown)}")

    human = frame.loc[frame["translation"] == "human"]
    llm = frame.loc[frame["translation"] == "llm"]
    merged = human.merge(llm, on=list(PAIR_KEYS), how="inner", suffixes=("_human", "_llm"))
    dropped = int(len(human) - len(merged))
    merged["sim_human"] = merged["similarity_human"]
    merged["sim_llm"] = merged["similarity_llm"]
    merged["raw_human"] = merged["raw_value_human"]
    merged["raw_llm"] = merged["raw_value_llm"]
    merged["delta"] = merged["sim_human"] - merged["sim_llm"]
    keep = list(PAIR_KEYS) + ["sim_human", "sim_llm", "raw_human", "raw_llm", "delta"]
    return merged.loc[:, keep].reset_index(drop=True), dropped


def descriptive_paired_ttest(wide: pd.DataFrame) -> dict[str, float | str]:
    """Item-level paired t-test. This ignores scale nesting."""

    result = stats.ttest_rel(wide["sim_human"], wide["sim_llm"])
    return {
        "role": "descriptive_ignores_scale_nesting",
        "estimate": float(wide["delta"].mean()),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_items": int(len(wide)),
    }
