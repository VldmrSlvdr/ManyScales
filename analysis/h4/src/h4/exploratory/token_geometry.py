"""Item-level geometry summaries and cross-model disagreement.

``V_i`` is the variance, across embedding models, of the item-level
human-minus-LLM cosine difference. Its association with SOCM is reported
as an exploratory Spearman correlation. A missing similarity table leaves
the association unestimated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def model_disagreement(wide_cosine: pd.DataFrame) -> pd.DataFrame:
    """Variance of confirmatory-scale deltas across models, per item.

    ``wide_cosine`` contains one row per item and embedding model for a
    single comparison and the cosine metric.
    """

    frame = wide_cosine.copy()
    frame["scale_id"] = frame["scale_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    grouped = frame.groupby(["scale_id", "item_id"], sort=True)["delta"]
    rows = []
    for (scale_id, item_id), values in grouped:
        array = values.to_numpy(dtype=np.float64)
        rows.append(
            {
                "scale_id": scale_id,
                "item_id": item_id,
                "n_models": int(array.size),
                "variance_across_models": float(np.var(array, ddof=1)) if array.size >= 2 else float("nan"),
                "analysis_tier": "exploratory",
                "directional_test": False,
            }
        )
    return pd.DataFrame(rows)


def associate_disagreement(
    disagreement: pd.DataFrame,
    collapse: pd.DataFrame,
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, Any]:
    """Spearman correlation of cross-model variance with mean SOCM.

    The correlation is exploratory. It is not a test of H4.
    """

    if disagreement.empty or "variance_across_models" not in disagreement.columns:
        return _empty_association("disagreement table is empty")
    usable = disagreement.dropna(subset=["variance_across_models"]).copy()
    usable["scale_id"] = usable["scale_id"].map(_key)
    usable["item_id"] = usable["item_id"].map(_key)
    if usable["n_models"].max() < 2 if len(usable) else True:
        return _empty_association("at least two embedding models are required")
    collapsed = collapse.copy()
    collapsed["scale_id"] = collapsed["scale_id"].map(_key)
    collapsed["item_id"] = collapsed["item_id"].map(_key)
    collapsed = (
        collapsed.groupby(["scale_id", "item_id"], sort=True)["socm"].mean().rename("mean_socm").reset_index()
    )
    merged = usable.merge(collapsed, on=["scale_id", "item_id"], how="inner")
    if len(merged) < 5:
        return _empty_association("fewer than five items have both quantities")
    rho, _pvalue = stats.spearmanr(merged["variance_across_models"], merged["mean_socm"])
    lo, hi = _spearman_interval(
        merged["variance_across_models"].to_numpy(),
        merged["mean_socm"].to_numpy(),
        seed=seed,
        iterations=iterations,
    )
    return {
        "analysis_tier": "exploratory",
        "directional_test": False,
        "n_items": int(len(merged)),
        "spearman_rho": float(rho),
        "ci_low": lo,
        "ci_high": hi,
        "status": "estimated",
        "reason": "Exploratory association between cross-model delta variance and mean SOCM.",
    }


def _spearman_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    count = len(left)
    for index in range(iterations):
        take = rng.integers(0, count, size=count)
        rho, _ = stats.spearmanr(left[take], right[take])
        draws[index] = rho
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


def _empty_association(reason: str) -> dict[str, Any]:
    return {
        "analysis_tier": "exploratory",
        "directional_test": False,
        "n_items": 0,
        "spearman_rho": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "status": "not_estimated",
        "reason": reason,
    }
