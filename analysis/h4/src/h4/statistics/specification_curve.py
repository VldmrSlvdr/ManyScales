"""Specification curve over embedding models and comparisons.

Raw cosine scores are not averaged across models. The curve reports each
within-model paired difference. Sensitivity transforms of cosine are left
out of the independent-specification summary. No model is selected.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from h4.statistics.mixed import fit_hierarchical


def specification_curve(
    wide: pd.DataFrame,
    *,
    confidence: float,
    engine: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    grouped = wide.groupby(
        ["embedding_model", "short_name", "metric", "comparison", "analysis_tier", "independent_geometry"],
        sort=False,
        dropna=False,
    )
    for keys, group in grouped:
        model, short_name, metric, comparison, tier, independent = keys
        record = {
            "embedding_model": model,
            "short_name": short_name,
            "metric": metric,
            "comparison": comparison,
            "analysis_tier": tier,
            "independent_geometry": bool(independent),
            "n_items": int(len(group)),
            "n_scales": int(group["scale_id"].nunique()),
            "item_mean_delta": float(group["delta"].mean()),
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "sign": _sign(float(group["delta"].mean())),
            "interval_status": "unavailable",
        }
        if group["scale_id"].nunique() >= 2:
            try:
                fitted = fit_hierarchical(
                    group,
                    confidence=confidence,
                    engine=engine,
                    seed=seed,
                )
            except (RuntimeError, ValueError):
                fitted = None
            if fitted is not None:
                record.update(
                    {
                        "estimate": fitted["estimate"],
                        "ci_low": fitted["ci_low"],
                        "ci_high": fitted["ci_high"],
                        "sign": fitted["sign"],
                        "interval_status": "hierarchical",
                    }
                )
        rows.append(record)
    table = pd.DataFrame(rows)
    independent = table.loc[table["metric"] == "cosine"].copy()
    n_independent = int(len(independent))
    n_positive = int((independent["item_mean_delta"] > 0).sum()) if n_independent else 0
    summary = {
        "n_independent_specs": n_independent,
        "n_positive_item_means": n_positive,
        "sign_consistency": (n_positive / n_independent) if n_independent else float("nan"),
        "min_item_mean_delta": float(independent["item_mean_delta"].min()) if n_independent else float("nan"),
        "max_item_mean_delta": float(independent["item_mean_delta"].max()) if n_independent else float("nan"),
        "selected_model": None,
        "cross_model_mean_of_raw_similarities": None,
        "reason": "Raw similarities are not averaged across embedding models, and no specification is promoted to the confirmatory test.",
    }
    return table, summary


def _sign(estimate: float) -> str:
    if estimate > 0:
        return "positive"
    if estimate < 0:
        return "negative"
    return "zero"
