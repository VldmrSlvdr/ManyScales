"""Hierarchical bootstrap of the item-mean human-minus-LLM difference.

Each draw resamples scales with replacement, then resamples items inside
each drawn scale with replacement, and averages the concatenated item
deltas. Longer scales therefore contribute more items to the mean. This
follows the protocol's resampling recipe and is reported as a robustness
interval, separate from the mixed-model intercept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    data = frame.loc[:, ["scale_id", "delta"]].copy()
    data["scale_id"] = data["scale_id"].astype(str)
    groups = {
        scale: values["delta"].to_numpy(dtype=np.float64)
        for scale, values in data.groupby("scale_id", sort=True)
    }
    scale_ids = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for draw in range(iterations):
        chosen = rng.choice(scale_ids, size=len(scale_ids), replace=True)
        parts = []
        for scale in chosen:
            values = groups[str(scale)]
            parts.append(rng.choice(values, size=len(values), replace=True))
        draws[draw] = np.concatenate(parts).mean()
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "method": "hierarchical_bootstrap",
        "bootstrap_estimand": "item_weighted_mean_under_scale_then_item_resampling",
        "observed_mean": float(data["delta"].mean()),
        "bootstrap_mean": float(draws.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence,
        "proportion_positive": float(np.mean(draws > 0)),
        "iterations": int(iterations),
        "n_items": int(len(data)),
        "n_scales": int(len(scale_ids)),
        "seed": int(seed),
        "draws": draws,
    }
