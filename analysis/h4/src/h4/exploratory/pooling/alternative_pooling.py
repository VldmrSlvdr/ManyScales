"""Aggregation operators used only in the exploratory tier.

Mean and CLS are the configured pair. Max pooling is implemented so the
operator exists, and the default configuration leaves it off.
"""

from __future__ import annotations

import numpy as np

from h4.exploratory.pooling.collapse_metric import cls_vector, mean_pool


def pool(tokens: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return mean_pool(tokens)
    if method == "cls":
        return cls_vector(tokens)
    if method == "max":
        return np.max(np.asarray(tokens, dtype=np.float64), axis=0)
    raise ValueError(f"Unknown pooling method: {method}")
