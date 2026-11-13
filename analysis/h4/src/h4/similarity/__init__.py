"""Dispatch from a metric name to a raw value and a higher-is-closer score."""

from __future__ import annotations

import numpy as np

from h4.similarity.angular import angular_distance
from h4.similarity.cosine import cosine_similarity
from h4.similarity.distance import euclidean_normalized


def score_vectors(metric: str, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(similarity, raw_value)``.

    ``similarity`` is oriented so that a larger number means the two vectors
    are closer. Distances are negated for that column; ``raw_value`` keeps
    the metric as defined.
    """

    if metric == "cosine":
        raw = cosine_similarity(left, right)
        return raw, raw
    if metric == "angular_distance":
        raw = angular_distance(left, right)
        return 1.0 - raw, raw
    if metric == "euclidean_normalized":
        raw = euclidean_normalized(left, right)
        return -raw, raw
    raise ValueError(f"Unknown similarity metric: {metric}")
