"""Angular distance, a monotone transform of cosine."""

from __future__ import annotations

import numpy as np

from h4.similarity.cosine import cosine_similarity


def angular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``arccos(cosine) / pi``, in ``[0, 1]`` for cosine in ``[-1, 1]``."""

    cosine = np.clip(cosine_similarity(left, right), -1.0, 1.0)
    return np.arccos(cosine) / np.pi
