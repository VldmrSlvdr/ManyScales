"""Euclidean distance after L2 normalization.

For unit vectors, ``||x - y||^2 = 2 (1 - cos(x, y))``. The value is a
distance: smaller means closer.
"""

from __future__ import annotations

import numpy as np

from h4.similarity.cosine import _l2_rows, _rows


def euclidean_normalized(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = _l2_rows(_rows(left))
    b = _l2_rows(_rows(right))
    if a.shape != b.shape:
        raise ValueError("Euclidean distance expects two matrices with the same shape.")
    return np.linalg.norm(a - b, axis=1)
