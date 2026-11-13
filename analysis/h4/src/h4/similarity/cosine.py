"""Row-wise cosine similarity of embedding matrices."""

from __future__ import annotations

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine of corresponding rows. Vectors are L2-normalized first."""

    a = _rows(left)
    b = _rows(right)
    if a.shape != b.shape:
        raise ValueError("Cosine similarity expects two matrices with the same shape.")
    return np.sum(_l2_rows(a) * _l2_rows(b), axis=1)


def _rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("Embeddings must be a vector or a matrix.")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain non-finite values.")
    return array


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding.")
    return matrix / norms
