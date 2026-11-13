"""Distances between empirical token-embedding distributions.

These distances are exploratory. Unequal tokenization changes the uniform
empirical measure, so a larger Wasserstein value is not by itself evidence
about translation fidelity, and it is not entered into the H4 model.

The exact solver is a transportation linear program. When the cost matrix
exceeds 4096 entries, entropic Sinkhorn is used and the solver name is
recorded on the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog


def pairwise_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * a @ b.T, 0.0))


def pairwise_cosine_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = _unit_rows(np.asarray(left, dtype=np.float64))
    b = _unit_rows(np.asarray(right, dtype=np.float64))
    return 1.0 - a @ b.T


def wasserstein(cost: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("The cost matrix must be two-dimensional and non-empty.")
    if matrix.shape[0] * matrix.shape[1] > 4096:
        value = sinkhorn(matrix)
        return {"distance": value, "solver": "sinkhorn_fallback"}
    value = _emd(matrix)
    return {"distance": value, "solver": "emd"}


def sinkhorn(cost: np.ndarray, *, epsilon: float = 0.05, n_iter: int = 200) -> float:
    matrix = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = matrix.shape
    kernel = np.exp(-matrix / epsilon)
    kernel = np.maximum(kernel, 1e-12)
    scale_row = np.ones(n_rows)
    scale_col = np.ones(n_cols)
    target_row = np.full(n_rows, 1.0 / n_rows)
    target_col = np.full(n_cols, 1.0 / n_cols)
    for _ in range(n_iter):
        scale_row = target_row / (kernel @ scale_col)
        scale_col = target_col / (kernel.T @ scale_row)
    plan = scale_row[:, None] * kernel * scale_col[None, :]
    return float(np.sum(plan * matrix))


def _emd(cost: np.ndarray) -> float:
    n_rows, n_cols = cost.shape
    objective = cost.reshape(-1)
    n_constraints = n_rows + n_cols - 1
    constraints = np.zeros((n_constraints, n_rows * n_cols), dtype=np.float64)
    target = np.concatenate(
        [np.full(n_rows, 1.0 / n_rows), np.full(n_cols - 1, 1.0 / n_cols)]
    )
    for row in range(n_rows):
        constraints[row, row * n_cols : (row + 1) * n_cols] = 1.0
    for col in range(n_cols - 1):
        constraints[n_rows + col, col::n_cols] = 1.0
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=target,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Earth-mover solver failed: {result.message}")
    return float(result.fun)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)
