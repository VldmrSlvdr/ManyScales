"""Second-order collapse by mean pooling (SOCM).

Implements Hara, Kurita, Imaizumi, Inui, and Yokoi (2026), ACL long paper
2183 / arXiv:2604.27398.

Token lists are rescaled so the mean has unit norm. Then

    d_mu = ||mu_1 - mu_2||^2 / 4
    d_sigma = BuresWasserstein(Sigma_1, Sigma_2)^2 / 4
    SOCM = (1 - d_mu) * d_sigma

Covariance uses the 1/n divisor from the paper. The Bures term is computed
in the joint column space of the two centered clouds, which is exact for
these low-rank covariances. ``trace_bound_ok`` records whether both traces
are at most 2, the paper's condition for keeping ``d_sigma`` inside [0, 1].
Values are still returned when the bound fails.

This quantity is exploratory. It is not an input to the confirmatory H4 test.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def socm(tokens_a: np.ndarray, tokens_b: np.ndarray) -> dict[str, Any]:
    left = _as_tokens(tokens_a)
    right = _as_tokens(tokens_b)
    left_n = _normalize_by_mean(left)
    right_n = _normalize_by_mean(right)
    mu_left = left_n.mean(axis=0)
    mu_right = right_n.mean(axis=0)
    d_mu = float(np.sum((mu_left - mu_right) ** 2) / 4.0)
    d_mu = _snap_unit_interval(d_mu)
    cov_left, cov_right, trace_left, trace_right = _joint_covariances(left_n, right_n)
    bures = _bures_squared(cov_left, cov_right)
    d_sigma = float(bures / 4.0)
    if d_sigma < 0 and d_sigma > -1e-8:
        d_sigma = 0.0
    cosine = float(np.dot(mu_left, mu_right))
    return {
        "cosine": cosine,
        "d_mu": d_mu,
        "d_sigma": d_sigma,
        "socm": float((1.0 - d_mu) * d_sigma),
        "trace_reference": trace_left,
        "trace_comparison": trace_right,
        "trace_bound_ok": bool(trace_left <= 2.0 + 1e-8 and trace_right <= 2.0 + 1e-8),
        "spread_reference": trace_left,
        "spread_comparison": trace_right,
        "n_tokens_reference": int(len(left_n)),
        "n_tokens_comparison": int(len(right_n)),
        "directional_test": False,
    }


def mean_pool(tokens: np.ndarray) -> np.ndarray:
    matrix = _as_tokens(tokens)
    return matrix.mean(axis=0)


def cls_vector(tokens: np.ndarray) -> np.ndarray:
    matrix = _as_tokens(tokens)
    return matrix[0]


def _as_tokens(tokens: np.ndarray) -> np.ndarray:
    array = np.asarray(tokens, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Token embeddings must have shape (n_tokens, dimension) with n_tokens > 0.")
    if not np.isfinite(array).all():
        raise ValueError("Token embeddings contain non-finite values.")
    return array


def _normalize_by_mean(tokens: np.ndarray) -> np.ndarray:
    mu = tokens.mean(axis=0)
    norm = float(np.linalg.norm(mu))
    if norm < 1e-12:
        raise ValueError(
            "The mean token vector is near zero, so SOCM's unit-mean normalization is undefined."
        )
    return tokens / norm


def _snap_unit_interval(value: float) -> float:
    if -1e-8 < value < 0:
        return 0.0
    if 1.0 < value < 1.0 + 1e-8:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"d_mu left the unit interval: {value}")
    return value


def _joint_covariances(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu_left = left.mean(axis=0)
    mu_right = right.mean(axis=0)
    centered_left = left - mu_left
    centered_right = right - mu_right
    trace_left = float(np.sum(centered_left**2) / len(left))
    trace_right = float(np.sum(centered_right**2) / len(right))
    stacked = np.vstack([centered_left, centered_right])
    if np.allclose(stacked, 0.0):
        zero = np.zeros((1, 1), dtype=np.float64)
        return zero, zero, trace_left, trace_right
    _u, singular, vt = np.linalg.svd(stacked, full_matrices=False)
    tolerance = float(singular[0]) * 1e-8 if singular.size else 0.0
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        zero = np.zeros((1, 1), dtype=np.float64)
        return zero, zero, trace_left, trace_right
    basis = vt[:rank].T
    cov_left = (centered_left @ basis).T @ (centered_left @ basis) / len(left)
    cov_right = (centered_right @ basis).T @ (centered_right @ basis) / len(right)
    return _sym(cov_left), _sym(cov_right), trace_left, trace_right


def _bures_squared(cov_left: np.ndarray, cov_right: np.ndarray) -> float:
    evals, evecs = np.linalg.eigh(_sym(cov_left))
    evals = np.clip(evals, 0.0, None)
    sqrt_left = (evecs * np.sqrt(evals)) @ evecs.T
    middle = _sym(sqrt_left @ cov_right @ sqrt_left)
    fidelity = float(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None))))
    value = float(np.trace(cov_left) + np.trace(cov_right) - 2.0 * fidelity)
    if value < 0 and value > -1e-8:
        return 0.0
    return value


def _sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
