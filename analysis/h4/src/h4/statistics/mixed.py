"""Random-intercept model for the item-level human-minus-LLM difference.

The fitted equation is ``delta ~ 1`` with a random intercept for
``scale_id``. Provenance columns are not entered.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from h4.errors import SchemaError


def fit_hierarchical(
    frame: pd.DataFrame,
    *,
    confidence: float = 0.95,
    engine: str = "statsmodels_mixedlm",
    seed: int = 0,
) -> dict[str, Any]:
    data = _model_frame(frame)
    if engine == "bambi":
        return _fit_bambi(data, confidence=confidence, seed=seed)
    if engine != "statsmodels_mixedlm":
        raise SchemaError(f"Unknown hierarchical engine: {engine}")
    return _fit_mixedlm(data, confidence=confidence)


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "delta" not in frame.columns or "scale_id" not in frame.columns:
        raise SchemaError("The hierarchical model needs delta and scale_id.")
    data = frame.loc[:, ["delta", "scale_id"]].copy()
    data["scale_id"] = data["scale_id"].astype(str)
    data["delta"] = pd.to_numeric(data["delta"], errors="raise")
    data = data.reset_index(drop=True)
    n_scales = data["scale_id"].nunique()
    if n_scales < 2:
        raise SchemaError("The hierarchical model needs at least two scales.")
    if data["delta"].isna().any():
        raise SchemaError("delta contains missing values.")
    return data


def _fit_mixedlm(data: pd.DataFrame, *, confidence: float) -> dict[str, Any]:
    model = smf.mixedlm("delta ~ 1", data, groups=data["scale_id"])
    if model.exog.shape[1] != 1:
        raise SchemaError("The hierarchical design matrix must contain only an intercept.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    estimate = float(np.asarray(fit.fe_params).reshape(-1)[0])
    se = float(np.asarray(fit.bse_fe).reshape(-1)[0])
    if not np.isfinite(estimate) or not np.isfinite(se) or se < 0:
        raise RuntimeError("The hierarchical model did not return a finite intercept and standard error.")
    alpha = 1.0 - confidence
    ci = fit.conf_int(alpha=alpha)
    if "Intercept" in ci.index:
        ci_low, ci_high = (float(value) for value in ci.loc["Intercept"].tolist())
    else:
        ci_low, ci_high = (float(value) for value in ci.iloc[0].tolist())
    scale_variance = float(np.asarray(fit.cov_re, dtype=float).reshape(-1)[0])
    residual_variance = float(fit.scale)
    denom = scale_variance + residual_variance
    icc = float(scale_variance / denom) if denom > 0 else float("nan")
    scale_table = _scale_table(data, fit, estimate)
    warning_text = "; ".join(str(item.message) for item in caught)
    return {
        "method": "hierarchical",
        "engine": "statsmodels_mixedlm",
        "formula": "delta ~ 1 + (1|scale_id)",
        "estimate": estimate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence,
        "prob_positive": float(stats.norm.cdf(estimate / se)),
        "prob_positive_meaning": "normal_approximation_to_the_sampling_distribution",
        "n_items": int(len(data)),
        "n_scales": int(data["scale_id"].nunique()),
        "scale_variance": scale_variance,
        "residual_variance": residual_variance,
        "icc": icc,
        "item_mean": float(data["delta"].mean()),
        "unweighted_scale_mean": float(data.groupby("scale_id")["delta"].mean().mean()),
        "ci_above_zero": bool(ci_low > 0),
        "sign": _sign(estimate),
        "convergence_warning": warning_text,
        "scale_table": scale_table,
        "design_matrix_columns": 1,
    }


def _scale_table(data: pd.DataFrame, fit, intercept: float) -> pd.DataFrame:
    effects = getattr(fit, "random_effects", {})
    rows = []
    for scale_id, group in data.groupby("scale_id", sort=True):
        effect = effects.get(scale_id, effects.get(str(scale_id), 0.0))
        if hasattr(effect, "iloc"):
            shift = float(np.asarray(effect).reshape(-1)[0])
        else:
            array = np.asarray(effect, dtype=float).reshape(-1)
            shift = float(array[0]) if array.size else 0.0
        rows.append(
            {
                "scale_id": str(scale_id),
                "n_items": int(len(group)),
                "mean_delta": float(group["delta"].mean()),
                "blup": float(intercept + shift),
            }
        )
    return pd.DataFrame(rows)


def _fit_bambi(data: pd.DataFrame, *, confidence: float, seed: int) -> dict[str, Any]:
    try:
        import bambi as bmb
    except ImportError as exc:
        raise RuntimeError(
            "inference.engine is bambi, and bambi is not installed. "
            "Install the bayesian extra, or set engine to statsmodels_mixedlm."
        ) from exc
    model = bmb.Model("delta ~ 1 + (1|scale_id)", data=data)
    idata = model.fit(
        draws=1000,
        tune=1000,
        chains=2,
        random_seed=seed,
        progressbar=False,
    )
    posterior = idata.posterior
    name = "Intercept" if "Intercept" in posterior else "delta_Intercept"
    samples = np.asarray(posterior[name]).reshape(-1)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(samples, [alpha, 1.0 - alpha])
    estimate = float(samples.mean())
    return {
        "method": "hierarchical",
        "engine": "bambi",
        "formula": "delta ~ 1 + (1|scale_id)",
        "estimate": estimate,
        "se": float(samples.std(ddof=1)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence,
        "prob_positive": float(np.mean(samples > 0)),
        "prob_positive_meaning": "posterior_probability",
        "n_items": int(len(data)),
        "n_scales": int(data["scale_id"].nunique()),
        "scale_variance": float("nan"),
        "residual_variance": float("nan"),
        "icc": float("nan"),
        "item_mean": float(data["delta"].mean()),
        "unweighted_scale_mean": float(data.groupby("scale_id")["delta"].mean().mean()),
        "ci_above_zero": bool(ci_low > 0),
        "sign": _sign(estimate),
        "convergence_warning": "",
        "scale_table": data.groupby("scale_id")["delta"].agg(n_items="size", mean_delta="mean").reset_index(),
        "design_matrix_columns": 1,
    }


def _sign(estimate: float) -> str:
    if estimate > 0:
        return "positive"
    if estimate < 0:
        return "negative"
    return "zero"
