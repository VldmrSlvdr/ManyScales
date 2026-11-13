"""Beta regression for paired cosine similarities.

Cosine similarities lie in a bounded interval and are often piled up near
the ceiling, so a Gaussian residual model is a poor description of the
scores themselves. This check treats each similarity as a beta outcome
on (0, 1), with a logit mean that differs between the human and LLM
texts. Items stay paired, and uncertainty comes from a hierarchical
bootstrap plus a scale-clustered Wald test.

The beta mean difference is a robustness summary. It does not replace
the preregistered mixed-model estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.othermod.betareg import BetaModel

from h4.errors import SchemaError


def fit_beta_contrast(
    frame: pd.DataFrame,
    *,
    confidence: float = 0.95,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Test whether the beta mean similarity differs between human and LLM.

    ``frame`` has one row per item, with ``sim_human``, ``sim_llm``, and
    ``scale_id``. The human coefficient is on the logit-mean scale. The
    reported mean difference is the fitted beta mean for the human text
    minus the fitted beta mean for the LLM text, on the similarity scale.
    """

    data = _pair_frame(frame)
    human = np.array([1.0, 0.0], dtype=np.float64)
    point = _fit_beta(data["similarity"], data["human"])
    point["response"] = str(data.attrs.get("response", point["response"]))
    groups = data["scale_id"].to_numpy()
    wald = _cluster_wald(data["similarity"].to_numpy(), data["human"].to_numpy(), groups)
    draws = _bootstrap_differences(
        data,
        iterations=iterations,
        seed=seed,
    )
    alpha = (1.0 - confidence) / 2.0
    usable = draws[np.isfinite(draws)]
    if len(usable) < max(20, iterations // 5):
        raise SchemaError("The beta bootstrap produced too few successful fits.")
    ci_low, ci_high = np.quantile(usable, [alpha, 1.0 - alpha])
    mean_difference = point["mean_human"] - point["mean_llm"]
    return {
        "method": "beta_regression",
        "family": "beta",
        "link": "logit",
        "formula": "similarity ~ human",
        "response": point["response"],
        "estimate_logit": point["logit_human"],
        "se_logit_cluster": wald["se"],
        "z_cluster": wald["z"],
        "p_two_sided_cluster": wald["p_two_sided"],
        "covariance": wald["covariance"],
        "mean_human": point["mean_human"],
        "mean_llm": point["mean_llm"],
        "precision": point["precision"],
        "mean_difference": mean_difference,
        "bootstrap_mean_difference": float(np.mean(usable)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence,
        "proportion_positive": float(np.mean(usable > 0)),
        "bootstrap_iterations": int(iterations),
        "bootstrap_successes": int(len(usable)),
        "n_items": int(data["item_key"].nunique()),
        "n_scales": int(data["scale_id"].nunique()),
        "n_observations": int(len(data)),
        "sign": _sign(mean_difference),
        "ci_above_zero": bool(ci_low > 0),
        "ci_below_zero": bool(ci_high < 0),
        "seed": int(seed),
    }


def _pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("scale_id", "sim_human", "sim_llm")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError("Beta contrast needs " + ", ".join(missing) + ".")
    item_id = frame["item_id"].astype(str) if "item_id" in frame.columns else pd.Series(
        np.arange(len(frame)).astype(str), index=frame.index
    )
    scale_id = frame["scale_id"].map(_scale_key)
    human = _as_beta_response(frame["sim_human"].to_numpy())
    llm = _as_beta_response(frame["sim_llm"].to_numpy())
    response = "cosine_clipped_to_open_unit_interval"
    if np.nanmin(frame[["sim_human", "sim_llm"]].to_numpy(dtype=float)) < 0:
        response = "shifted_cosine_(cosine+1)/2_clipped_to_open_unit_interval"
    rows = pd.DataFrame(
        {
            "scale_id": np.repeat(scale_id.to_numpy(), 2),
            "item_key": np.repeat((scale_id + "::" + item_id).to_numpy(), 2),
            "human": np.tile(np.array([1.0, 0.0]), len(frame)),
            "similarity": np.column_stack([human, llm]).ravel(),
        }
    )
    rows.attrs["response"] = response
    return rows


def _as_beta_response(values: np.ndarray) -> np.ndarray:
    scored = np.asarray(values, dtype=np.float64)
    if np.nanmin(scored) < 0:
        scored = (scored + 1.0) / 2.0
    return np.clip(scored, 1e-6, 1.0 - 1e-6)


def _fit_beta(similarity: np.ndarray | pd.Series, human: np.ndarray | pd.Series) -> dict[str, Any]:
    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

    y = np.asarray(similarity, dtype=np.float64)
    indicator = np.asarray(human, dtype=np.float64)
    exog = np.column_stack([np.ones(len(y)), indicator])
    model = BetaModel(y, exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        fitted = model.fit(disp=False, maxiter=300)
    converged = getattr(fitted, "mle_retvals", {}).get("converged", True)
    if not converged:
        raise RuntimeError("beta fit did not converge")
    params = np.asarray(fitted.params, dtype=np.float64)
    mean_human = float(model.predict(params, np.array([[1.0, 1.0]]))[0])
    mean_llm = float(model.predict(params, np.array([[1.0, 0.0]]))[0])
    precision = float(np.exp(params[-1]))
    response = "cosine_clipped_to_open_unit_interval"
    return {
        "logit_human": float(params[1]),
        "mean_human": mean_human,
        "mean_llm": mean_llm,
        "precision": precision,
        "response": response,
        "params": params,
    }


def _cluster_wald(similarity: np.ndarray, human: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    from scipy import stats

    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

    exog = np.column_stack([np.ones(len(similarity)), human])
    model = BetaModel(similarity, exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        try:
            fitted = model.fit(
                disp=False,
                maxiter=300,
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
            covariance = "cluster_scale_id"
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            fitted = model.fit(disp=False, maxiter=300)
            covariance = "observed_information"
    if not getattr(fitted, "mle_retvals", {}).get("converged", True):
        raise RuntimeError("beta fit did not converge")
    se = float(np.asarray(fitted.bse)[1])
    estimate = float(np.asarray(fitted.params)[1])
    z_value = estimate / se if se > 0 else float("nan")
    return {
        "se": se,
        "z": float(z_value),
        "p_two_sided": float(2 * stats.norm.sf(abs(z_value))),
        "covariance": covariance,
    }


def _bootstrap_differences(
    data: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    by_scale: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for (scale_id, _item_key), group in data.groupby(["scale_id", "item_key"], sort=False):
        by_scale.setdefault(str(scale_id), []).append(
            (group["similarity"].to_numpy(), group["human"].to_numpy())
        )
    scale_ids = np.array(list(by_scale.keys()))
    rng = np.random.default_rng(seed)
    draws = np.full(iterations, np.nan)
    for draw in range(iterations):
        chosen = rng.choice(scale_ids, size=len(scale_ids), replace=True)
        similarities = []
        indicators = []
        for scale in chosen:
            pool = by_scale[str(scale)]
            take = rng.integers(0, len(pool), size=len(pool))
            for index in take:
                similarity, indicator = pool[index]
                similarities.append(similarity)
                indicators.append(indicator)
        try:
            fitted = _fit_beta(np.concatenate(similarities), np.concatenate(indicators))
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        draws[draw] = fitted["mean_human"] - fitted["mean_llm"]
    return draws


def _scale_key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


def _sign(estimate: float) -> str:
    if estimate > 0:
        return "positive"
    if estimate < 0:
        return "negative"
    return "zero"
