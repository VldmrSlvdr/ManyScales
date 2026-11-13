"""Paired token distances from Hara et al. (2026), without cosine similarity.

Each translation is already scored against the original token cloud. This
table keeps only the two Hara distances

    d_mu = ||mu_original - mu_translation||^2 / 4
    d_sigma = BuresWasserstein(Sigma_original, Sigma_translation)^2 / 4

and the human-minus-LLM difference of each. The composite SOCM score and the
mean-vector cosine stay out of this table. The contrast is exploratory.
"""

from __future__ import annotations

import pandas as pd

_KEYS = ["scale_id", "item_id", "comparison", "embedding_model", "short_name"]
_DISTANCES = ["d_mu", "d_sigma"]


def paired_token_contrast(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per item, model, and comparison, with human and LLM distances."""

    if pairs.empty:
        return pairs
    human = pairs.loc[pairs["translation"] == "human", _KEYS + _DISTANCES]
    llm = pairs.loc[pairs["translation"] == "llm", _KEYS + _DISTANCES]
    merged = human.merge(llm, on=_KEYS, suffixes=("_human", "_llm"))
    merged["delta_d_mu"] = merged["d_mu_human"] - merged["d_mu_llm"]
    merged["delta_d_sigma"] = merged["d_sigma_human"] - merged["d_sigma_llm"]
    merged["analysis_tier"] = "exploratory"
    merged["directional_test"] = False
    columns = _KEYS + [
        "d_mu_human",
        "d_mu_llm",
        "delta_d_mu",
        "d_sigma_human",
        "d_sigma_llm",
        "delta_d_sigma",
        "analysis_tier",
        "directional_test",
    ]
    return merged.loc[:, columns].reset_index(drop=True)


def paired_token_summary(contrast: pd.DataFrame) -> pd.DataFrame:
    """Item means of the two distances and of the human-minus-LLM gaps."""

    if contrast.empty:
        return contrast
    rows = []
    group_keys = ["short_name", "embedding_model", "comparison"]
    for key, group in contrast.groupby(group_keys, sort=True):
        short_name, model, comparison = key
        rows.append(
            {
                "short_name": short_name,
                "embedding_model": model,
                "comparison": comparison,
                "n_items": int(len(group)),
                "mean_d_mu_human": float(group["d_mu_human"].mean()),
                "mean_d_mu_llm": float(group["d_mu_llm"].mean()),
                "mean_delta_d_mu": float(group["delta_d_mu"].mean()),
                "mean_d_sigma_human": float(group["d_sigma_human"].mean()),
                "mean_d_sigma_llm": float(group["d_sigma_llm"].mean()),
                "mean_delta_d_sigma": float(group["delta_d_sigma"].mean()),
                "analysis_tier": "exploratory",
                "directional_test": False,
            }
        )
    return pd.DataFrame(rows)
