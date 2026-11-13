"""Build the tidy similarity table.

This layer knows which column is the human text and which is the LLM text.
It does not fit a model and it does not decide whether H4 is supported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h4.config import Config, assign_tier
from h4.similarity import score_vectors


def iter_specifications(config: Config):
    """Yield comparison, metric, and the tier label for every planned score."""

    comparisons = ["back_translation"]
    if config.direct_crosslingual:
        comparisons.append("direct_crosslingual")
    metrics = (config.primary_metric, *config.sensitivity_metrics)
    for model in config.models:
        for comparison in comparisons:
            for metric in metrics:
                yield model, comparison, metric, assign_tier(
                    comparison, model.name, metric.name, config
                )


def score_items(
    items: pd.DataFrame,
    vectors_by_field: dict[str, np.ndarray],
    *,
    model_name: str,
    short_name: str,
    metric_name: str,
    independent_geometry: bool,
    comparison: str,
    fields: dict[str, str],
    analysis_tier: str,
) -> pd.DataFrame:
    reference = vectors_by_field[fields["reference"]]
    rows: list[dict] = []
    for condition in ("human", "llm"):
        similarity, raw = score_vectors(
            metric_name,
            reference,
            vectors_by_field[fields[condition]],
        )
        for position, record in enumerate(items.itertuples(index=False)):
            rows.append(
                {
                    "scale_id": str(record.scale_id),
                    "item_id": str(record.item_id),
                    "translation": condition,
                    "comparison": comparison,
                    "embedding_model": model_name,
                    "short_name": short_name,
                    "metric": metric_name,
                    "independent_geometry": bool(independent_geometry),
                    "similarity": float(similarity[position]),
                    "raw_value": float(raw[position]),
                    "analysis_tier": analysis_tier,
                }
            )
    return pd.DataFrame(rows)
