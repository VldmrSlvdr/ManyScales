"""Positive controls: an item compared with itself, and optional paraphrases."""

from __future__ import annotations

import pandas as pd

from h4.validation.negation import negate


def identical_rows(items: pd.DataFrame) -> list[dict]:
    rows = []
    for record in items.itertuples(index=False):
        rows.append(
            {
                "scale_id": str(record.scale_id),
                "item_id": str(record.item_id),
                "other_scale_id": str(record.scale_id),
                "other_item_id": str(record.item_id),
                "pair_type": "identical",
                "text_a": record.original_en,
                "text_b": record.original_en,
            }
        )
    return rows


def negation_rows(items: pd.DataFrame) -> list[dict]:
    rows = []
    for record in items.itertuples(index=False):
        rows.append(
            {
                "scale_id": str(record.scale_id),
                "item_id": str(record.item_id),
                "other_scale_id": str(record.scale_id),
                "other_item_id": str(record.item_id),
                "pair_type": "negation",
                "text_a": record.original_en,
                "text_b": negate(record.original_en),
            }
        )
    return rows
