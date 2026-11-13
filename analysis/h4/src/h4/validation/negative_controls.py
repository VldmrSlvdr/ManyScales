"""Negative controls built only from original English items."""

from __future__ import annotations

import numpy as np
import pandas as pd


def within_scale_rows(items: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    blinded = items[["scale_id", "item_id", "original_en"]]
    for _, group in blinded.groupby("scale_id", sort=True):
        group = group.reset_index(drop=True)
        count = len(group)
        if count < 2:
            continue
        order = _derange(count, rng)
        for left, right in enumerate(order):
            source = group.iloc[left]
            other = group.iloc[int(right)]
            rows.append(
                {
                    "scale_id": str(source.scale_id),
                    "item_id": str(source.item_id),
                    "other_scale_id": str(other.scale_id),
                    "other_item_id": str(other.item_id),
                    "pair_type": "within_scale_permutation",
                    "text_a": source.original_en,
                    "text_b": other.original_en,
                }
            )
    return rows


def between_scale_rows(items: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    records = items[["scale_id", "item_id", "original_en"]].to_dict("records")
    rows: list[dict] = []
    if len({record["scale_id"] for record in records}) < 2:
        return rows
    for source in records:
        pool = [record for record in records if record["scale_id"] != source["scale_id"]]
        choice = pool[int(rng.integers(0, len(pool)))]
        rows.append(
            {
                "scale_id": str(source["scale_id"]),
                "item_id": str(source["item_id"]),
                "other_scale_id": str(choice["scale_id"]),
                "other_item_id": str(choice["item_id"]),
                "pair_type": "between_scale_permutation",
                "text_a": source["original_en"],
                "text_b": choice["original_en"],
            }
        )
    return rows


def _derange(count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 2:
        return np.arange(count)
    for _ in range(100):
        order = rng.permutation(count)
        if np.all(order != np.arange(count)):
            return order
    return np.roll(np.arange(count), 1)
