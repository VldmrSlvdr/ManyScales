"""Metric checks that never read the human-versus-LLM contrast.

The builder keeps only ``scale_id``, ``item_id``, and ``original_en``.
Diagnostic paraphrase files are rejected when they carry translation columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from h4.embedders.base import Embedder
from h4.errors import SchemaError
from h4.similarity import score_vectors
from h4.validation.lexical_overlap import jaccard
from h4.validation.negative_controls import between_scale_rows, within_scale_rows
from h4.validation.positive_controls import identical_rows, negation_rows

FORBIDDEN_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "translation",
        "condition",
        "human_back_en",
        "llm_back_en",
        "human_ja",
        "llm_ja",
        "sim_human",
        "sim_llm",
        "delta",
    }
)

_BLINDED_COLUMNS = ("scale_id", "item_id", "original_en")


def build_validation_pairs(
    items: pd.DataFrame,
    seed: int,
    *,
    include_identical: bool = True,
    include_within: bool = True,
    include_between: bool = True,
    include_negation: bool = True,
) -> pd.DataFrame:
    blinded = items.loc[:, list(_BLINDED_COLUMNS)].copy()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    if include_identical:
        rows.extend(identical_rows(blinded))
    if include_negation:
        rows.extend(negation_rows(blinded))
    if include_within:
        rows.extend(within_scale_rows(blinded, rng))
    if include_between:
        rows.extend(between_scale_rows(blinded, rng))
    if not rows:
        return pd.DataFrame(
            columns=[
                "scale_id",
                "item_id",
                "other_scale_id",
                "other_item_id",
                "pair_type",
                "text_a",
                "text_b",
            ]
        )
    return pd.DataFrame(rows)


def load_diagnostic_pairs(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["text_a", "text_b", "pair_type"])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    blocked = sorted(set(frame.columns) & FORBIDDEN_DIAGNOSTIC_COLUMNS)
    if blocked:
        raise SchemaError(
            "Diagnostic pairs must not carry translation labels. Found: " + ", ".join(blocked)
        )
    for column in ("text_a", "text_b", "pair_type"):
        if column not in frame.columns:
            raise SchemaError(f"Diagnostic pairs are missing {column}.")
    rows = []
    for record in frame.itertuples(index=False):
        kind = str(record.pair_type).strip()
        if not kind.startswith("diagnostic_"):
            kind = f"diagnostic_{kind}"
        rows.append(
            {
                "scale_id": "",
                "item_id": "",
                "other_scale_id": "",
                "other_item_id": "",
                "pair_type": kind,
                "text_a": str(record.text_a).strip(),
                "text_b": str(record.text_b).strip(),
            }
        )
    return pd.DataFrame(rows)


def score_control_pairs(
    pairs: pd.DataFrame,
    embedder: Embedder,
    metric: str,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    texts = pd.unique(pd.concat([pairs["text_a"], pairs["text_b"]], ignore_index=True))
    matrix = embedder.encode([str(text) for text in texts])
    lookup = {text: matrix[index] for index, text in enumerate(texts)}
    left = np.vstack([lookup[text] for text in pairs["text_a"]])
    right = np.vstack([lookup[text] for text in pairs["text_b"]])
    similarity, raw = score_vectors(metric, left, right)
    scored = pairs.copy()
    scored["similarity"] = similarity
    scored["raw_value"] = raw
    scored["jaccard"] = [
        jaccard(left_text, right_text)
        for left_text, right_text in zip(pairs["text_a"], pairs["text_b"], strict=True)
    ]
    scored["embedding_model"] = embedder.name
    scored["metric"] = metric
    scored["stage"] = "metric_validation"
    return scored


def summarize_controls(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    summary = (
        scored.groupby("pair_type", sort=True)
        .agg(
            n=("similarity", "size"),
            mean_similarity=("similarity", "mean"),
            sd_similarity=("similarity", "std"),
            mean_jaccard=("jaccard", "mean"),
        )
        .reset_index()
    )
    summary["sd_similarity"] = summary["sd_similarity"].fillna(0.0)
    summary["embedding_model"] = scored["embedding_model"].iloc[0]
    summary["metric"] = scored["metric"].iloc[0]
    summary["stage"] = "metric_validation"
    summary["order_ok"] = control_order_ok(summary)
    summary["identical_near_one"] = _identical_near_one(summary)
    summary["model_selection"] = "frozen"
    summary["action_on_failure"] = "report_only"
    return summary


def control_order_ok(summary: pd.DataFrame) -> bool:
    means = summary.set_index("pair_type")["mean_similarity"]
    needed = ("identical", "within_scale_permutation", "between_scale_permutation")
    if any(name not in means.index for name in needed):
        return False
    return bool(
        means["identical"] > means["within_scale_permutation"] > means["between_scale_permutation"]
    )


def _identical_near_one(summary: pd.DataFrame) -> bool:
    means = summary.set_index("pair_type")["mean_similarity"]
    if "identical" not in means.index:
        return False
    return bool(means["identical"] >= 0.99)
