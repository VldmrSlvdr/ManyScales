"""Deterministic text cleanup, then the recorded scale exclusion.

Text cleanup drops no rows. The only row deletion is TIVS, and that deletion
writes an audit row.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from h4.schemas import TEXT_COLUMNS, validate_items

_WHITESPACE = re.compile(r"\s+")


def preprocess_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize text, and sort by ``scale_id``, ``item_id``.

    Empty required fields raise. The analysis set is the file the analyst
    supplied, after Unicode and whitespace cleanup only.
    """

    df = validate_items(frame)
    for column in TEXT_COLUMNS:
        df[column] = df[column].map(_normalize_text)
    df = df.sort_values(["scale_id", "item_id"], kind="mergesort").reset_index(drop=True)
    df.attrs["text_normalization"] = "nfkc_strip_collapse_whitespace"
    return df


TIVS_EXCLUSION = {
    "scale_id": "11",
    "scale_name": "TIVS",
    "reason": (
        "Coauthors flagged the original English items for possible misuse. "
        "Preprocessing removes this scale from the primary analysis. "
        "The same rows stay in the with-TIVS table for the appendix robustness fit."
    ),
}


def exclude_flagged_scales(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove TIVS and return ``(primary items, one-row audit log)``.

    A table that does not contain TIVS is returned unchanged. The log still
    records that the step ran and that zero rows were removed.
    """

    work = frame.copy()
    scale_id = work["scale_id"].map(_scale_key)
    name = (
        work["scale_name"].astype(str)
        if "scale_name" in work.columns
        else pd.Series("", index=work.index)
    )
    flagged = scale_id.eq(TIVS_EXCLUSION["scale_id"]) | name.str.fullmatch(
        TIVS_EXCLUSION["scale_name"],
        case=False,
    )
    removed = work.loc[flagged]
    kept = work.loc[~flagged].reset_index(drop=True)
    observed_name = TIVS_EXCLUSION["scale_name"]
    if len(removed) and "scale_name" in removed.columns:
        observed_name = str(removed["scale_name"].iloc[0])
    log = pd.DataFrame(
        [
            {
                "step": "preprocess",
                "action": "exclude_scale",
                "scale_id": TIVS_EXCLUSION["scale_id"],
                "scale_name": observed_name,
                "n_items_removed": int(flagged.sum()),
                "n_items_before": int(len(work)),
                "n_items_after": int(len(kept)),
                "n_scales_before": int(scale_id.nunique()),
                "n_scales_after": int(scale_id.loc[~flagged].nunique()),
                "reason": TIVS_EXCLUSION["reason"],
                "primary_role": "excluded",
                "robustness_role": "retained_for_with_tivs_appendix",
            }
        ]
    )
    return kept, log


def _scale_key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", normalized).strip()
