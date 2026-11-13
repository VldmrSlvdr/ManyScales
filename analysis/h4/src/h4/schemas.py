"""Canonical item-table schema.

One row is one psychological-scale item. Provenance columns are retained
and are not predictors in the H4 model.
"""

from __future__ import annotations

import pandas as pd

from h4.errors import SchemaError

REQUIRED_COLUMNS = (
    "scale_id",
    "item_id",
    "original_en",
    "human_ja",
    "human_back_en",
    "llm_ja",
    "llm_back_en",
)

TEXT_COLUMNS = (
    "original_en",
    "human_ja",
    "human_back_en",
    "llm_ja",
    "llm_back_en",
)

OPTIONAL_COLUMNS = (
    "reverse_keyed",
    "human_translator_id",
    "backtranslation_method",
    "llm_translation_model",
    "llm_translation_date",
    "backtranslation_model",
    "record_status",
)

PROVENANCE_COLUMNS = (
    "human_translator_id",
    "backtranslation_method",
    "llm_translation_model",
    "llm_translation_date",
    "backtranslation_model",
    "record_status",
)

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def validate_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a column-cleaned copy, or raise ``SchemaError`` listing every problem."""

    if not isinstance(frame, pd.DataFrame):
        raise SchemaError("The item table must be a DataFrame.")
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    problems: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
        raise SchemaError("Item table schema failed: " + "; ".join(problems))

    for column in ("scale_id", "item_id"):
        df[column] = df[column].map(_as_id)
        empty_id = df[column].eq("")
        if empty_id.any():
            problems.append(f"{column} has {int(empty_id.sum())} empty values")

    duplicated = df.duplicated(["scale_id", "item_id"], keep=False)
    if duplicated.any():
        keys = (
            df.loc[duplicated, ["scale_id", "item_id"]]
            .drop_duplicates()
            .astype(str)
            .agg(" / ".join, axis=1)
            .tolist()
        )
        problems.append("duplicate scale_id × item_id: " + ", ".join(keys[:12]))

    for column in TEXT_COLUMNS:
        text = df[column].map(_as_text)
        if text.isna().any():
            problems.append(f"{column} has {int(text.isna().sum())} empty values")
        df[column] = text

    if "reverse_keyed" in df.columns:
        try:
            df["reverse_keyed"] = df["reverse_keyed"].map(_as_bool)
        except ValueError as exc:
            problems.append(str(exc))

    if problems:
        raise SchemaError("Item table schema failed: " + "; ".join(problems))
    return df


def _as_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("reverse_keyed contains a missing value")
    token = str(value).strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise ValueError(f"reverse_keyed value {value!r} is not a boolean")
