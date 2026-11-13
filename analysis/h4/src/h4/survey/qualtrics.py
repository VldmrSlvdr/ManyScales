"""Read a Qualtrics TSV without dropping the question-text row."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from h4.errors import SchemaError


def read_qualtrics_tsv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return response rows and the two Qualtrics header rows.

    The file is UTF-16, as exported from Qualtrics. Row 1 is the question
    wording, row 2 is the ImportId JSON, and later rows are respondents.
    Blank cells and the literal ``NA`` are missing, matching the Quarto read.
    """

    path = Path(path)
    if not path.exists():
        raise SchemaError(f"Qualtrics export not found at {path}.")
    raw = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-16",
        header=None,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    if len(raw) < 3:
        raise SchemaError(f"{path} has no respondent rows under the Qualtrics header.")
    names = [_header_name(value, index) for index, value in enumerate(raw.iloc[0].tolist())]
    if len(names) != len(set(names)):
        raise SchemaError(f"{path} has duplicate column names.")
    header = pd.DataFrame(
        {
            "column": names,
            "question_text": raw.iloc[1].tolist(),
            "import_id": raw.iloc[2].tolist(),
        }
    )
    frame = raw.iloc[3:].copy()
    frame.columns = names
    frame = frame.replace({"": pd.NA, "NA": pd.NA}).reset_index(drop=True)
    return frame, header


def _header_name(value: object, index: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise SchemaError(f"Qualtrics column {index + 1} has no variable name.")
    return text


def import_qid(import_id: object) -> str:
    """Pull the Qualtrics question id out of an ImportId cell."""

    text = "" if import_id is None or (isinstance(import_id, float)) else str(import_id)
    if not text or text == "nan":
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("ImportId", ""))
    return ""
