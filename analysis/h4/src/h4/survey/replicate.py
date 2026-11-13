"""Reproduce the Quarto survey cut and check it against the saved tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from h4.io import write_json
from h4.survey.codebook import attach_catalog, codebook_frame, questions_from_qsf
from h4.survey.datasets import (
    analysis_rows_citizen,
    analysis_rows_expert,
    citizen_sets,
    expert_sets,
    write_flags,
)
from h4.survey.flags import add_citizen_flags, add_expert_flags
from h4.survey.page import write_codebook_page
from h4.survey.qualtrics import read_qualtrics_tsv

_ROOT = Path(__file__).resolve().parents[3]
_ANALYSIS = _ROOT.parent / "ratings/primary"
_SURVEY = _ROOT / "data" / "survey"


def main() -> None:
    citizen = _run_citizen()
    expert = _run_expert()
    codebook = _write_codebook()
    summary = {"citizen": citizen, "expert": expert, "codebook": codebook}
    write_json(summary, _SURVEY / "verification.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_citizen() -> dict:
    raw_path = _ANALYSIS / "data/0_raw_data/2026_ManyScales_一般_2026年7月29日_06.39.tsv"
    frame, _header = read_qualtrics_tsv(raw_path)
    flagged = add_citizen_flags(frame)
    flags_path = _SURVEY / "1_flags/ManyScales_一般_analysis_flags.tsv"
    write_flags(flagged, flags_path)
    all_counts = citizen_sets(flagged, _SURVEY / "2_all", "ManyScales_市民_all_")
    kept = analysis_rows_citizen(flagged)
    processed_counts = citizen_sets(kept, _SURVEY / "3_processed", "ManyScales_市民_processed_")
    return {
        "n_respondents": int(len(flagged)),
        "analysis_target": int(flagged["analysis_target"].sum()),
        "excl1": int(flagged["excl1"].sum()),
        "excl2": int(flagged["excl2"].sum()),
        "excl3": int(flagged["excl3"].sum()),
        "processed": processed_counts,
        "all_sets": all_counts,
        "matches_saved_flags": _same_table(
            flagged,
            _ANALYSIS / "data/1_data_flags/ManyScales_一般_analysis_flags.tsv",
            sep="\t",
        ),
        "matches_saved_processed": _same_sets(
            _SURVEY / "3_processed",
            _ANALYSIS / "data/3_data_processed",
            "ManyScales_市民_processed_",
        ),
    }


def _run_expert() -> dict:
    raw_path = _ANALYSIS / "data/0_raw_data/2026_ManyScales_専門_2026年9月16日_08.46.tsv"
    frame, _header = read_qualtrics_tsv(raw_path)
    flagged = add_expert_flags(frame)
    write_flags(flagged, _SURVEY / "1_flags/ManyScales_専門_analysis_flags.tsv")
    all_counts = expert_sets(flagged, _SURVEY / "2_all", "ManyScales_専門家_all_")
    kept = analysis_rows_expert(flagged)
    processed_counts = expert_sets(kept, _SURVEY / "3_processed", "ManyScales_専門家_processed_")
    return {
        "n_respondents": int(len(flagged)),
        "analysis_target": int(flagged["analysis_target"].sum()),
        "excl1": int(flagged["excl1"].sum()),
        "processed": processed_counts,
        "all_sets": all_counts,
        "matches_saved_flags": _same_table(
            flagged,
            _ANALYSIS / "data/1_data_flags/ManyScales_専門_analysis_flags.tsv",
            sep="\t",
        ),
        "matches_saved_processed": _same_sets(
            _SURVEY / "3_processed",
            _ANALYSIS / "data/3_data_processed",
            "ManyScales_専門家_processed_",
        ),
    }


def _write_codebook() -> dict:
    catalog = _ROOT / "data/raw/scales.json"
    citizen = attach_catalog(
        questions_from_qsf(_ROOT / "data/survey/questionnaires/2026_ManyScales_一般（調査票）.qsf"),
        catalog,
    )
    expert = attach_catalog(
        questions_from_qsf(_ROOT / "data/survey/questionnaires/2026_ManyScales_専門（調査票）.qsf"),
        catalog,
    )
    frame = pd.concat(
        [codebook_frame(citizen, "citizen"), codebook_frame(expert, "expert")],
        ignore_index=True,
    )
    destination = _SURVEY / "question_codebook.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    page_records = {
        "citizen": [_page_row(row) for row in citizen],
        "expert": [_page_row(row) for row in expert],
    }
    write_codebook_page(page_records, _ROOT / "review/survey_codebook.html")
    item_rows = frame.loc[frame["role"].isin(["item_response", "item_problem"])]
    compared = item_rows.loc[item_rows["matches_workbook"].notna()]
    return {
        "n_columns": int(len(frame)),
        "citizen_columns": int((frame["survey"] == "citizen").sum()),
        "expert_columns": int((frame["survey"] == "expert").sum()),
        "item_text_compared": int(len(compared)),
        "item_text_matches_workbook": int(compared["matches_workbook"].sum()),
    }


def _page_row(row: dict) -> dict:
    choices = row.get("choices") or []
    return {
        "column": row["column"],
        "role": row["role"],
        "set_id": row["set_id"],
        "scale_id": row["scale_id"],
        "scale_name": row.get("scale_name", ""),
        "item_id": row["item_id"],
        "condition": row["condition"],
        "dimension": row["dimension"],
        "stem": row["stem"],
        "shown_text": row["shown_text"],
        "choices": " | ".join(f"{choice['code']}. {choice['label']}" for choice in choices),
        "original_en": row.get("original_en", ""),
        "human_ja": row.get("human_ja", ""),
        "llm_ja": row.get("llm_ja", ""),
        "matches_workbook": row.get("matches_workbook"),
    }


def _same_sets(ours: Path, reference: Path, prefix: str) -> dict:
    reports = {}
    for survey_set in ("A", "B", "C", "D", "E", "F"):
        name = f"{prefix}{survey_set}.csv"
        reports[survey_set] = _same_table(
            pd.read_csv(ours / name, dtype=str, keep_default_na=False),
            reference / name,
            sep=",",
        )
    reports["all_match"] = all(item["match"] for item in reports.values())
    return reports


def _same_table(frame: pd.DataFrame, reference: Path, *, sep: str) -> dict:
    saved = pd.read_csv(reference, sep=sep, dtype=str, keep_default_na=False)
    left = _text(frame)
    right = _text(saved)
    missing = [column for column in right.columns if column not in left.columns]
    extra = [column for column in left.columns if column not in right.columns]
    if missing or extra or len(left) != len(right):
        return {
            "match": False,
            "rows": [int(len(left)), int(len(right))],
            "missing_columns": missing[:12],
            "extra_columns": extra[:12],
        }
    aligned = left.loc[:, list(right.columns)].reset_index(drop=True)
    right = right.reset_index(drop=True)
    mismatch = aligned.ne(right)
    n_bad = int(mismatch.to_numpy().sum())
    sample = []
    if n_bad:
        stacked = mismatch.stack()
        for (row_index, column), bad in stacked.items():
            if not bad:
                continue
            sample.append(
                {
                    "row": int(row_index),
                    "column": str(column),
                    "ours": aligned.iat[int(row_index), aligned.columns.get_loc(column)],
                    "saved": right.iat[int(row_index), right.columns.get_loc(column)],
                }
            )
            if len(sample) >= 5:
                break
    return {"match": n_bad == 0, "mismatched_cells": n_bad, "sample": sample}


def _text(frame: pd.DataFrame) -> pd.DataFrame:
    text = frame.fillna("").astype(str).replace({"<NA>": "", "nan": "", "None": ""})
    for column in ("StartDate", "EndDate", "RecordedDate"):
        if column in text.columns:
            text[column] = text[column].str.replace("T", " ", regex=False).str.replace("Z", "", regex=False)
    return text


if __name__ == "__main__":
    main()
