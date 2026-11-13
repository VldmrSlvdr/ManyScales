"""Tie each survey column to the question text stored in the Qualtrics file."""

from __future__ import annotations

import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from h4.catalog import load_catalog

_TAG_ITEM = re.compile(r"^([A-F])([0-9]{2})_item([0-9]+)_(human|ai)$")
_TAG_PROBLEM = re.compile(r"^([A-F])([0-9]{2})_problem_items_(human|ai)$")
_TAG_RATING = re.compile(
    r"^([A-F])([0-9]{2})_(semantic|naturalness|cultural_validity|choice_fit|wording)_(human|ai)$"
)
_SCALE_PREFIX = re.compile(r"^([A-F])([0-9]{2})_")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: object) -> str:
    raw = "" if value is None else str(value)
    parser = _TextExtractor()
    parser.feed(raw)
    text = " ".join("".join(parser.parts).split())
    return text


def _norm(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", plain_text(value)).split())


def _choices(payload: dict[str, Any]) -> list[dict[str, str]]:
    choices = payload.get("Choices") or {}
    if not isinstance(choices, dict):
        return []
    order = [str(item) for item in (payload.get("ChoiceOrder") or choices.keys())]
    rows = []
    for key in order:
        block = choices.get(key) or choices.get(int(key) if key.isdigit() else key) or {}
        if isinstance(block, dict):
            rows.append({"code": key, "label": plain_text(block.get("Display", ""))})
    return rows


def _answers(payload: dict[str, Any]) -> list[dict[str, str]]:
    answers = payload.get("Answers") or {}
    if not isinstance(answers, dict):
        return []
    order = [str(item) for item in (payload.get("AnswerOrder") or answers.keys())]
    rows = []
    for key in order:
        block = answers.get(key) or {}
        if isinstance(block, dict):
            rows.append({"code": key, "label": plain_text(block.get("Display", ""))})
    return rows


def questions_from_qsf(path: Path) -> list[dict[str, Any]]:
    """One record per exported column a respondent could answer."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for element in document.get("SurveyElements") or []:
        if element.get("Element") != "SQ":
            continue
        payload = element.get("Payload") or {}
        tag = str(payload.get("DataExportTag") or "")
        if not tag:
            continue
        stem = plain_text(payload.get("QuestionText", ""))
        qid = str(payload.get("QuestionID") or "")
        item_match = _TAG_ITEM.match(tag)
        problem_match = _TAG_PROBLEM.match(tag)
        if problem_match:
            survey_set, scale_id, condition = problem_match.groups()
            for choice in _choices(payload):
                for answer in _answers(payload):
                    records.append(
                        {
                            "column": f"{tag}_{choice['code']}_{answer['code']}",
                            "question_id": qid,
                            "export_tag": tag,
                            "set_id": survey_set,
                            "scale_id": scale_id,
                            "item_id": choice["code"].zfill(2),
                            "condition": condition,
                            "dimension": answer["label"],
                            "role": "item_problem",
                            "stem": stem,
                            "shown_text": choice["label"],
                            "choices": [{"code": "1", "label": "Checked"}],
                        }
                    )
            continue
        if item_match:
            survey_set, scale_id, item_number, condition = item_match.groups()
            records.append(
                {
                    "column": tag,
                    "question_id": qid,
                    "export_tag": tag,
                    "set_id": survey_set,
                    "scale_id": scale_id,
                    "item_id": item_number.zfill(2),
                    "condition": condition,
                    "dimension": "",
                    "role": "item_response",
                    "stem": stem,
                    "shown_text": stem,
                    "choices": _choices(payload),
                }
            )
            continue
        rating = _TAG_RATING.match(tag)
        prefix = _SCALE_PREFIX.match(tag)
        role = "scale_rating" if rating else _role(tag)
        records.append(
            {
                "column": tag,
                "question_id": qid,
                "export_tag": tag,
                "set_id": rating.group(1) if rating else (prefix.group(1) if prefix else ""),
                "scale_id": rating.group(2) if rating else (prefix.group(2) if prefix else ""),
                "item_id": "",
                "condition": rating.group(4) if rating else _condition(tag),
                "dimension": rating.group(3) if rating else "",
                "role": role,
                "stem": stem,
                "shown_text": stem,
                "choices": _choices(payload),
            }
        )
    return records


def _role(tag: str) -> str:
    lowered = tag.lower()
    if "dqs" in lowered or "trap" in lowered:
        return "attention_check"
    if lowered.startswith("eligibility") or lowered in {"consent", "age", "gender"}:
        return "eligibility"
    if "timer" in lowered:
        return "timer"
    if "inst" in lowered:
        return "instruction"
    if "preference" in lowered or "llm_guess" in lowered:
        return "preference"
    return "other"


def _condition(tag: str) -> str:
    if tag.endswith("_human"):
        return "human"
    if tag.endswith("_ai"):
        return "llm"
    return ""


def attach_catalog(records: list[dict[str, Any]], catalog_path: Path) -> list[dict[str, Any]]:
    """Add workbook item text and whether the survey showed that text."""

    catalog = load_catalog(catalog_path)
    names = {}
    items = {}
    for scale in catalog["scales"]:
        if scale.get("status") != "study":
            continue
        names[str(scale["scale_id"])] = str(scale.get("short_name") or scale["scale_id"])
        for item in scale["items"]:
            items[(str(scale["scale_id"]), str(item["item_id"]).zfill(2))] = item
    attached = []
    for record in records:
        row = dict(record)
        key = (row.get("scale_id") or "", str(row.get("item_id") or "").zfill(2) if row.get("item_id") else "")
        item = items.get(key) if row.get("item_id") else None
        row["scale_name"] = names.get(row.get("scale_id") or "", "")
        row["original_en"] = "" if item is None else str(item.get("original_en") or "")
        row["human_ja"] = "" if item is None else str(item.get("human_ja") or "")
        row["llm_ja"] = "" if item is None else str(item.get("llm_ja") or "")
        expected = ""
        if item is not None and row.get("condition") == "human":
            expected = row["human_ja"]
        elif item is not None and row.get("condition") in {"ai", "llm"}:
            expected = row["llm_ja"]
        shown = row.get("shown_text") or ""
        if expected and shown and row.get("role") in {"item_response", "item_problem"}:
            row["matches_workbook"] = _norm(shown) == _norm(expected)
        else:
            row["matches_workbook"] = None
        attached.append(row)
    return attached


def codebook_frame(records: list[dict[str, Any]], survey: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame.insert(0, "survey", survey)
    if "choices" in frame.columns:
        frame["choices"] = frame["choices"].map(
            lambda rows: " | ".join(f"{row['code']}. {row['label']}" for row in rows) if isinstance(rows, list) else ""
        )
    return frame
