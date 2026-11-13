"""Scale catalog extracted from the ManyScales workbook.

Each scale keeps its sheet metadata, the instruction, the response
labels, and the items. Every one of those texts keeps the original
English, the manual Japanese, and the AI Japanese. Back-translations
and reconciled wordings stay beside them because the H4 item table is
built from this catalog.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from h4.errors import SchemaError
from h4.preprocess import exclude_flagged_scales, preprocess_items

_LINEBREAK = re.compile(r"\s*[\r\n]+\s*")
_SHEET_NUMBER = re.compile(r"^(\d+)\s+(.*)$")

TEXT_KEYS = (
    "original_en",
    "human_ja",
    "llm_ja",
    "human_back_en",
    "llm_back_en",
    "human_ja_reconciled",
    "llm_ja_reconciled",
    "human_reconcile_note",
    "llm_reconcile_note",
)

META_PREFIXES = (
    ("メンバー", "team"),
    ("対象母集団", "population"),
    ("尺度名(original)", "name_en"),
    ("尺度名 in Japanese", "name_ja"),
    ("概念", "construct"),
    ("項目数", "n_items_declared"),
    ("回答形式", "response_format"),
    ("因子数", "factors"),
    ("Index", "scoring"),
    ("原版の信頼性", "reliability"),
    ("原版の妥当性", "validity"),
    ("doi", "doi"),
    ("論文情報", "citation"),
    ("翻訳過程", "translation_process"),
    ("特記事項", "notes"),
)

_REVERSE_TRUE = {"r", "reverse", "yes", "true", "y", "1", "逆転"}
_OVERVIEW_SKIP = "対象とする尺度について"


def extract_workbook(path: Path) -> dict[str, Any]:
    """Read the scale workbook into an ordered catalog document."""

    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        registry = _overview(workbook["対象尺度 の詳細"]) if "対象尺度 の詳細" in workbook.sheetnames else []
        scales = []
        for sheet_name in workbook.sheetnames:
            if sheet_name in {"実査ログ", "調査票レビュー", "特任研究員作業メモ", "対象尺度 の詳細"}:
                continue
            scales.append(_scale_sheet(workbook[sheet_name], sheet_name, registry))
    finally:
        workbook.close()
    return {
        "document": "manyscales_scale_catalog",
        "source_workbook": path.name,
        "text_fields": {
            "original_en": "Original English",
            "human_ja": "Manual Japanese translation",
            "llm_ja": "AI Japanese translation",
            "human_back_en": "English back-translation of the manual Japanese",
            "llm_back_en": "English back-translation of the AI Japanese",
            "human_ja_reconciled": "Reconciled manual Japanese",
            "llm_ja_reconciled": "Reconciled AI Japanese",
            "human_reconcile_note": "Note on the manual reconciliation",
            "llm_reconcile_note": "Note on the AI reconciliation",
        },
        "n_scales": len(scales),
        "scales": scales,
    }


def write_catalog(catalog: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SchemaError(
            f"Scale catalog not found at {path}. "
            "Build it with scripts/00_extract_scales.py."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("document") != "manyscales_scale_catalog" or "scales" not in document:
        raise SchemaError(f"{path} is not a ManyScales scale catalog.")
    return document


def items_from_catalog(catalog: dict[str, Any]) -> pd.DataFrame:
    """Study items only. Instructions and labels stay in the catalog."""

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for scale in catalog["scales"]:
        if scale.get("status") != "study":
            continue
        registry = scale.get("registry") or {}
        for item in scale["items"]:
            record = {
                "scale_id": scale["scale_id"],
                "item_id": item["item_id"],
                "original_en": item["original_en"],
                "human_ja": item["human_ja"],
                "human_back_en": item["human_back_en"],
                "llm_ja": item["llm_ja"],
                "llm_back_en": item["llm_back_en"],
                "reverse_keyed": bool(item["reverse_keyed"]),
                "factor": item.get("factor", ""),
                "survey_set": registry.get("survey_set", ""),
                "human_translator_id": registry.get("team", ""),
                "llm_translation_model": registry.get("forward_llm", ""),
                "backtranslation_model": registry.get("back_llm", ""),
                "backtranslation_method": "workbook_back_translation",
                "record_status": "study",
                "scale_name": scale.get("short_name", ""),
                "source_sheet": scale.get("sheet", ""),
                "source_row": item.get("source_row", ""),
            }
            missing = [
                column
                for column in (
                    "original_en",
                    "human_ja",
                    "human_back_en",
                    "llm_ja",
                    "llm_back_en",
                )
                if not str(record[column]).strip()
            ]
            if missing:
                problems.append(
                    f"{scale['scale_id']} item {item['item_id']} missing {', '.join(missing)}"
                )
            rows.append(record)
    if problems:
        preview = "; ".join(problems[:8])
        raise SchemaError(
            f"{len(problems)} study items are missing a text required for H4. {preview}"
        )
    if not rows:
        raise SchemaError("The scale catalog has no study items.")
    return pd.DataFrame(rows)


def study_items_from_catalog(catalog: dict[str, Any]) -> pd.DataFrame:
    """Every study item, including TIVS. This is the robustness input."""

    return preprocess_items(items_from_catalog(catalog))


def analysis_items_from_catalog(catalog: dict[str, Any]) -> pd.DataFrame:
    """Primary items. TIVS is removed by ``exclude_flagged_scales``."""

    primary, _log = exclude_flagged_scales(study_items_from_catalog(catalog))
    return primary


def write_preanalysis_tables(catalog: dict[str, Any], directory: Path) -> None:
    """Flat tables for the instruction, labels, and scale metadata."""

    directory.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    instruction_rows = []
    label_rows = []
    for scale in catalog["scales"]:
        registry = scale.get("registry") or {}
        sheet_meta = scale.get("sheet_meta") or {}
        meta_rows.append(
            {
                "scale_id": scale["scale_id"],
                "short_name": scale.get("short_name", ""),
                "sheet": scale.get("sheet", ""),
                "status": scale.get("status", ""),
                "survey_set": registry.get("survey_set", ""),
                "team": registry.get("team") or sheet_meta.get("team", ""),
                "name_en": registry.get("name") or sheet_meta.get("name_en", ""),
                "name_ja": sheet_meta.get("name_ja", ""),
                "construct": registry.get("construct") or sheet_meta.get("construct", ""),
                "population": registry.get("population") or sheet_meta.get("population", ""),
                "response_format": registry.get("response_format") or sheet_meta.get("response_format", ""),
                "n_items_declared": registry.get("n_items") or sheet_meta.get("n_items_declared", ""),
                "n_items_extracted": len(scale.get("items") or []),
                "n_labels": len(scale.get("labels") or []),
                "forward_llm": registry.get("forward_llm", ""),
                "back_llm": registry.get("back_llm", ""),
                "reconcile_llm": registry.get("reconcile_llm", ""),
                "doi": sheet_meta.get("doi", ""),
            }
        )
        instruction = dict(scale.get("instruction") or _empty_text())
        instruction_rows.append(
            {
                "scale_id": scale["scale_id"],
                "short_name": scale.get("short_name", ""),
                "status": scale.get("status", ""),
                **{key: instruction.get(key, "") for key in TEXT_KEYS},
            }
        )
        for label in scale.get("labels") or []:
            label_rows.append(
                {
                    "scale_id": scale["scale_id"],
                    "short_name": scale.get("short_name", ""),
                    "label_id": label.get("label_id", ""),
                    "score": label.get("score", ""),
                    **{key: label.get(key, "") for key in TEXT_KEYS},
                }
            )
    pd.DataFrame(meta_rows).to_csv(directory / "scale_meta.csv", index=False)
    pd.DataFrame(instruction_rows).to_csv(directory / "instructions.csv", index=False)
    pd.DataFrame(label_rows).to_csv(directory / "labels.csv", index=False)


def _scale_sheet(worksheet, sheet_name: str, registry: list[dict[str, str]]) -> dict[str, Any]:
    rows = [
        [_clean(cell) for cell in row]
        for row in worksheet.iter_rows(max_col=12, values_only=True)
    ]
    scale_id, short_name = _sheet_identity(sheet_name)
    status = _status(sheet_name, scale_id)
    meta_rows, layout_notes, instruction, labels, items = _blocks(rows)
    sheet_meta = _canonical_meta(meta_rows)
    matched = _match_registry(scale_id, short_name, sheet_name, sheet_meta, registry)
    return {
        "scale_id": scale_id,
        "short_name": short_name,
        "sheet": sheet_name,
        "status": status,
        "registry_matched": bool(matched),
        "registry": matched,
        "sheet_meta": sheet_meta,
        "sheet_meta_as_written": meta_rows,
        "layout_notes": layout_notes,
        "instruction": instruction,
        "labels": labels,
        "items": items,
    }


def _blocks(rows: list[list[str]]):
    meta_rows: list[dict[str, str]] = []
    layout_notes: list[str] = []
    instruction = _empty_text()
    labels: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    mode = "meta"
    seen_item_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        padded = row + [""] * (12 - len(row))
        marker = padded[0]
        if marker == "教示":
            mode = "instruction"
            continue
        if marker == "選択肢とラベル":
            mode = "labels"
            continue
        if marker == "項目":
            mode = "items"
            continue
        if mode == "meta":
            if marker and marker != "尺度概要" and padded[1] and not marker.startswith(_OVERVIEW_SKIP):
                meta_rows.append({"label": marker, "value": padded[1]})
            elif not marker and padded[1] and padded[1] not in {"内容", "Original in English"}:
                layout_notes.append(padded[1])
            continue
        if mode == "instruction":
            block = _text_block(padded)
            if any(block.values()):
                instruction = _merge_text(instruction, block)
            continue
        if mode == "labels":
            label = _label_row(padded, row_number)
            if label is not None:
                labels.append(label)
            continue
        item = _item_row(padded, row_number)
        if item is None:
            continue
        if item["item_id"] in seen_item_ids:
            raise SchemaError(f"Duplicate item id {item['item_id']} at row {row_number}.")
        seen_item_ids.add(item["item_id"])
        items.append(item)
    return meta_rows, layout_notes, instruction, labels, items


def _label_row(row: list[str], row_number: int) -> dict[str, Any] | None:
    block = _text_block(row)
    score = row[3]
    label_id = _number_id(row[0])
    if not any(block.values()) and not score:
        return None
    return {
        "label_id": label_id,
        "score": score,
        "source_row": row_number,
        **block,
    }


def _item_row(row: list[str], row_number: int) -> dict[str, Any] | None:
    block = _text_block(row)
    if not block["original_en"] and not block["human_ja"] and not block["llm_ja"]:
        return None
    return {
        "item_id": _number_id(row[0], width=2),
        "reverse_keyed": _reverse_flag(row[3]),
        "factor": row[4],
        "source_row": row_number,
        **block,
    }


def _text_block(row: list[str]) -> dict[str, str]:
    return {
        "original_en": row[1],
        "human_ja": row[2],
        "llm_ja": row[8],
        "human_back_en": row[5],
        "llm_back_en": row[9],
        "human_ja_reconciled": row[6],
        "llm_ja_reconciled": row[10],
        "human_reconcile_note": row[7],
        "llm_reconcile_note": row[11],
    }


def _empty_text() -> dict[str, str]:
    return {key: "" for key in TEXT_KEYS}


def _merge_text(current: dict[str, str], addition: dict[str, str]) -> dict[str, str]:
    merged = {}
    for key in TEXT_KEYS:
        left = current.get(key, "")
        right = addition.get(key, "")
        if left and right:
            merged[key] = f"{left} {right}"
        else:
            merged[key] = left or right
    return merged


def _overview(worksheet) -> list[dict[str, str]]:
    records = []
    started = False
    for row in worksheet.iter_rows(max_col=12, values_only=True):
        cells = [_clean(cell) for cell in row]
        cells += [""] * (12 - len(cells))
        if not started:
            if cells[0] == "順LLM":
                started = True
            continue
        if not cells[4] and not cells[6] and not cells[8]:
            continue
        if cells[5].startswith(_OVERVIEW_SKIP):
            continue
        records.append(
            {
                "forward_llm": cells[0],
                "back_llm": cells[1],
                "reconcile_llm": cells[2],
                "survey_set": cells[3],
                "scale_id": _number_id(cells[4], width=2) if cells[4] else "",
                "team": cells[5],
                "description": cells[6],
                "population": cells[7],
                "name": cells[8],
                "construct": cells[9],
                "n_items": cells[10],
                "response_format": cells[11],
            }
        )
    return records


def _match_registry(
    scale_id: str,
    short_name: str,
    sheet_name: str,
    sheet_meta: dict[str, str],
    registry: list[dict[str, str]],
) -> dict[str, str]:
    if scale_id.isdigit() and scale_id != "00":
        for record in registry:
            if record.get("scale_id") == scale_id:
                return record
        return {}
    haystack = " ".join(
        [sheet_name, short_name, scale_id, *sheet_meta.values()]
    ).lower()
    for record in registry:
        for key in ("name", "description"):
            phrase = record.get(key, "").strip().lower()
            if len(phrase) >= 4 and phrase in haystack:
                return record
    return {}


def _canonical_meta(rows: list[dict[str, str]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for row in rows:
        label = row["label"]
        for prefix, key in META_PREFIXES:
            if label.startswith(prefix) and key not in found:
                found[key] = row["value"]
                break
    return found


def _sheet_identity(sheet_name: str) -> tuple[str, str]:
    matched = _SHEET_NUMBER.match(sheet_name.strip())
    if matched:
        return f"{int(matched.group(1)):02d}", matched.group(2).strip()
    return _slug(sheet_name), sheet_name.strip()


def _status(sheet_name: str, scale_id: str) -> str:
    stripped = sheet_name.strip().lower()
    if stripped.startswith("x") or sheet_name.strip().startswith("ｘ"):
        return "excluded"
    if scale_id == "00" or "サンプル" in sheet_name:
        return "sample"
    if scale_id.isdigit():
        return "study"
    return "excluded"


def _reverse_flag(value: str) -> bool:
    token = value.strip().lower()
    if not token or token in {"0", "false", "no", "n"}:
        return False
    return token in _REVERSE_TRUE or token == "r"


def _number_id(value: str, width: int = 0) -> str:
    token = value.strip()
    if re.fullmatch(r"\d+", token):
        number = str(int(token))
        return number.zfill(width) if width else number
    return token


def _slug(value: str) -> str:
    token = re.sub(r"^[xｘ]\s*", "", value.strip(), flags=re.IGNORECASE)
    token = re.sub(r"\s+", "-", token)
    return token or value.strip()


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = unicodedata.normalize("NFKC", str(value))
    text = _LINEBREAK.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
