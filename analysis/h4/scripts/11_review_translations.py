"""Write a self-contained HTML review of original text, both Japanese translations, and both back-translations.

The page is for reading the workbook. It does not enter the confirmatory estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

from h4.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("original_en", "human_ja", "llm_ja", "human_back_en", "llm_back_en")
PLACEHOLDERS = {"", "na", "n/a", "n.a.", "該当なし", "なし", "none", "-", "―", "not applicable"}


def main() -> None:
    catalog = load_catalog(ROOT / "data" / "raw" / "scales.json")
    payload = {
        "cosine_scope": "item",
        "scales": [_scale(scale) for scale in catalog["scales"] if scale.get("status") == "study"],
    }
    html = (ROOT / "review" / "template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    out = ROOT / "review" / "translation_check.html"
    out.write_text(html.replace("/*__DATA__*/", data), encoding="utf-8")
    gaps = sum(1 for scale in payload["scales"] for row in scale["rows"] if row["flags"])
    print(f"Wrote {out} ({gaps} flagged rows)")


def _scale(scale: dict) -> dict:
    rows = [_row("instruction", "Instruction", scale.get("instruction") or {})]
    rows.extend(_row("label", f"Label {label.get('label_id', '')}", label) for label in scale.get("labels") or [])
    rows.extend(_row("item", f"Item {item.get('item_id', '')}", item) for item in scale.get("items") or [])
    return {
        "scale_id": scale.get("scale_id", ""),
        "short_name": scale.get("short_name", ""),
        "sheet": scale.get("sheet", ""),
        "layout_notes": list(scale.get("layout_notes") or []),
        "meta": [{"label": row.get("label", ""), "value": row.get("value", "")} for row in scale.get("sheet_meta_as_written") or []],
        "rows": rows,
    }


def _row(section: str, title: str, record: dict) -> dict:
    texts = {field: str(record.get(field) or "") for field in FIELDS}
    blank = {field: _blank(texts[field]) for field in FIELDS}
    flags = []
    if blank["original_en"]:
        flags.append("Original is empty or a placeholder such as NA.")
    if any(blank.values()) and not all(blank.values()):
        missing = [field for field, is_blank in blank.items() if is_blank]
        flags.append("Text is present on one side and missing on: " + ", ".join(missing) + ".")
    if blank["original_en"] and not (blank["human_ja"] and blank["llm_ja"]):
        flags.append("A Japanese instruction or label was written where the original has no text.")
    return {
        "section": section,
        "title": title,
        "texts": texts,
        "blank": blank,
        "flags": flags,
        "notes": {
            "human": str(record.get("human_reconcile_note") or ""),
            "llm": str(record.get("llm_reconcile_note") or ""),
        },
    }


def _blank(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDERS


if __name__ == "__main__":
    main()
