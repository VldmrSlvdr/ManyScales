from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from h4.catalog import (
    analysis_items_from_catalog,
    extract_workbook,
    items_from_catalog,
    load_catalog,
    write_catalog,
    write_preanalysis_tables,
)
from h4.io import read_item_table


def _workbook(path: Path) -> None:
    book = Workbook()
    overview = book.active
    overview.title = "対象尺度 の詳細"
    overview.append(
        ["順LLM", "逆LLM", "調整LLM", "セット", "ID", "メンバー", "尺度", "対象母集団", "尺度名", "概念", "項目数", "回答形式"]
    )
    overview.append(
        ["gpt-a", "gpt-b", "gpt-c", "A", 1, "team-1", "Demo scale", "adults", "DEMO", "mood", 2, "likert"]
    )
    overview.append(["", "", "", "", "", "対象とする尺度について、後から追加した行", "", "", "", "", "", ""])

    study = book.create_sheet("1 DEMO")
    study.append(["メンバー", "team-1"])
    study.append(["尺度名(original)", "DEMO"])
    study.append(["概念", "mood"])
    study.append(["", "A layout note that sits above the instruction."])
    study.append(["教示", "", "", "", "", "", "", "", "", "", "", ""])
    study.append(["", "Read each statement.", "各文を読んでください。", "", "", "Read each statement, human.", "", "", "各文を読んでください。AI", "Read each statement, AI."])
    study.append(["選択肢とラベル"])
    study.append(["1", "Not at all", "まったくない", "1", "", "Not at all, human.", "", "", "まったくない AI", "Not at all, AI."])
    study.append(["2", "", "やや", "2"])
    study.append(["3", "", "", "3"])
    study.append(["6"])
    study.append(["項目"])
    study.append(["1", "I feel tense.", "緊張している。", "r", "body", "I feel tense, human.", "", "", "緊張。AI", "I feel tense, AI."])
    study.append(["2", "I feel calm.", "落ち着いている。", "", "mood", "I feel calm, human.", "", "", "落ち着き。AI", "I feel calm, AI."])
    study.append(["3"])

    sample = book.create_sheet("0 サンプル")
    sample.append(["教示"])
    sample.append(["", "Sample instruction.", "見本の教示。"])
    sample.append(["項目"])
    sample.append(["1", "Sample item.", "見本項目。"])

    excluded = book.create_sheet("x OESQ")
    excluded.append(["尺度名(original)", "Open and Engaged"])
    excluded.append(["教示"])
    excluded.append(["", "Excluded instruction.", "除外する教示。"])

    book.save(path)
    book.close()


def test_catalog_keeps_texts_and_limits_h4_to_study_items(tmp_path):
    source = tmp_path / "scales.xlsx"
    _workbook(source)
    catalog = extract_workbook(source)
    by_id = {scale["scale_id"]: scale for scale in catalog["scales"]}

    study = by_id["01"]
    assert study["status"] == "study"
    assert study["registry_matched"] is True
    assert study["registry"]["survey_set"] == "A"
    assert study["sheet_meta"]["name_en"] == "DEMO"
    assert study["layout_notes"] == ["A layout note that sits above the instruction."]
    assert study["instruction"]["original_en"] == "Read each statement."
    assert study["instruction"]["human_ja"] == "各文を読んでください。"
    assert study["instruction"]["llm_ja"] == "各文を読んでください。AI"
    assert [label["label_id"] for label in study["labels"]] == ["1", "2", "3"]
    assert study["labels"][1]["human_ja"] == "やや"
    assert study["labels"][2]["score"] == "3"
    assert study["items"][0]["item_id"] == "01"
    assert study["items"][0]["reverse_keyed"] is True
    assert study["items"][0]["factor"] == "body"
    assert study["items"][1]["llm_back_en"] == "I feel calm, AI."

    assert by_id["00"]["status"] == "sample"
    assert by_id["OESQ"]["status"] == "excluded"
    assert by_id["OESQ"]["registry_matched"] is False

    items = items_from_catalog(catalog)
    assert list(items["scale_id"]) == ["01", "01"]
    assert items.loc[0, "llm_translation_model"] == "gpt-a"
    assert items.loc[0, "survey_set"] == "A"

    destination = tmp_path / "scales.json"
    write_catalog(catalog, destination)
    loaded = analysis_items_from_catalog(load_catalog(destination))
    assert loaded["reverse_keyed"].tolist() == [True, False]

    write_preanalysis_tables(catalog, tmp_path / "processed")
    meta = pd.read_csv(tmp_path / "processed" / "scale_meta.csv")
    instructions = pd.read_csv(tmp_path / "processed" / "instructions.csv")
    labels = pd.read_csv(tmp_path / "processed" / "labels.csv")
    assert set(meta["status"]) == {"study", "sample", "excluded"}
    assert instructions.loc[instructions["scale_id"] == "01", "human_ja"].item() == "各文を読んでください。"
    assert len(labels) == 3

    frame = read_item_table(destination, preanalysis_dir=tmp_path / "from_loader")
    assert len(frame) == 2
    assert (tmp_path / "from_loader" / "labels.csv").exists()

    parquet = tmp_path / "items_clean.parquet"
    frame.to_parquet(parquet, index=False)
    reloaded = read_item_table(parquet)
    assert len(reloaded) == 2
    assert reloaded["original_en"].tolist() == frame["original_en"].tolist()
    assert reloaded["reverse_keyed"].tolist() == [True, False]
