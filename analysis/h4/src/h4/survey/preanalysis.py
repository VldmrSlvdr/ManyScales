"""Check the Quarto survey cut, then write the fixed long analysis tables.

Run ``python -m h4.survey.preanalysis``. The confirmatory pipeline does not
call this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from h4.io import write_json
from h4.survey.flags import add_citizen_flags, add_expert_flags
from h4.survey.qualtrics import read_qualtrics_tsv
from h4.survey.replicate import _same_sets, _same_table
from h4.survey.reshape import (
    assert_checks,
    catalog_items,
    check_tables,
    load_processed,
    reshape_survey,
)

_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    summary = run(_ROOT)
    print(json.dumps({"checks_passed": summary["checks_passed"], "outputs": summary["outputs"]}, indent=2))


def run(root: Path) -> dict:
    verification = verify_saved_cut(root)
    _require_match(verification)
    survey = root / "data" / "survey"
    citizen_frames = load_processed(survey / "3_processed", "ManyScales_市民_processed_")
    expert_frames = load_processed(survey / "3_processed", "ManyScales_専門家_processed_")
    citizen = reshape_survey(citizen_frames, "citizen")
    expert = reshape_survey(expert_frames, "expert")
    items = catalog_items(root / "data/raw/scales.json", root / "data/processed/scale_meta.csv")
    labels = pd.read_csv(root / "data/processed/labels.csv", dtype=str)
    names = items.drop_duplicates("scale_id").set_index("scale_id")["scale_name"]
    for table in (*citizen.values(), *expert.values()):
        table["scale_name"] = table["scale_id"].map(names)
    checks = check_tables(
        citizen_frames=citizen_frames,
        expert_frames=expert_frames,
        citizen=citizen,
        expert=expert,
        items=items,
        labels=labels,
    )
    assert_checks(checks)
    destination = survey / "4_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "citizen_item_response": _write(citizen["item_response"], destination / "citizen_item_response.parquet"),
        "citizen_scale_judgment": _write(citizen["scale_judgment"], destination / "citizen_scale_judgment.parquet"),
        "expert_scale_rating": _write(expert["scale_rating"], destination / "expert_scale_rating.parquet"),
        "expert_item_problem": _write(expert["item_problem"], destination / "expert_item_problem.parquet"),
        "expert_scale_choice": _write(expert["scale_choice"], destination / "expert_scale_choice.parquet"),
        "catalog_items": _write(items, destination / "catalog_items.parquet"),
    }
    checks.to_csv(destination / "reshape_checks.csv", index=False)
    summary = {
        "verification": _shrink(verification),
        "checks_passed": bool(checks["passed"].all()),
        "checks": checks.to_dict(orient="records"),
        "outputs": outputs,
    }
    write_json(summary, destination / "preanalysis_checks.json")
    return summary


def verify_saved_cut(root: Path) -> dict:
    """Recompute the exclusion flags and compare them with the Quarto files."""

    analysis = root.parent / "ratings/primary"
    survey = root / "data" / "survey"
    citizen_raw, _header = read_qualtrics_tsv(
        analysis / "data/0_raw_data/2026_ManyScales_一般_2026年7月29日_06.39.tsv"
    )
    expert_raw, _header = read_qualtrics_tsv(
        analysis / "data/0_raw_data/2026_ManyScales_専門_2026年9月16日_08.46.tsv"
    )
    citizen_flags = add_citizen_flags(citizen_raw)
    expert_flags = add_expert_flags(expert_raw)
    return {
        "citizen_flags": _same_table(
            citizen_flags,
            analysis / "data/1_data_flags/ManyScales_一般_analysis_flags.tsv",
            sep="\t",
        ),
        "expert_flags": _same_table(
            expert_flags,
            analysis / "data/1_data_flags/ManyScales_専門_analysis_flags.tsv",
            sep="\t",
        ),
        "citizen_processed": _same_sets(
            survey / "3_processed",
            analysis / "data/3_data_processed",
            "ManyScales_市民_processed_",
        ),
        "expert_processed": _same_sets(
            survey / "3_processed",
            analysis / "data/3_data_processed",
            "ManyScales_専門家_processed_",
        ),
        "n_citizen_raw": int(len(citizen_raw)),
        "n_expert_raw": int(len(expert_raw)),
        "n_citizen_flags": int(len(citizen_flags)),
        "n_expert_flags": int(len(expert_flags)),
    }


def _require_match(verification: dict) -> None:
    failed = [
        name
        for name in ("citizen_flags", "expert_flags", "citizen_processed", "expert_processed")
        if not verification[name].get("match", verification[name].get("all_match"))
    ]
    if failed:
        raise RuntimeError("The saved Quarto tables differ from the recomputed cut: " + ", ".join(failed))


def _write(frame: pd.DataFrame, path: Path) -> str:
    frame.to_parquet(path, index=False)
    return str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path)


def _shrink(verification: dict) -> dict:
    shrunk = {}
    for key, value in verification.items():
        if isinstance(value, dict):
            shrunk[key] = {
                "match": value.get("match", value.get("all_match")),
                "mismatched_cells": value.get("mismatched_cells", 0),
            }
            if "all_match" in value:
                shrunk[key]["sets"] = {
                    name: item.get("match") for name, item in value.items() if isinstance(item, dict)
                }
        else:
            shrunk[key] = value
    return shrunk


if __name__ == "__main__":
    main()
