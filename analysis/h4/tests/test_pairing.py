from __future__ import annotations

import pandas as pd
import pytest

from h4.errors import SchemaError
from h4.preprocess import exclude_flagged_scales, preprocess_items
from h4.statistics.paired import wide_delta
from h4.validation import build_validation_pairs, load_diagnostic_pairs
from h4.validation.lexical_overlap import jaccard
from h4.validation.negation import negate


def test_schema_rejects_duplicate_items(items):
    broken = pd.concat([items, items.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        preprocess_items(broken)


def test_schema_rejects_empty_text(items):
    broken = items.copy()
    broken.loc[0, "original_en"] = "   "
    with pytest.raises(SchemaError, match="original_en"):
        preprocess_items(broken)


def test_provenance_is_preserved_and_text_is_sorted(items):
    flipped = items.iloc[::-1].copy()
    cleaned = preprocess_items(flipped)
    assert cleaned["human_translator_id"].eq("T1").all()
    assert cleaned["scale_id"].is_monotonic_increasing or cleaned["item_id"].is_monotonic_increasing
    assert list(cleaned.sort_values(["scale_id", "item_id"])["item_id"]) == list(cleaned["item_id"])


def test_preprocess_records_the_tivs_deletion(items):
    extra = items.iloc[[0, 1]].copy()
    extra["scale_id"] = "11"
    extra["scale_name"] = "TIVS"
    extra["item_id"] = ["01", "02"]
    combined = pd.concat([items, extra], ignore_index=True)
    primary, log = exclude_flagged_scales(preprocess_items(combined))
    assert set(primary["scale_id"]) == set(items["scale_id"])
    assert len(primary) == len(items)
    row = log.iloc[0]
    assert row["step"] == "preprocess"
    assert row["scale_id"] == "11"
    assert row["scale_name"] == "TIVS"
    assert row["n_items_removed"] == 2
    assert row["n_items_before"] == len(items) + 2
    assert row["n_items_after"] == len(items)
    assert row["n_scales_after"] == row["n_scales_before"] - 1
    assert row["primary_role"] == "excluded"
    assert "misuse" in row["reason"]


def test_delta_is_human_minus_llm():
    long_df = pd.DataFrame(
        [
            _row("human", 0.9),
            _row("llm", 0.4),
        ]
    )
    wide, dropped = wide_delta(long_df)
    assert dropped == 0
    assert wide.loc[0, "delta"] == pytest.approx(0.5)
    assert wide.loc[0, "sim_human"] - wide.loc[0, "sim_llm"] == pytest.approx(wide.loc[0, "delta"])


def test_validation_pairs_do_not_read_translation_text(items):
    leaked = items.copy()
    leaked["human_back_en"] = "UNIQUE_HUMAN_LEAK"
    leaked["llm_back_en"] = "UNIQUE_LLM_LEAK"
    leaked["human_ja"] = "UNIQUE_JA_HUMAN"
    leaked["llm_ja"] = "UNIQUE_JA_LLM"
    pairs = build_validation_pairs(preprocess_items(leaked), seed=20260928)
    blob = " ".join(pairs["text_a"].astype(str)) + " " + " ".join(pairs["text_b"].astype(str))
    assert "UNIQUE_HUMAN_LEAK" not in blob
    assert "UNIQUE_LLM_LEAK" not in blob
    assert "UNIQUE_JA_HUMAN" not in blob
    assert "translation" not in pairs.columns


def test_within_scale_permutation_has_no_fixed_points(items):
    pairs = build_validation_pairs(preprocess_items(items), seed=7)
    within = pairs.loc[pairs["pair_type"] == "within_scale_permutation"]
    assert not within.empty
    assert not (within["item_id"] == within["other_item_id"]).any()


def test_validation_pairs_are_seeded(items):
    cleaned = preprocess_items(items)
    first = build_validation_pairs(cleaned, seed=3)
    second = build_validation_pairs(cleaned, seed=3)
    third = build_validation_pairs(cleaned, seed=4)
    assert first["text_b"].tolist() == second["text_b"].tolist()
    assert first["text_b"].tolist() != third["text_b"].tolist()


def test_diagnostic_file_rejects_translation_columns(tmp_path):
    path = tmp_path / "diagnostic.csv"
    path.write_text("text_a,text_b,pair_type,human_back_en\nA,B,semantic_equivalence,leak\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="translation"):
        load_diagnostic_pairs(path)


def test_negation_and_jaccard():
    negated = negate("I feel happy.")
    assert negated == "It is not the case that I feel happy."
    assert jaccard("I feel happy.", "I feel happy.") == 1.0
    assert jaccard("I feel happy.", negated) < 1.0


def _row(translation: str, similarity: float) -> dict:
    return {
        "scale_id": "S01",
        "item_id": "01",
        "embedding_model": "demo",
        "short_name": "demo",
        "metric": "cosine",
        "comparison": "back_translation",
        "analysis_tier": "confirmatory",
        "independent_geometry": True,
        "translation": translation,
        "similarity": similarity,
        "raw_value": similarity,
    }
