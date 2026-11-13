"""Exclusion rules copied from the citizen and expert Quarto notebooks."""

from __future__ import annotations

import pandas as pd

from h4.survey.flags import add_citizen_flags, add_expert_flags


def _citizen_row(**updates: str) -> dict[str, str]:
    row = {
        "set_id": "A",
        "mcs_gender": "female",
        "dqs01_check": "0",
        "dqs02_check": "0",
        "dqs03_check": "0",
        "ai-trap_check": "0",
        "A01_item01_human": "1",
        "A01_item02_human": "2",
        "A01_item01_ai": "1",
        "A01_item02_ai": "2",
        "A01_timer_human": "1000",
        "A01_timer_ai": "1000",
        "C10_item01_human": "",
        "C10_item01_ai": "",
        "C10_timer_human": "",
        "C10_timer_ai": "",
    }
    row.update(updates)
    return row


def test_citizen_attention_treats_a_blank_as_a_failure() -> None:
    frame = pd.DataFrame([_citizen_row(**{"dqs02_check": ""})])
    flagged = add_citizen_flags(frame)
    assert int(flagged.loc[0, "analysis_target"]) == 1
    assert int(flagged.loc[0, "excl1"]) == 1


def test_citizen_mcs_items_are_required_only_for_men_on_sets_c_and_d() -> None:
    woman = _citizen_row(
        set_id="C",
        mcs_gender="female",
        C06_item01_human="2",
        C06_item01_ai="2",
        C10_item01_human="",
        C10_item01_ai="",
    )
    man = _citizen_row(
        set_id="C",
        mcs_gender="male",
        C06_item01_human="2",
        C06_item01_ai="2",
        C10_item01_human="",
        C10_item01_ai="",
    )
    answered = _citizen_row(
        set_id="C",
        mcs_gender="male",
        C06_item01_human="2",
        C06_item01_ai="2",
        C10_item01_human="3",
        C10_item01_ai="3",
    )
    flagged = add_citizen_flags(pd.DataFrame([woman, man, answered]))
    assert list(flagged["analysis_target"]) == [1, 0, 1]


def test_straightlining_uses_the_quarto_run_length() -> None:
    short = _citizen_row()
    long_run = _citizen_row()
    for index in range(1, 7):
        column = f"A01_item{index:02d}_human"
        short[column] = "1" if index <= 4 else str(index)
        long_run[column] = "1" if index <= 5 else "2"
        short[f"A01_item{index:02d}_ai"] = str(index)
        long_run[f"A01_item{index:02d}_ai"] = str(index)
    flagged = add_citizen_flags(pd.DataFrame([short, long_run]))
    assert list(flagged["excl3"]) == [0, 1]
    assert int(flagged.loc[1, "excl01human"]) == 1
    assert flagged.loc[1, "excl3where"] == "01_human"


def test_speeding_flags_only_complete_respondents_below_the_scale_cutoff() -> None:
    rows = []
    for _index in range(15):
        rows.append(_citizen_row(A01_timer_human="5000", A01_timer_ai="5000"))
    rows.append(_citizen_row(A01_timer_human="50", A01_timer_ai="50"))
    rows.append(_citizen_row(A01_item01_human="", A01_timer_human="1", A01_timer_ai="1"))
    flagged = add_citizen_flags(pd.DataFrame(rows))
    assert list(flagged["excl2"].iloc[:15]) == [0] * 15
    assert int(flagged["excl2"].iloc[15]) == 1
    assert int(flagged["excl2"].iloc[16]) == 0


def test_expert_blank_attention_is_not_a_failure_and_eligibility_is_required() -> None:
    base = {
        "set_id": "A",
        "eligibility_native_japanese": "1",
        "eligibility_scale_paper": "1",
        "eligibility_english_citation": "1",
        "eligibility_masters_or_higher": "1",
        "dqs01_check": "",
        "dqs02_check": "0",
        "dqs03_check": "",
        "ai-trap_check": "0",
        "A01_semantic_ai": "4",
        "A01_naturalness_ai": "4",
        "A01_cultural_validity_ai": "4",
        "A01_choice_fit_ai": "4",
        "A01_semantic_human": "",
    }
    failed_check = dict(base)
    failed_check["dqs02_check"] = "1"
    ineligible = dict(base)
    ineligible["eligibility_masters_or_higher"] = "0"
    flagged = add_expert_flags(pd.DataFrame([base, failed_check, ineligible]))
    assert list(flagged["analysis_target"]) == [1, 1, 0]
    assert list(flagged["excl1"]) == [0, 1, 0]
