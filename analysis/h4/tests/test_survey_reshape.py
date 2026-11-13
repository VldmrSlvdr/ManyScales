"""The fixed survey reshape keeps every wide cell and the legal response codes."""

import pandas as pd

from h4.survey.reshape import allowed_scores, check_tables, reshape_survey


def _citizen_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": ["30", "41"],
            "gender": ["1", "2"],
            "trans_cond": ["human", "ai"],
            "A01_item01_human": ["1", "5"],
            "A01_item01_ai": ["2", pd.NA],
            "A01_item02_human": ["3", "4"],
            "A01_item02_ai": ["3", "1"],
            "A01_wording_human": ["4", "2"],
            "A01_wording_ai": ["5", "2"],
            "A01_preference": ["1", "2"],
            "A01_llm_guess": ["2", "1"],
        }
    )


def _expert_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": ["35"],
            "gender": ["1"],
            "highest_degree": ["3"],
            "trans_cond": ["ai"],
            "A01_semantic_human": ["5"],
            "A01_semantic_ai": ["3"],
            "A01_naturalness_human": ["4"],
            "A01_naturalness_ai": ["4"],
            "A01_cultural_validity_human": ["5"],
            "A01_cultural_validity_ai": ["2"],
            "A01_choice_fit_human": ["4"],
            "A01_choice_fit_ai": ["3"],
            "A01_problem_items_human_1_1": ["1"],
            "A01_problem_items_ai_1_2": [pd.NA],
            "A01_preference": ["1"],
            "A01_llm_guess": ["2"],
        }
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scale_id": ["01", "01"],
            "item_id": ["01", "02"],
            "survey_set": ["A", "A"],
            "reverse_keyed": [False, True],
            "scale_name": ["IDAS-II", "IDAS-II"],
        }
    )


def _labels() -> pd.DataFrame:
    rows = [{"scale_id": "01", "label_id": str(score), "score": str(score)} for score in range(1, 6)]
    rows.append({"scale_id": "18", "label_id": "1", "score": pd.NA})
    rows.append({"scale_id": "18", "label_id": "4", "score": pd.NA})
    rows.append({"scale_id": "18", "label_id": "7", "score": pd.NA})
    return pd.DataFrame(rows)


def test_anchor_labels_fill_the_integer_range():
    allowed = allowed_scores(_labels())
    assert allowed["01"] == {"1", "2", "3", "4", "5"}
    assert allowed["18"] == {"1", "2", "3", "4", "5", "6", "7"}


def test_reshape_roundtrip_and_codes():
    citizen = reshape_survey({"A": _citizen_frame()}, "citizen")
    expert = reshape_survey({"A": _expert_frame()}, "expert")
    assert set(citizen["item_response"]["version"]) == {"human", "llm"}
    assert int(citizen["item_response"]["raw_value"].isna().sum()) == 1
    checks = check_tables(
        citizen_frames={"A": _citizen_frame()},
        expert_frames={"A": _expert_frame()},
        citizen=citizen,
        expert=expert,
        items=_items(),
        labels=_labels(),
    )
    failed = checks.loc[~checks["passed"], "check"].tolist()
    assert failed == []


def test_cultural_ratings_preserve_exported_quality_direction_and_problem_flags():
    frame = _expert_frame()
    frame["A01_problem_items_human_1_3"] = "1"
    frame["A01_problem_items_ai_1_3"] = "0"
    expert = reshape_survey({"A": frame}, "expert")
    cultural = expert["scale_rating"].loc[lambda f: f.dimension.eq("cultural_validity")]
    scores = cultural.set_index("version")["raw_value"].astype(int)
    # The input is a Qualtrics export: 5 already means fewer unfamiliar expressions.
    assert scores.to_dict() == {"human": 5, "llm": 2}
    assert scores["human"] - scores["llm"] == 3
    problems = expert["item_problem"].loc[lambda f: f.dimension.eq("cultural")]
    assert problems.loc[problems.version.eq("human"), "raw_value"].tolist() == ["1"]
    assert problems.loc[problems.version.eq("llm"), "raw_value"].tolist() == ["0"]
