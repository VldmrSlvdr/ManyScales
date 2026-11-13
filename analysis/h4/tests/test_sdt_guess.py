"""The LLM-column guess is one forced choice, so the four cells come in pairs."""

import numpy as np

from h4.survey.sdt import dprime_2afc, score_guess


def test_naming_column_2_is_a_hit_and_a_correct_rejection():
    scored = score_guess("2")
    assert scored["hit"] == 1.0
    assert scored["correct_rejection"] == 1.0
    assert scored["miss"] == 0.0
    assert scored["false_alarm"] == 0.0


def test_naming_column_1_is_a_miss_and_a_false_alarm():
    scored = score_guess("1")
    assert scored["hit"] == 0.0
    assert scored["miss"] == 1.0
    assert scored["false_alarm"] == 1.0
    assert scored["correct_rejection"] == 0.0


def test_two_alternative_dprime_is_zero_at_chance():
    assert abs(dprime_2afc(0.5)) < 1e-12
    assert dprime_2afc(0.6) > 0
    assert np.isfinite(dprime_2afc(0.0))
