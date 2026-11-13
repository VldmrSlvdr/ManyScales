from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h4.config import FROZEN_PRIMARY_MODEL, STAGE_ORDER, load_config
from h4.config import _reject_forbidden
from h4.errors import ConfigError
from h4.statistics.beta import fit_beta_contrast
from h4.statistics.bootstrap import hierarchical_bootstrap
from h4.statistics.mixed import fit_hierarchical
from h4.statistics.specification_curve import specification_curve


def test_preregistered_config_is_frozen():
    config = load_config("configs/analysis.yaml")
    assert config.preregistered
    assert config.primary.name == FROZEN_PRIMARY_MODEL
    assert config.primary_metric.name == "cosine"
    assert config.comparison_fields["back_translation"]["human"] == "human_back_en"
    assert "exploratory" not in STAGE_ORDER


def test_result_driven_model_selection_is_rejected():
    with pytest.raises(ConfigError, match="H4 result"):
        _reject_forbidden({"select_by_h4": True})


def test_hierarchical_model_recovers_a_positive_mean():
    rng = np.random.default_rng(0)
    rows = []
    for scale in range(8):
        shift = rng.normal(0, 0.05)
        for _item in range(6):
            rows.append(
                {
                    "scale_id": f"S{scale}",
                    "delta": 0.2 + shift + rng.normal(0, 0.05),
                    "translator_id": "should-not-enter",
                }
            )
    fitted = fit_hierarchical(pd.DataFrame(rows), confidence=0.95)
    assert fitted["design_matrix_columns"] == 1
    assert fitted["formula"] == "delta ~ 1 + (1|scale_id)"
    assert fitted["estimate"] == pytest.approx(0.2, abs=0.05)
    assert fitted["ci_low"] > 0
    assert fitted["prob_positive"] > 0.95
    assert len(fitted["scale_table"]) == 8


def test_beta_contrast_detects_a_higher_human_mean():
    rng = np.random.default_rng(1)
    rows = []
    for scale in range(4):
        for item in range(8):
            rows.append(
                {
                    "scale_id": f"S{scale}",
                    "item_id": f"{item:02d}",
                    "sim_human": float(np.clip(0.82 + rng.normal(0, 0.02), 0.05, 0.95)),
                    "sim_llm": float(np.clip(0.62 + rng.normal(0, 0.02), 0.05, 0.95)),
                }
            )
    fitted = fit_beta_contrast(pd.DataFrame(rows), iterations=40, seed=3)
    assert fitted["mean_difference"] > 0.15
    assert fitted["ci_low"] > 0
    assert fitted["p_two_sided_cluster"] < 0.01
    assert fitted["n_items"] == 32
    assert fitted["family"] == "beta"


def test_bootstrap_is_reproducible_and_respects_the_seed():
    frame = pd.DataFrame(
        {
            "scale_id": ["A", "A", "B", "B", "C", "C"],
            "delta": [0.1, 0.2, 0.3, 0.0, 0.4, 0.5],
        }
    )
    first = hierarchical_bootstrap(frame, iterations=150, seed=11)
    second = hierarchical_bootstrap(frame, iterations=150, seed=11)
    third = hierarchical_bootstrap(frame, iterations=150, seed=12)
    assert np.allclose(first["draws"], second["draws"])
    assert not np.allclose(first["draws"], third["draws"])
    assert first["ci_low"] < first["observed_mean"] < first["ci_high"] or first["ci_low"] <= first["ci_high"]


def test_specification_curve_does_not_select_a_model():
    rows = []
    for model, short, delta in (("model-a", "a", 0.2), ("model-b", "b", -0.1)):
        for scale in ("S1", "S2"):
            for item, nudge in (("01", 0.0), ("02", 0.02)):
                rows.append(
                    {
                        "scale_id": scale,
                        "item_id": item,
                        "embedding_model": model,
                        "short_name": short,
                        "metric": "cosine",
                        "comparison": "back_translation",
                        "analysis_tier": "confirmatory" if model == "model-a" else "robustness",
                        "independent_geometry": True,
                        "delta": delta + nudge,
                        "sim_human": 0.5 + delta + nudge,
                        "sim_llm": 0.5,
                        "raw_human": 0.5 + delta + nudge,
                        "raw_llm": 0.5,
                    }
                )
    table, summary = specification_curve(pd.DataFrame(rows), confidence=0.95, engine="statsmodels_mixedlm", seed=1)
    assert summary["selected_model"] is None
    assert summary["cross_model_mean_of_raw_similarities"] is None
    assert set(table["embedding_model"]) == {"model-a", "model-b"}
