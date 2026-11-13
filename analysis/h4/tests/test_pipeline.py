from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from h4.config import STAGE_ORDER
from h4.exploratory.runner import main as exploratory_main
from h4.pipeline import main


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_config(tmp_path, items):
    """Keep the end-to-end demo independent of downloaded data files."""
    source = tmp_path / "synthetic_items.csv"
    items.to_csv(source, index=False)
    config = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())
    config["paths"] = {
        name: str((ROOT / value).resolve()) for name, value in config["paths"].items()
    }
    config["paths"]["items"] = str(source)
    config["paths"]["diagnostic_pairs"] = str(tmp_path / "optional_diagnostics.csv")
    config["paths"]["processed"] = str(tmp_path / "items.parquet")
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_demo_pipeline_writes_confirmatory_and_skips_exploratory(tmp_path, demo_config):
    results = tmp_path / "results"
    processed = tmp_path / "items.parquet"
    main(
        [
            "--config",
            str(demo_config),
            "--results",
            str(results),
            "--processed",
            str(processed),
            "--through",
            "figures",
        ]
    )
    assert "exploratory" not in STAGE_ORDER
    assert not (results / "exploratory").exists()
    estimate = pd.read_csv(results / "tables" / "confirmatory_estimate.csv")
    assert estimate.loc[0, "analysis_tier"] == "confirmatory"
    assert estimate.loc[0, "metric"] == "cosine"
    assert estimate.loc[0, "comparison"] == "back_translation"
    assert estimate.loc[0, "design_matrix_columns"] == 1
    assert "socm" not in estimate.columns
    curve = pd.read_csv(results / "tables" / "specification_curve.csv")
    assert curve.loc[curve["analysis_tier"] == "confirmatory", "metric"].eq("cosine").all()
    validation = pd.read_csv(results / "validation" / "metric_validation.csv")
    assert "translation" not in validation.columns
    assert (results / "figures" / "specification_curve.png").exists()
    assert (results / "figures" / "scale_estimates.png").exists()


def test_exploratory_outputs_stay_outside_the_confirmatory_estimate(tmp_path, demo_config):
    results = tmp_path / "results"
    processed = tmp_path / "items.parquet"
    main(
        [
            "--config",
            str(demo_config),
            "--results",
            str(results),
            "--processed",
            str(processed),
            "--through",
            "robustness",
        ]
    )
    before = (results / "tables" / "confirmatory_estimate.csv").read_text(encoding="utf-8")
    exploratory_main(
        [
            "--config",
            str(demo_config),
            "--results",
            str(results),
            "--part",
            "pooling",
        ]
    )
    exploratory_main(
        [
            "--config",
            str(demo_config),
            "--results",
            str(results),
            "--part",
            "geometry",
        ]
    )
    after = (results / "tables" / "confirmatory_estimate.csv").read_text(encoding="utf-8")
    assert before == after
    contrast = pd.read_csv(results / "exploratory" / "pooling" / "socm_contrast.csv")
    assert set(contrast["analysis_tier"]) == {"exploratory"}
    assert not contrast["directional_test"].any()
    distances = pd.read_csv(results / "exploratory" / "geometry" / "wasserstein.csv")
    assert set(distances["analysis_tier"]) == {"exploratory"}
    assert (results / "exploratory" / "pooling" / "alternative_pooling.csv").exists()
    assert (results / "token_embeddings" / "hash" / "index.parquet").exists()
