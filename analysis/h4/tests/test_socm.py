from __future__ import annotations

import numpy as np
import pytest

from h4.exploratory.distributional_distance import pairwise_euclidean, sinkhorn, wasserstein
import pandas as pd

from h4.exploratory.pooling.collapse_metric import socm
from h4.exploratory.pooling.paired_distances import paired_token_contrast, paired_token_summary


def test_socm_is_zero_for_identical_clouds():
    tokens = np.array([[1.0, 0.0], [1.0, 0.2], [0.8, 0.0]])
    result = socm(tokens, tokens.copy())
    assert result["d_mu"] == pytest.approx(0.0, abs=1e-8)
    assert result["d_sigma"] == pytest.approx(0.0, abs=1e-8)
    assert result["socm"] == pytest.approx(0.0, abs=1e-8)
    assert result["directional_test"] is False


def test_same_mean_and_different_spread_raises_socm():
    collapsed = np.array([[1.0, 0.0], [1.0, 0.0]])
    spread = np.array([[1.0, 1.0], [1.0, -1.0]])
    same_mean = socm(collapsed, spread)
    far = np.array([[0.0, 1.0], [0.0, 1.0]])
    different_mean = socm(collapsed, far)
    assert same_mean["d_mu"] == pytest.approx(0.0, abs=1e-8)
    assert same_mean["socm"] == pytest.approx(0.25, abs=1e-6)
    assert different_mean["socm"] == pytest.approx(0.0, abs=1e-8)
    assert same_mean["socm"] > different_mean["socm"]


def test_second_order_gap_matters_less_when_means_differ():
    collapsed = np.array([[1.0, 0.0], [1.0, 0.0]])
    spread = np.array([[1.0, 1.0], [1.0, -1.0]])
    far_spread = np.array([[0.0, 2.0], [0.0, 0.0]])
    near = socm(collapsed, spread)
    far = socm(collapsed, far_spread)
    assert near["d_sigma"] == pytest.approx(far["d_sigma"], abs=1e-6)
    assert far["socm"] < near["socm"]


def test_zero_mean_is_undefined():
    tokens = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="near zero"):
        socm(tokens, tokens)


def test_paired_token_contrast_keeps_hara_distances_only():
    pairs = pd.DataFrame(
        [
            {"scale_id": "01", "item_id": "01", "comparison": "back_translation", "embedding_model": "m", "short_name": "mpnet", "translation": "human", "d_mu": 0.20, "d_sigma": 0.05, "cosine": 0.9, "socm": 0.04},
            {"scale_id": "01", "item_id": "01", "comparison": "back_translation", "embedding_model": "m", "short_name": "mpnet", "translation": "llm", "d_mu": 0.10, "d_sigma": 0.02, "cosine": 0.95, "socm": 0.018},
        ]
    )
    contrast = paired_token_contrast(pairs)
    summary = paired_token_summary(contrast)
    assert "cosine" not in contrast.columns
    assert "socm" not in contrast.columns
    assert contrast.loc[0, "delta_d_mu"] == pytest.approx(0.10)
    assert contrast.loc[0, "delta_d_sigma"] == pytest.approx(0.03)
    assert contrast.loc[0, "directional_test"] == False
    assert summary.loc[0, "mean_delta_d_mu"] == pytest.approx(0.10)
    assert summary.loc[0, "mean_d_sigma_llm"] == pytest.approx(0.02)


def test_wasserstein_of_a_cloud_with_itself_is_zero():
    cloud = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = wasserstein(pairwise_euclidean(cloud, cloud))
    assert result["solver"] == "emd"
    assert result["distance"] == pytest.approx(0.0, abs=1e-8)
    assert sinkhorn(np.zeros((3, 3))) == pytest.approx(0.0, abs=1e-8)
