from __future__ import annotations

import numpy as np

from h4.similarity import score_vectors
from h4.similarity.angular import angular_distance
from h4.similarity.cosine import cosine_similarity
from h4.similarity.distance import euclidean_normalized


def test_cosine_identities():
    same = cosine_similarity(np.array([[1.0, 0.0]]), np.array([[4.0, 0.0]]))
    orthogonal = cosine_similarity(np.array([[1.0, 0.0]]), np.array([[0.0, 2.0]]))
    opposite = cosine_similarity(np.array([[1.0, 0.0]]), np.array([[-3.0, 0.0]]))
    assert same == pytest_approx(1.0)
    assert orthogonal == pytest_approx(0.0)
    assert opposite == pytest_approx(-1.0)


def test_euclidean_matches_cosine_identity():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(20, 8))
    right = rng.normal(size=(20, 8))
    cosine = cosine_similarity(left, right)
    euclidean = euclidean_normalized(left, right)
    assert np.allclose(euclidean**2, 2.0 * (1.0 - cosine))


def test_angular_is_a_decreasing_function_of_cosine():
    pairs = [
        (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), 0.0),
        (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 0.5),
        (np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]]), 1.0),
    ]
    distances = [float(angular_distance(left, right)[0]) for left, right, _expected in pairs]
    assert distances == pytest_list([0.0, 0.5, 1.0])
    assert distances[0] < distances[1] < distances[2]


def test_distance_orientation_keeps_higher_similarity_closer():
    left = np.array([[1.0, 0.0]])
    near = np.array([[1.0, 0.1]])
    far = np.array([[-1.0, 0.0]])
    near_score, near_raw = score_vectors("euclidean_normalized", left, near)
    far_score, far_raw = score_vectors("euclidean_normalized", left, far)
    assert near_raw[0] < far_raw[0]
    assert near_score[0] > far_score[0]


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)


def pytest_list(values):
    import pytest

    return [pytest.approx(value) for value in values]
