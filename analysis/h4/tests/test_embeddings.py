from __future__ import annotations

import numpy as np
import pytest

from h4.config import ModelSpec
from h4.embedders import build_embedder
from h4.embedders.deterministic import DeterministicEmbedder
from h4.errors import ConfigError


def test_deterministic_embeddings_are_unit_length():
    embedder = DeterministicEmbedder(dimension=32)
    matrix = embedder.encode(["I often feel anxious.", "I trust other people."])
    norms = np.linalg.norm(matrix, axis=1)
    assert matrix.shape == (2, 32)
    assert np.allclose(norms, 1.0)


def test_mean_of_token_vectors_matches_sentence_vector():
    embedder = DeterministicEmbedder(dimension=16)
    text = "I often feel anxious."
    sentence = embedder.encode([text])[0]
    tokens = embedder.encode_tokens([text])[0]
    pooled = tokens.mean(axis=0)
    pooled = pooled / np.linalg.norm(pooled)
    assert np.allclose(sentence, pooled, atol=1e-6)


def test_sentence_transformer_import_does_not_require_the_library():
    import h4.embedders.sentence_transformer as module

    assert hasattr(module, "SentenceTransformerEmbedder")


def test_confirmatory_api_embedder_is_refused():
    spec = ModelSpec(
        name="text-embedding-3-large",
        short_name="openai",
        family="api",
        role="confirmatory",
        normalize=True,
    )
    with pytest.raises(ConfigError):
        build_embedder(spec)
