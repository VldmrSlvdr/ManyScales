"""Local hashing embedder for tests and the synthetic demo.

The sentence vector is the L2-normalized mean of hashed token vectors.
It is a software stand-in, not a semantic-fidelity instrument.
"""

from __future__ import annotations

import hashlib

import numpy as np

from h4.embedders.base import Embedder


class DeterministicEmbedder(Embedder):
    family = "deterministic"
    pooling = "mean"
    revision = "local-sha256"

    def __init__(
        self,
        name: str = "deterministic-hash",
        *,
        short_name: str = "hash",
        dimension: int = 32,
        normalize: bool = True,
        expected_dimension: int | None = None,
    ) -> None:
        self.name = name
        self.short_name = short_name
        self.dimension = dimension
        self.normalize = normalize
        self.expected_dimension = expected_dimension or dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            mean = self._token_matrix(text).mean(axis=0)
            rows.append(_l2(mean) if self.normalize else mean)
        return np.vstack(rows).astype(np.float32)

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        return [self._token_matrix(text).astype(np.float32) for text in texts]

    def cls_vectors(self, texts: list[str]) -> np.ndarray:
        rows = [self._token_matrix(text)[0] for text in texts]
        return np.vstack(rows).astype(np.float32)

    def _token_matrix(self, text: str) -> np.ndarray:
        parts = text.split() or ["<empty>"]
        return np.vstack([_hash_token(part, self.dimension) for part in parts])


def _hash_token(token: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    vector = np.zeros(dimension, dtype=np.float64)
    index = int.from_bytes(digest[:4], "little") % dimension
    vector[index] = 1.0
    return vector


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        out = np.zeros_like(vector)
        out[0] = 1.0
        return out
    return vector / norm
