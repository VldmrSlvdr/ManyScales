"""Embedder interface.

An embedder maps texts to vectors. It does not know which strings are
human translations, LLM translations, or the H4 contrast.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    name: str
    short_name: str
    family: str
    revision: str | None
    pooling: str
    normalize: bool
    expected_dimension: int | None

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError(
            f"{self.name} does not expose contextual token embeddings."
        )

    def cls_vectors(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not expose a CLS vector.")
