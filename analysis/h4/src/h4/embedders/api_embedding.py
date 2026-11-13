"""Optional API embedding adapter.

This class is a later robustness hook. The confirmatory loader rejects
``family: api`` before an adapter of this kind can become the primary
instrument.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from h4.embedders.base import Embedder


class OpenAIEmbedder(Embedder):
    family = "api"
    pooling = "provider_default"

    def __init__(
        self,
        name: str,
        *,
        short_name: str,
        normalize: bool = True,
        expected_dimension: int | None = None,
        batch_size: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        revision: str | None = None,
    ) -> None:
        self.name = name
        self.short_name = short_name
        self.normalize = normalize
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.api_key_env = api_key_env
        self.revision = revision
        self.device_name = "api"

    def encode(self, texts: list[str]) -> np.ndarray:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Set {self.api_key_env} before calling the API embedder. "
                "The key is read from the environment and is not written to result files."
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload = json.dumps({"model": self.name, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            ordered = sorted(body["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in ordered)
        matrix = np.asarray(vectors, dtype=np.float64)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix.astype(np.float32)
