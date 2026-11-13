"""Sentence-Transformers adapter for open-weight sentence embedders."""

from __future__ import annotations

import numpy as np

from h4.embedders.base import Embedder
from h4.embedders.device import resolve_device


class SentenceTransformerEmbedder(Embedder):
    """Pinned sentence-transformers encoder.

    Sentence vectors come from the library's own pooling module.
    Token lists for exploratory work are extracted separately from the
    underlying transformer, using the raw item text.
    """

    family = "sentence_transformer"

    def __init__(
        self,
        name: str,
        *,
        short_name: str,
        normalize: bool = True,
        revision: str | None = None,
        expected_dimension: int | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.short_name = short_name
        self.normalize = normalize
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.device_name = resolve_device(device)
        self.model = SentenceTransformer(
            name,
            revision=revision,
            device=self.device_name,
        )
        self.revision = _commit_hash(self.model) or revision
        self.pooling = _pooling_name(self.model)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        _check_dimension(matrix, self.expected_dimension, self.name)
        return matrix


def _commit_hash(model: object) -> str | None:
    try:
        return model[0].auto_model.config._commit_hash  # type: ignore[index, attr-defined]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _pooling_name(model: object) -> str:
    try:
        modules = list(model)  # type: ignore[call-overload]
    except TypeError:
        return "sentence_transformers_default"
    for module in modules:
        if module.__class__.__name__ != "Pooling":
            continue
        if getattr(module, "pooling_mode_mean_tokens", False):
            return "mean"
        if getattr(module, "pooling_mode_cls_token", False):
            return "cls"
        if getattr(module, "pooling_mode_max_tokens", False):
            return "max"
    return "sentence_transformers_default"


def _check_dimension(matrix: np.ndarray, expected: int | None, name: str) -> None:
    if expected is not None and matrix.shape[1] != expected:
        raise ValueError(
            f"{name} produced dimension {matrix.shape[1]}; the registry expected {expected}."
        )
