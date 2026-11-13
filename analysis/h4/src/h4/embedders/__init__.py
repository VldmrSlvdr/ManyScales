"""Construct an embedder from a registry entry."""

from __future__ import annotations

from h4.config import ModelSpec
from h4.embedders.base import Embedder
from h4.errors import ConfigError


def build_embedder(spec: ModelSpec, *, device: str = "cpu", batch_size: int = 32) -> Embedder:
    if spec.role == "confirmatory" and spec.family == "api":
        raise ConfigError(
            "Confirmatory H4 uses a pinned open-weight embedder. API embeddings stay off that path."
        )
    if spec.family == "sentence_transformer":
        from h4.embedders.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            spec.name,
            short_name=spec.short_name,
            normalize=spec.normalize,
            revision=spec.revision,
            expected_dimension=spec.expected_dimension,
            device=device,
            batch_size=batch_size,
        )
    if spec.family == "e5_instruct":
        from h4.embedders.e5 import E5InstructEmbedder

        return E5InstructEmbedder(
            spec.name,
            short_name=spec.short_name,
            instruction=spec.instruction or "",
            normalize=spec.normalize,
            revision=spec.revision,
            expected_dimension=spec.expected_dimension,
            device=device,
            batch_size=batch_size,
            prompt_policy=spec.prompt_policy,
        )
    if spec.family == "deterministic":
        from h4.embedders.deterministic import DeterministicEmbedder

        return DeterministicEmbedder(
            spec.name,
            short_name=spec.short_name,
            dimension=spec.dimension or spec.expected_dimension or 32,
            normalize=spec.normalize,
            expected_dimension=spec.expected_dimension,
        )
    if spec.family == "api":
        from h4.embedders.api_embedding import OpenAIEmbedder

        return OpenAIEmbedder(
            spec.name,
            short_name=spec.short_name,
            normalize=spec.normalize,
            expected_dimension=spec.expected_dimension,
            batch_size=batch_size,
            revision=spec.revision,
        )
    raise ConfigError(f"Unknown embedder family: {spec.family}")
