"""Multilingual E5 instruct adapter.

The published instruct model puts a task prefix on the query side only.
A symmetric item-to-item comparison applies that same prefix to every
item, so the human and LLM strings are transformed identically. Token-level
exploratory analyses use the raw item text and do not include this prefix.
"""

from __future__ import annotations

import numpy as np

from h4.embedders.base import Embedder
from h4.embedders.device import resolve_device
from h4.errors import ConfigError


class E5InstructEmbedder(Embedder):
    family = "e5_instruct"
    pooling = "mean"
    prompt_policy = "symmetric_instruction"

    def __init__(
        self,
        name: str,
        *,
        short_name: str,
        instruction: str,
        normalize: bool = True,
        revision: str | None = None,
        expected_dimension: int | None = None,
        device: str = "auto",
        batch_size: int = 32,
        prompt_policy: str | None = "symmetric_instruction",
    ) -> None:
        if prompt_policy not in (None, "symmetric_instruction"):
            raise ConfigError("E5 comparisons in this repo use prompt_policy: symmetric_instruction.")
        if not instruction or not str(instruction).strip():
            raise ConfigError("E5 instruct embedders require an instruction string.")
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.short_name = short_name
        self.instruction = " ".join(str(instruction).split())
        self.normalize = normalize
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.device_name = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision)
        self.model.to(self.device_name)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def prepare(self, text: str) -> str:
        return f"Instruct: {self.instruction}\nQuery: {text}"

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        prepared = [self.prepare(text) for text in texts]
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device_name) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                rows.append(pooled.cpu().numpy().astype(np.float32))
        matrix = np.vstack(rows) if rows else np.zeros((0, self.expected_dimension or 0), dtype=np.float32)
        if self.expected_dimension is not None and matrix.shape[1] != self.expected_dimension:
            raise ValueError(
                f"{self.name} produced dimension {matrix.shape[1]}; "
                f"the registry expected {self.expected_dimension}."
            )
        return matrix
