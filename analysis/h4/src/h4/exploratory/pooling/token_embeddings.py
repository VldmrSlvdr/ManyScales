"""Last-layer token states for exploratory representation checks.

Sentence-level H4 embeddings are not rebuilt from these states. Special
tokens and padding are removed for SOCM. CLS pooling reads position 0 of
the unpadded sequence, which still includes the CLS token when the
tokenizer emits one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from h4.embedders.device import resolve_device


class TransformerTokenEncoder:
    """Load one transformer and return content, CLS, and unpadded states."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.device_name = resolve_device(device)
        self.model.to(self.device_name)
        self.model.eval()
        self.special_ids = set(self.tokenizer.all_special_ids)

    def encode(self, texts: list[str]) -> list[dict[str, np.ndarray]]:
        batches: list[dict[str, np.ndarray]] = []
        torch = self.torch
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                attention = encoded["attention_mask"]
                hidden = self.model(
                    input_ids=input_ids.to(self.device_name),
                    attention_mask=attention.to(self.device_name),
                ).last_hidden_state.cpu().numpy()
                ids = input_ids.cpu().numpy()
                mask = attention.cpu().numpy()
                for row in range(hidden.shape[0]):
                    keep_content = (mask[row] == 1) & np.array(
                        [int(token_id) not in self.special_ids for token_id in ids[row]]
                    )
                    keep_unpadded = mask[row] == 1
                    content = hidden[row][keep_content]
                    if len(content) == 0:
                        content = hidden[row][keep_unpadded]
                    batches.append(
                        {
                            "content": content.astype(np.float32),
                            "cls": hidden[row][0].astype(np.float32),
                            "unpadded": hidden[row][keep_unpadded].astype(np.float32),
                        }
                    )
        return batches


def extract_token_batches(
    model_name: str,
    texts: list[str],
    *,
    device: str = "auto",
    batch_size: int = 8,
    revision: str | None = None,
) -> list[dict[str, np.ndarray]]:
    encoder = TransformerTokenEncoder(
        model_name,
        device=device,
        batch_size=batch_size,
        revision=revision,
    )
    return encoder.encode(texts)


def token_filename(scale_id: str, item_id: str, text_field: str) -> str:
    return f"{_safe(scale_id)}_{_safe(item_id)}_{_safe(text_field)}.npy"


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def save_token_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))
