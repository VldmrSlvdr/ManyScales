"""Resolve a torch device name without importing torch when the run stays on CPU."""

from __future__ import annotations


def resolve_device(setting: str) -> str:
    if setting in {"cpu", "api"}:
        return "cpu" if setting == "cpu" else setting
    try:
        import torch
    except ImportError:
        return "cpu"
    if setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return setting
