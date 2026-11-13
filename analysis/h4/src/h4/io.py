"""CSV, Parquet, and JSON helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from h4.errors import SchemaError
from h4.preprocess import preprocess_items


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_item_table(path: Path, *, preanalysis_dir: Path | None = None) -> pd.DataFrame:
    """Load the H4 item table.

    The analysis source is the cleaned parquet (``items_clean.parquet``).
    A scale-catalog JSON is accepted while building that table: only study
    items are returned, and ``preanalysis_dir`` also receives the scale,
    instruction, and label tables. A CSV is the synthetic demo layout.
    """

    if not path.exists():
        raise SchemaError(
            f"Study file not found at {path}. "
            "The analysis source is data/processed/items_clean.parquet. "
            "Build it with scripts/00_extract_scales.py. "
            "A scale catalog JSON or a CSV with columns scale_id, item_id, "
            "original_en, human_ja, human_back_en, llm_ja, llm_back_en is also accepted."
        )
    suffix = path.suffix.lower()
    if suffix == ".json":
        from h4.catalog import analysis_items_from_catalog, load_catalog, write_preanalysis_tables

        catalog = load_catalog(path)
        if preanalysis_dir is not None:
            write_preanalysis_tables(catalog, preanalysis_dir)
        return analysis_items_from_catalog(catalog)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return preprocess_items(frame)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_embeddings(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    storable = frame.copy()
    storable["embedding"] = storable["embedding"].map(
        lambda vector: np.asarray(vector, dtype=np.float32).tolist()
    )
    storable.to_parquet(path, index=False)


def read_embeddings(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["scale_id"] = frame["scale_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    return frame


def align_field(
    embeddings: pd.DataFrame,
    items: pd.DataFrame,
    text_field: str,
) -> np.ndarray:
    """Return one embedding matrix in the same row order as ``items``."""

    want = items[["scale_id", "item_id"]].copy()
    want["scale_id"] = want["scale_id"].astype(str)
    want["item_id"] = want["item_id"].astype(str)
    subset = embeddings.loc[embeddings["text_field"] == text_field].copy()
    merged = want.merge(
        subset,
        on=["scale_id", "item_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["embedding"].isna().any():
        raise SchemaError(f"Embedding cache is missing rows for {text_field}.")
    return np.vstack([np.asarray(vector, dtype=np.float64) for vector in merged["embedding"]])
