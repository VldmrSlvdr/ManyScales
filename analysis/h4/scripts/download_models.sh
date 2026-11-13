#!/usr/bin/env bash
# Download the four open-weight embedders into the local Hugging Face cache.
# The cache directory is HF_HOME, or ~/.cache/huggingface when that variable is unset.
# The script does not print the cache path.

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

python3 - << 'PY'
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(
        "Install the embedding extra first: pip install -e \".[embeddings]\""
    ) from exc

models = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large-instruct",
]
for name in models:
    SentenceTransformer(name)
    print(f"cached {name}")

hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
print(f"cache entries present: {sum(1 for _ in hub.glob('models--*'))}")
PY
