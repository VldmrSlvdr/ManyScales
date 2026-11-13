"""Provenance recorded beside every result file."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from h4 import __version__
from h4.io import file_sha256


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    token = completed.stdout.strip()
    return token or None


def collect_metadata(root: Path, **fields: Any) -> dict[str, Any]:
    torch_version = _package_version("torch")
    payload: dict[str, Any] = {
        "h4_version": __version__,
        "python_version": sys.version.split()[0],
        "numpy_version": _package_version("numpy"),
        "pandas_version": _package_version("pandas"),
        "pyarrow_version": _package_version("pyarrow"),
        "statsmodels_version": _package_version("statsmodels"),
        "scipy_version": _package_version("scipy"),
        "torch_version": torch_version,
        "transformers_version": _package_version("transformers"),
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "git_commit": git_commit(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": {
            "top_blueprint": file_sha256(root / "protocl" / "top_blueprint.md"),
            "explorative_blueprint": file_sha256(root / "protocl" / "explorative_blueprint.md"),
        },
    }
    payload.update(fields)
    return payload
