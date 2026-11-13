"""Make the repository importable when a script is launched directly."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def with_config(argv: list[str], *, through: str | None = None) -> list[str]:
    args = list(argv)
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = ["--config", str(ROOT / "configs" / "analysis.yaml"), *args]
    if through is not None and not any(
        arg == "--through" or arg.startswith("--through=") for arg in args
    ):
        args = ["--through", through, *args]
    return args
