"""Propositional negation used only as a polarity diagnostic.

The rule is deliberately mechanical: prefix the original item with
"It is not the case that". It does not try to conjugate the verb.
"""

from __future__ import annotations

PREFIX = "It is not the case that "


def negate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(PREFIX):
        return stripped
    if stripped.endswith("."):
        return f"{PREFIX}{stripped[:-1]}."
    return f"{PREFIX}{stripped}"
