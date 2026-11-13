"""Lexical overlap between two strings. Higher means more shared tokens."""

from __future__ import annotations

import re

_TOKENS = re.compile(r"\w+", flags=re.UNICODE)


def token_set(text: str) -> set[str]:
    return set(_TOKENS.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
