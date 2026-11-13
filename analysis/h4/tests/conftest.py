from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def items() -> pd.DataFrame:
    rows = []
    for scale in range(1, 4):
        for item in range(1, 5):
            rows.append(
                {
                    "scale_id": f"S0{scale}",
                    "item_id": f"{item:02d}",
                    "original_en": f"Scale {scale} item {item} original statement.",
                    "human_ja": f"尺度{scale}項目{item}の人手訳。",
                    "human_back_en": f"Scale {scale} item {item} human back translation.",
                    "llm_ja": f"尺度{scale}項目{item}のLLM訳。",
                    "llm_back_en": f"Scale {scale} item {item} machine back translation.",
                    "reverse_keyed": item == 4,
                    "human_translator_id": "T1",
                }
            )
    return pd.DataFrame(rows)
