"""Logit for the expert item checkbox, three dimensions, TIVS removed.

Each row is one expert, one item, one translation, and one problem dimension.
The checkbox is 1 when the expert marked that problem. The formula is

    flagged ~ version_human * delta_z + ai_first
              + (1 | respondent_id) + (1 | scale_id) + (1 | item_uid)

``delta_z`` is the MPNet back-translation cosine gap, human minus LLM,
standardized across the 486 primary items. ``ai_first`` is 1 when the LLM
translation was administered first. ``version_human`` is 1 on the human
translation. The fit writes coefficients and rates only. It does not write
the respondent file into the published cache.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

_FORMULA = "flagged ~ version_human * delta_z + ai_first + (1 | respondent_id) + (1 | scale_id) + (1 | item_uid)"
_DIMENSIONS = ("semantic", "understandability", "cultural")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the expert item-flag logits.")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = _frame(args.problems, args.similarity)
    rates = (
        frame.groupby(["dimension", "version"], as_index=False)
        .agg(n=("flagged", "size"), n_flagged=("flagged", "sum"), rate=("flagged", "mean"))
    )
    rates.to_csv(args.out / "expert_item_flag_rates.csv", index=False)
    r_script = Path(__file__).resolve().parents[1] / "src" / "h4" / "survey" / "lme4_fit.R"
    coef_rows = []
    status_rows = []
    for dimension in _DIMENSIONS:
        part = frame.loc[frame["dimension"].eq(dimension)].copy()
        data_path = args.work / f"{dimension}.csv"
        part.drop(columns=["dimension", "version"]).to_csv(data_path, index=False)
        prefix = args.work / dimension
        subprocess.run(
            ["Rscript", str(r_script), str(data_path), _FORMULA, "binomial", str(prefix), "0"],
            check=True,
        )
        coef = pd.read_csv(f"{prefix}_coef.csv")
        coef.insert(0, "dimension", dimension)
        status = pd.read_csv(f"{prefix}_status.csv")
        status.insert(0, "dimension", dimension)
        coef_rows.append(coef)
        status_rows.append(status)
    pd.concat(coef_rows, ignore_index=True).to_csv(args.out / "expert_item_flag_coef.csv", index=False)
    pd.concat(status_rows, ignore_index=True).to_csv(args.out / "expert_item_flag_status.csv", index=False)
    print({"n_rows": int(len(frame)), "n_items": int(frame["item_uid"].nunique()), "formula": _FORMULA})


def _frame(problems: Path, similarity: Path) -> pd.DataFrame:
    flags = pd.read_parquet(problems)
    flags["scale_id"] = flags["scale_id"].map(_key)
    flags["item_id"] = flags["item_id"].map(_key)
    flags = flags.loc[flags["scale_id"].ne("11")].copy()
    flags["flagged"] = flags["raw_value"].map(lambda value: int(str(value).strip() == "1"))
    flags["version_human"] = flags["version"].eq("human").astype(int)
    flags["ai_first"] = flags["trans_cond"].eq("ai").astype(int)
    flags["item_uid"] = flags["scale_id"] + "_" + flags["item_id"]
    gap = _item_delta(similarity)
    merged = flags.merge(gap, on=["scale_id", "item_id"], how="inner")
    return merged.loc[
        :,
        ["respondent_id", "scale_id", "item_uid", "dimension", "version", "flagged", "version_human", "ai_first", "delta_z"],
    ]


def _item_delta(similarity: Path) -> pd.DataFrame:
    frame = pd.read_parquet(similarity)
    part = frame.loc[
        frame["metric"].eq("cosine")
        & frame["embedding_model"].eq("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        & frame["comparison"].eq("back_translation"),
        ["scale_id", "item_id", "translation", "similarity"],
    ].copy()
    part["scale_id"] = part["scale_id"].map(_key)
    part["item_id"] = part["item_id"].map(_key)
    part = part.loc[part["scale_id"].ne("11")]
    wide = part.pivot_table(index=["scale_id", "item_id"], columns="translation", values="similarity", aggfunc="first")
    wide = wide.reset_index()
    wide["delta"] = wide["human"] - wide["llm"]
    spread = float(wide["delta"].std(ddof=1))
    wide["delta_z"] = (wide["delta"] - wide["delta"].mean()) / spread
    return wide.loc[:, ["scale_id", "item_id", "delta_z"]]


def _key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


if __name__ == "__main__":
    main()
