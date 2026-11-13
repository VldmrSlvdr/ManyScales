"""Build exploratory frames whose predictor is own similarity to the original.

Each back-translation has its own cosine, Wasserstein-1, and Wasserstein-2
to the original English. The human-minus-LLM difference is stored only as a
sensitivity column. Scale 11 (TIVS) is left out. Respondent identifiers stay
in this results directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "own_level" / "frames"
PUBLISHED = ROOT / "data" / "published"


def _z(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std(ddof=1)


def _key(value: object) -> str:
    return str(int(value)).zfill(2)


def similarities() -> pd.DataFrame:
    gaps = pd.read_csv(PUBLISHED / "mpnet_item_delta.csv")
    gaps = gaps.loc[gaps["comparison"].eq("back_translation")].copy()
    gaps["scale_id"] = gaps["scale_id"].map(_key)
    gaps["item_id"] = gaps["item_id"].map(_key)
    gaps = gaps.loc[gaps["in_primary"].astype(bool), ["scale_id", "item_id", "human", "llm"]]
    gaps = gaps.rename(columns={"human": "cosine_human", "llm": "cosine_llm"})

    token = pd.read_csv(PUBLISHED / "token_item_distance.csv")
    token = token.loc[
        token["short_name"].eq("mpnet") & token["comparison"].eq("back_translation")
    ].copy()
    token["scale_id"] = token["scale_id"].map(_key)
    token["item_id"] = token["item_id"].map(_key)
    token = token.loc[
        token["in_primary"].astype(bool),
        ["scale_id", "item_id", "distance_human", "distance_llm", "w2_human", "w2_llm"],
    ]
    token = token.rename(columns={"distance_human": "w1_human", "distance_llm": "w1_llm"})
    items = gaps.merge(token, on=["scale_id", "item_id"], how="inner")
    if items.shape[0] != 486:
        raise RuntimeError(f"expected 486 primary items, found {items.shape[0]}")
    return items


def scale_scores(items: pd.DataFrame) -> pd.DataFrame:
    means = items.groupby("scale_id", as_index=False).mean(numeric_only=True)
    for metric in ("cosine", "w1", "w2"):
        both = pd.concat([means[f"{metric}_human"], means[f"{metric}_llm"]], ignore_index=True)
        center, spread = both.mean(), both.std(ddof=1)
        means[f"{metric}_human_z"] = (means[f"{metric}_human"] - center) / spread
        means[f"{metric}_llm_z"] = (means[f"{metric}_llm"] - center) / spread
        relative = means[f"{metric}_human"] - means[f"{metric}_llm"]
        means[f"relative_{metric}_z"] = _z(relative)
    return means


def item_scores(items: pd.DataFrame) -> pd.DataFrame:
    long = []
    for version, side in (("human", "human"), ("llm", "llm")):
        part = items.loc[:, ["scale_id", "item_id"]].copy()
        part["version"] = version
        for metric in ("cosine", "w1", "w2"):
            part[metric] = items[f"{metric}_{side}"].to_numpy()
            part[f"relative_{metric}"] = (items[f"{metric}_human"] - items[f"{metric}_llm"]).to_numpy()
        long.append(part)
    stacked = pd.concat(long, ignore_index=True)
    for metric in ("cosine", "w1", "w2"):
        stacked[f"{metric}_z"] = _z(stacked[metric])
        stacked[f"relative_{metric}_z"] = _z(stacked[f"relative_{metric}"])
    return stacked


def _ai_first(frame: pd.DataFrame) -> pd.Series:
    return frame["trans_cond"].eq("ai").astype(int)


def _choice(code: object, positive: str) -> float:
    """Map a two-option Qualtrics code onto 1 when it equals ``positive``.

    Stored tables use ``1`` or ``1.0`` depending on whether the column was
    written as text or as a number.
    """

    text = str(code).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return np.nan
    if text.endswith(".0"):
        text = text[:-2]
    if text == positive:
        return 1.0
    if text in {"1", "2"}:
        return 0.0
    return np.nan


def ratings(scales: pd.DataFrame, citizen_path: Path | None = None) -> pd.DataFrame:
    analysis = ROOT / "data" / "survey" / "4_analysis"
    citizen = pd.read_parquet(citizen_path or (analysis / "citizen_scale_judgment.parquet"))
    citizen = citizen.loc[~citizen["scale_id"].eq("11")].copy()
    human = citizen.assign(version="human", rating=pd.to_numeric(citizen["wording_human"], errors="coerce"))
    llm = citizen.assign(version="llm", rating=pd.to_numeric(citizen["wording_llm"], errors="coerce"))
    citizen_long = pd.concat([human, llm], ignore_index=True)
    citizen_long["outcome"] = "naturalness"
    citizen_long["expert"] = 0

    expert = pd.read_parquet(analysis / "expert_scale_rating.parquet")
    expert = expert.loc[~expert["scale_id"].eq("11")].copy()
    # Qualtrics already scores cultural_validity as high = better validity.
    # Preserve exported codes; applying 6 - raw_value would reverse it twice.
    expert["rating"] = pd.to_numeric(expert["raw_value"], errors="coerce")
    expert["outcome"] = expert["dimension"]
    expert["expert"] = 1

    rows = pd.concat(
        [
            citizen_long.loc[:, ["respondent_id", "scale_id", "version", "trans_cond", "outcome", "expert", "rating"]],
            expert.loc[:, ["respondent_id", "scale_id", "version", "trans_cond", "outcome", "expert", "rating"]],
        ],
        ignore_index=True,
    )
    rows = rows.dropna(subset=["rating", "trans_cond"])
    rows["ai_first"] = _ai_first(rows)
    keep = [column for column in scales.columns if column.endswith("_z") or column == "scale_id"]
    rows = rows.merge(scales.loc[:, keep], on="scale_id", how="inner")
    rows["sim_cosine"] = np.where(rows["version"].eq("human"), rows["cosine_human_z"], rows["cosine_llm_z"])
    rows["sim_w1"] = np.where(rows["version"].eq("human"), rows["w1_human_z"], rows["w1_llm_z"])
    rows["sim_w2"] = np.where(rows["version"].eq("human"), rows["w2_human_z"], rows["w2_llm_z"])
    return rows


def choices(scales: pd.DataFrame, citizen_path: Path | None = None) -> pd.DataFrame:
    analysis = ROOT / "data" / "survey" / "4_analysis"
    citizen = pd.read_parquet(citizen_path or (analysis / "citizen_scale_judgment.parquet"))
    expert = pd.read_parquet(analysis / "expert_scale_choice.parquet")
    citizen["expert"] = 0
    expert["expert"] = 1
    rows = pd.concat(
        [
            citizen.loc[:, ["respondent_id", "scale_id", "trans_cond", "expert", "preference_code", "llm_guess_code"]],
            expert.loc[:, ["respondent_id", "scale_id", "trans_cond", "expert", "preference_code", "llm_guess_code"]],
        ],
        ignore_index=True,
    )
    rows = rows.loc[~rows["scale_id"].eq("11")].copy()
    rows["prefer_human"] = rows["preference_code"].map(lambda code: _choice(code, "1"))
    rows["guess_llm"] = rows["llm_guess_code"].map(lambda code: _choice(code, "2"))
    rows["ai_first"] = _ai_first(rows)
    keep = [column for column in scales.columns if column.endswith("_z") or column == "scale_id"]
    return rows.merge(scales.loc[:, keep], on="scale_id", how="inner")


def flags(items: pd.DataFrame) -> pd.DataFrame:
    analysis = ROOT / "data" / "survey" / "4_analysis"
    frame = pd.read_parquet(analysis / "expert_item_problem.parquet")
    frame = frame.loc[~frame["scale_id"].eq("11")].copy()
    frame["item_id"] = frame["item_id"].map(_key)
    frame["flagged"] = frame["raw_value"].eq("1").fillna(False).astype(int)
    frame["ai_first"] = _ai_first(frame)
    frame["version_human"] = frame["version"].eq("human").astype(int)
    frame["item_uid"] = frame["scale_id"] + "_" + frame["item_id"]
    scored = items.loc[:, ["scale_id", "item_id", "version", "cosine_z", "w1_z", "w2_z", "relative_cosine_z", "relative_w1_z", "relative_w2_z"]]
    merged = frame.merge(scored, on=["scale_id", "item_id", "version"], how="inner")
    return merged.loc[:, [
        "respondent_id", "scale_id", "item_uid", "dimension", "version", "version_human",
        "ai_first", "flagged", "cosine_z", "w1_z", "w2_z",
        "relative_cosine_z", "relative_w1_z", "relative_w2_z",
    ]]


def main(out: Path | None = None, citizen_path: Path | None = None) -> None:
    destination = out or OUT
    destination.mkdir(parents=True, exist_ok=True)
    items = similarities()
    scales = scale_scores(items)
    item_long = item_scores(items)
    rating = ratings(scales, citizen_path)
    choice = choices(scales, citizen_path)
    flag = flags(item_long)
    scales.to_csv(destination / "scale_similarity.csv", index=False)
    rating.to_csv(destination / "rating_long.csv", index=False)
    choice.to_csv(destination / "choice.csv", index=False)
    flag.to_csv(destination / "flag_long.csv", index=False)
    print("items", items.shape[0], "scales", scales.shape[0])
    print("ratings", rating.groupby("outcome").size().to_dict(), "people", rating.groupby("outcome")["respondent_id"].nunique().to_dict())
    print("choices", choice.shape[0], "people", choice["respondent_id"].nunique())
    print("flags", flag.groupby("dimension").size().to_dict(), "items", flag["item_uid"].nunique(), "people", flag["respondent_id"].nunique())
    print("cosine human-llm scale correlation", scales["cosine_human"].corr(scales["cosine_llm"]))


if __name__ == "__main__":
    import sys

    out_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    citizen_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    main(out_arg, citizen_arg)
