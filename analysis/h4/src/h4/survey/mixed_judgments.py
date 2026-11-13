"""Crossed random-intercept tests of back-translation similarity and judgments.

Scale-level ratings, preference, and the LLM guess use a participant
intercept and a scale intercept. Item intercepts are used only for the
expert item-problem flags, which are the judgments recorded on items.

These fits do not enter the confirmatory cosine model.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from h4.survey.judgment import decode_guess, decode_preference

_MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_R = Path(__file__).with_name("lme4_fit.R")


def back_translation_delta(similarity_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Item and scale MPNet back-translation gaps, human cosine minus LLM cosine."""

    frame = pd.read_parquet(similarity_path)
    part = frame.loc[
        frame["metric"].eq("cosine")
        & frame["embedding_model"].eq(_MPNET)
        & frame["comparison"].eq("back_translation"),
        ["scale_id", "item_id", "translation", "similarity"],
    ].copy()
    part["scale_id"] = part["scale_id"].map(_key)
    part["item_id"] = part["item_id"].map(lambda value: str(value).zfill(2))
    wide = part.pivot_table(index=["scale_id", "item_id"], columns="translation", values="similarity", aggfunc="first")
    items = wide.reset_index()
    items["delta_back"] = items["human"] - items["llm"]
    items["abs_delta_back"] = items["delta_back"].abs()
    scales = items.groupby("scale_id", as_index=False).agg(
        delta_back=("delta_back", "mean"),
        abs_delta_back=("abs_delta_back", "mean"),
        n_items=("delta_back", "size"),
    )
    items["delta_z"] = _z(items["delta_back"])
    items["abs_delta_z"] = _z(items["abs_delta_back"])
    scales["delta_z"] = _z(scales["delta_back"])
    scales["abs_delta_z"] = _z(scales["abs_delta_back"])
    return items, scales


def rating_gap_frame(root: Path, scales: pd.DataFrame) -> pd.DataFrame:
    """Human minus LLM exported ratings; cultural scores are already high = good."""

    citizen = pd.read_parquet(root / "data/survey/4_analysis/citizen_scale_judgment.parquet")
    expert = pd.read_parquet(root / "data/survey/4_analysis/expert_scale_rating.parquet")
    citizen_rows = citizen.rename(columns={"wording_human": "human", "wording_llm": "llm"})
    citizen_rows["outcome"] = "naturalness"
    citizen_rows["expert"] = 0
    citizen_rows = citizen_rows.loc[:, ["respondent_id", "scale_id", "scale_name", "trans_cond", "expert", "outcome", "human", "llm"]]
    expert = expert.copy()
    expert["rating"] = pd.to_numeric(expert["raw_value"], errors="coerce")
    wide = expert.pivot_table(
        index=["respondent_id", "scale_id", "scale_name", "trans_cond", "dimension"],
        columns="version",
        values="rating",
        aggfunc="first",
    ).reset_index()
    wide["outcome"] = wide["dimension"]
    wide["expert"] = 1
    wide = wide.rename(columns={"human": "human", "llm": "llm"})
    rows = pd.concat([citizen_rows, wide.loc[:, citizen_rows.columns]], ignore_index=True)
    rows["human"] = pd.to_numeric(rows["human"], errors="coerce")
    rows["llm"] = pd.to_numeric(rows["llm"], errors="coerce")
    rows = rows.dropna(subset=["human", "llm", "trans_cond"])
    rows["rating_gap"] = rows["human"] - rows["llm"]
    rows["ai_first"] = rows["trans_cond"].eq("ai").astype(int)
    return rows.merge(scales.loc[:, ["scale_id", "delta_back", "delta_z", "abs_delta_back", "abs_delta_z"]], on="scale_id", how="left")


def choice_frame(root: Path, scales: pd.DataFrame) -> pd.DataFrame:
    """One Bernoulli row per respondent and scale for preference and the guess."""

    citizen = pd.read_parquet(root / "data/survey/4_analysis/citizen_scale_judgment.parquet")
    expert = pd.read_parquet(root / "data/survey/4_analysis/expert_scale_choice.parquet")
    citizen["expert"] = 0
    expert["expert"] = 1
    rows = pd.concat(
        [
            citizen.loc[:, ["respondent_id", "scale_id", "scale_name", "trans_cond", "expert", "preference_code", "llm_guess_code"]],
            expert.loc[:, ["respondent_id", "scale_id", "scale_name", "trans_cond", "expert", "preference_code", "llm_guess_code"]],
        ],
        ignore_index=True,
    )
    rows["prefer_human"] = rows["preference_code"].map(decode_preference)
    rows["guess_correct"] = rows["llm_guess_code"].map(decode_guess)
    rows["ai_first"] = rows["trans_cond"].eq("ai").astype(int)
    return rows.merge(scales.loc[:, ["scale_id", "delta_back", "delta_z", "abs_delta_back", "abs_delta_z"]], on="scale_id", how="left")


def item_flag_frame(root: Path, items: pd.DataFrame) -> pd.DataFrame:
    """Expert problem checkbox, one Bernoulli row per rater, item, version, and dimension."""

    flags = pd.read_parquet(root / "data/survey/4_analysis/expert_item_problem.parquet")
    flags["flagged"] = flags["raw_value"].eq("1").fillna(False).astype(int)
    flags["version_human"] = flags["version"].eq("human").astype(int)
    flags["ai_first"] = flags["trans_cond"].eq("ai").astype(int)
    flags["item_uid"] = flags["scale_id"].astype(str) + "_" + flags["item_id"].astype(str)
    keep = items.loc[:, ["scale_id", "item_id", "delta_back", "delta_z"]]
    return flags.merge(keep, on=["scale_id", "item_id"], how="inner")


def run(root: Path) -> pd.DataFrame:
    """Fit the rating, preference, and guess models and write the tables."""

    out = root / "results" / "survey_mixed"
    frames = out / "frames"
    fits = out / "fits"
    frames.mkdir(parents=True, exist_ok=True)
    fits.mkdir(parents=True, exist_ok=True)
    items, scales = back_translation_delta(root / "results/similarities/similarity_long.parquet")
    ratings = rating_gap_frame(root, scales)
    choices = choice_frame(root, scales)
    flags = item_flag_frame(root, items)
    ratings.to_parquet(frames / "rating_gaps.parquet", index=False)
    rating_long_frame(ratings).to_parquet(frames / "naturalness_long.parquet", index=False)
    choices.to_parquet(frames / "choices.parquet", index=False)
    flags.to_parquet(frames / "item_flags.parquet", index=False)
    scales.to_csv(frames / "scale_delta_back.csv", index=False)

    specs = _specifications(ratings, choices, flags)
    coef_rows = []
    var_rows = []
    status_rows = []
    for spec in specs:
        coef, var, status = _fit(spec["frame"], spec["formula"], spec["family"], fits / spec["name"], spec["nagq"])
        coef.insert(0, "model", spec["name"])
        var.insert(0, "model", spec["name"])
        status.insert(0, "model", spec["name"])
        status["family"] = spec["family"]
        status["formula"] = spec["formula"]
        status["level"] = spec["level"]
        coef_rows.append(coef)
        var_rows.append(var)
        status_rows.append(status)
    coefficients = pd.concat(coef_rows, ignore_index=True)
    variances = pd.concat(var_rows, ignore_index=True)
    status = pd.concat(status_rows, ignore_index=True)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    variances.to_csv(out / "variances.csv", index=False)
    status.to_csv(out / "model_status.csv", index=False)
    report = _report(coefficients, variances, status, ratings, choices)
    (root / "protocl" / "h4_mixed_judgment.md").write_text(report)
    return coefficients


def rating_long_frame(ratings: pd.DataFrame) -> pd.DataFrame:
    """Two rows per naturalness pair, so a version term can be estimated."""

    natural = ratings.loc[ratings["outcome"].eq("naturalness")].copy()
    human = natural.assign(rating=natural["human"], version_human=1)
    llm = natural.assign(rating=natural["llm"], version_human=0)
    return pd.concat([human, llm], ignore_index=True)


def _specifications(ratings: pd.DataFrame, choices: pd.DataFrame, flags: pd.DataFrame) -> list[dict]:
    natural = ratings.loc[ratings["outcome"].eq("naturalness")].copy()
    long = rating_long_frame(ratings)
    specs = [
        {
            "name": "naturalness_gap",
            "level": "scale",
            "family": "gaussian",
            "formula": "rating_gap ~ ai_first + expert + delta_z + (1 | respondent_id) + (1 | scale_id)",
            "frame": natural,
            "nagq": 1,
        },
        {
            "name": "naturalness_level",
            "level": "scale",
            "family": "gaussian",
            "formula": "rating ~ ai_first + expert + delta_z + (1 | respondent_id) + (1 | scale_id)",
            "frame": long,
            "nagq": 1,
        },
        {
            "name": "naturalness_version",
            "level": "scale",
            "family": "gaussian",
            "formula": "rating ~ version_human * delta_z + version_human * ai_first + expert + (1 | respondent_id) + (0 + version_human | respondent_id) + (1 | scale_id) + (0 + version_human | scale_id)",
            "frame": long,
            "nagq": 1,
        },
    ]
    for outcome in ("semantic", "cultural_validity", "choice_fit", "naturalness"):
        part = ratings.loc[ratings["outcome"].eq(outcome) & ratings["expert"].eq(1)].copy()
        specs.append(
            {
                "name": f"expert_{outcome}_gap",
                "level": "scale",
                "family": "gaussian",
                "formula": "rating_gap ~ ai_first + delta_z + (1 | respondent_id) + (1 | scale_id)",
                "frame": part,
                "nagq": 1,
            }
        )
    prefer = choices.dropna(subset=["prefer_human"]).copy()
    guess = choices.dropna(subset=["guess_correct"]).copy()
    specs.append(
        {
            "name": "preference",
            "level": "scale",
            "family": "binomial",
            "formula": "prefer_human ~ ai_first + expert + delta_z + (1 | respondent_id) + (1 | scale_id)",
            "frame": prefer,
            "nagq": 1,
        }
    )
    specs.append(
        {
            "name": "guess",
            "level": "scale",
            "family": "binomial",
            "formula": "guess_correct ~ ai_first + expert + delta_z + (1 | respondent_id) + (1 | scale_id)",
            "frame": guess,
            "nagq": 1,
        }
    )
    specs.append(
        {
            "name": "guess_abs_delta",
            "level": "scale",
            "family": "binomial",
            "formula": "guess_correct ~ ai_first + expert + abs_delta_z + (1 | respondent_id) + (1 | scale_id)",
            "frame": guess,
            "nagq": 1,
        }
    )
    for dimension, part in flags.groupby("dimension"):
        specs.append(
            {
                "name": f"item_flag_{dimension}",
                "level": "item",
                "family": "binomial",
                "formula": "flagged ~ version_human * delta_z + ai_first + (1 | respondent_id) + (1 | scale_id) + (1 | item_uid)",
                "frame": part,
                "nagq": 0,
            }
        )
    return specs


def _fit(
    frame: pd.DataFrame,
    formula: str,
    family: str,
    prefix: Path,
    nagq: int,
    ml: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_path = prefix.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    command = ["Rscript", str(_R), str(csv_path), formula, family, str(prefix), str(nagq)]
    if ml:
        command.append("ML")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:] or completed.stdout[-2000:])
    coef = pd.read_csv(prefix.with_name(prefix.name + "_coef.csv"))
    var = pd.read_csv(prefix.with_name(prefix.name + "_var.csv"))
    status = pd.read_csv(prefix.with_name(prefix.name + "_status.csv"))
    return coef, var, status


def _report(
    coefficients: pd.DataFrame,
    variances: pd.DataFrame,
    status: pd.DataFrame,
    ratings: pd.DataFrame,
    choices: pd.DataFrame,
) -> str:
    natural = ratings.loc[ratings["outcome"].eq("naturalness")]
    lines = [
        "# Crossed random-intercept tests of back-translation similarity",
        "",
        "The requested equations were checked against the level at which each outcome was recorded, then fit. The confirmatory cosine estimate is unchanged.",
        "",
        "## What the equations can identify",
        "",
        "Naturalness, semantic fidelity, cultural validity, choice fit, preference, and the LLM guess are one judgment per respondent per scale. The back-translation gap is one number per item. A participant who rates a scale once does not supply a separate naturalness score for each item, so an item intercept on those outcomes would repeat the same judgment once per item. That repetition does not identify an item variance, and it makes the standard errors too small.",
        "",
        "The random structure that matches the scale judgments is a participant intercept plus a scale intercept. The predictor is the scale mean of the MPNet back-translation gap, human cosine minus LLM cosine, standardized across the 22 scales. `ai_first` is 1 when `trans_cond` is `ai`. `expert` is 1 for the expert survey.",
        "",
        "For a rating of each version, the gap human minus LLM is the outcome. A positive coefficient on the back-translation gap means that scales whose human back-translation is closer to the original receive a higher human-minus-LLM rating. An additive model of the raw 1–5 score on that gap, without the version contrast, asks only whether both versions are rated higher on scales with a larger gap. The same contrast can be written as a version-by-gap interaction on the long rows. That interaction needs a random slope of version for the participant and for the scale. A version intercept alone reprints the coefficient and treats the two ratings inside a scale as if they independently identified a between-scale slope.",
        "",
        "Preference and the guess are Bernoulli. Preference is 1 when the respondent chose the human column. The guess is 1 when the respondent named the LLM column. Signed delta is the preference predictor. For the guess, the absolute item gap is the companion predictor, because detection depends on how far apart the two translations are, not on which one is closer to the English.",
        "",
        "Expert problem checkboxes are item judgments. Those models use a participant intercept, a scale intercept, and an item intercept. The version-by-gap interaction asks whether a human translation that is closer to the original is less often flagged.",
        "",
        f"Naturalness gaps: {len(natural)} respondent-scale pairs ({int((natural.expert == 0).sum())} citizen, {int((natural.expert == 1).sum())} expert). Preference rows: {int(choices.prefer_human.notna().sum())}. Guess rows: {int(choices.guess_correct.notna().sum())}.",
        "",
        "## Coefficients",
        "",
        _coef_table(coefficients, status),
        "",
        _prose(coefficients),
        "",
        "Variance components are in `results/survey_mixed/variances.csv`. A singular fit means one variance component sat on zero. In `naturalness_version` that component is the participant slope of version. The scale slope of version is not zero, and the interaction matches the gap model.",
        "",
    ]
    return "\n".join(lines)


def _coef_table(coefficients: pd.DataFrame, status: pd.DataFrame) -> str:
    header = "| Model | Term | Estimate | SE | 95% CI | p |"
    rule = "| --- | --- | ---: | ---: | --- | ---: |"
    body = []
    keep = coefficients.loc[~coefficients["term"].eq("(Intercept)")]
    for _, row in keep.iterrows():
        body.append(
            f"| {row['model']} | {row['term']} | {row['estimate']:.3f} | {row['se']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {_p(row['p'])} |"
        )
    singular = status.loc[status["singular"].astype(str).isin(["True", "TRUE", "true"]), "model"].tolist()
    note = "Singular fits: " + (", ".join(singular) if singular else "none") + "."
    return "\n".join([header, rule, *body, "", note])


def _prose(coefficients: pd.DataFrame) -> str:
    def term(model: str, name: str) -> pd.Series:
        hit = coefficients.loc[coefficients["model"].eq(model) & coefficients["term"].eq(name)]
        return hit.iloc[0]

    def sentence(label: str, row: pd.Series) -> str:
        return (
            f"{label} is {row['estimate']:.3f} "
            f"(95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], p {_p(row['p'])})."
        )

    gap = term("naturalness_gap", "delta_z")
    level = term("naturalness_level", "delta_z")
    version = term("naturalness_version", "version_human:delta_z")
    prefer = term("preference", "delta_z")
    guess = term("guess", "delta_z")
    guess_abs = term("guess_abs_delta", "abs_delta_z")
    cultural = term("item_flag_cultural", "version_human:delta_z")
    semantic = term("item_flag_semantic", "version_human:delta_z")
    understand = term("item_flag_understandability", "version_human:delta_z")
    choice = term("expert_choice_fit_gap", "delta_z")
    return " ".join(
        [
            sentence("The naturalness gap per standard deviation of the scale back-translation gap", gap),
            sentence("The same predictor on the raw 1–5 level, without a version contrast,", level),
            sentence("The version-by-gap interaction on those long rows", version),
            sentence("The expert choice-fit gap", choice),
            sentence("The log-odds of preferring the human column", prefer),
            sentence("The log-odds of naming the LLM column, using the signed gap,", guess),
            sentence("The same log-odds using the absolute item gap", guess_abs),
            "A negative version-by-gap interaction on a problem flag would mean the human column is flagged less often when its back-translation is closer to the original.",
            sentence("That interaction for semantic flags", semantic),
            sentence("The interaction for understandability", understand),
            sentence("The interaction for cultural validity", cultural),
        ]
    )


def _z(values: pd.Series) -> pd.Series:
    spread = float(values.std(ddof=1))
    if spread == 0 or not np.isfinite(spread):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / spread


def _key(value: object) -> str:
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() else text


def _p(value: float) -> str:
    if pd.isna(value):
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
