"""Signal-detection scoring of the generative-AI column guess.

The question is which of the two Japanese columns was produced by a
generative AI. Column 1 is always the human translation and column 2 is
always the LLM translation, so the response is a two-alternative forced
choice. Naming column 2 is a hit on the LLM text and a correct rejection
of the human text. Naming column 1 is a miss and a false alarm. Those
pairs are the same choice, so a yes/no criterion is not a separate
parameter. Sensitivity is the two-alternative d'.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from h4.survey.judgment import decode_guess
from h4.survey.mixed_judgments import _fit

_DRAWS = 5000
_SEED = 20260928


def score_guess(code: object) -> dict[str, float]:
    """Map one column choice onto the four detection outcomes."""

    named_llm = decode_guess(code)
    if not np.isfinite(named_llm):
        return {"hit": np.nan, "miss": np.nan, "false_alarm": np.nan, "correct_rejection": np.nan}
    hit = float(named_llm)
    return {
        "hit": hit,
        "miss": 1.0 - hit,
        "false_alarm": 1.0 - hit,
        "correct_rejection": hit,
    }


def dprime_2afc(proportion_correct: float) -> float:
    """√2 Z(P_correct). Zero is chance in a two-alternative forced choice."""

    bounded = float(np.clip(proportion_correct, 1e-6, 1 - 1e-6))
    return float(np.sqrt(2) * stats.norm.ppf(bounded))


def run(root: Path) -> pd.DataFrame:
    """Score the guess, estimate d', and write the report."""

    out = root / "results" / "sdt_guess"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    trials = _trials(root)
    trials.to_parquet(out / "trials.parquet", index=False)
    summary = _summarize(trials)
    summary.to_csv(out / "detection_summary.csv", index=False)
    scales = _by_scale(trials)
    scales.to_csv(out / "detection_by_scale.csv", index=False)
    model = _fit_detection(trials, out / "fits")
    model.to_csv(out / "detection_model.csv", index=False)
    _figure_rates(summary, figures / "detection_rates.png")
    _figure_dprime(summary, figures / "dprime.png")
    report = _report(summary, model)
    path = root / "protocl" / "h4_sdt_guess.md"
    path.write_text(report)
    return summary


def _trials(root: Path) -> pd.DataFrame:
    citizen = pd.read_parquet(root / "data/survey/4_analysis/citizen_scale_judgment.parquet")
    expert = pd.read_parquet(root / "data/survey/4_analysis/expert_scale_choice.parquet")
    citizen["expert"] = 0
    expert["expert"] = 1
    rows = pd.concat(
        [
            citizen.loc[:, ["respondent_id", "scale_id", "scale_name", "trans_cond", "expert", "llm_guess_code"]],
            expert.loc[:, ["respondent_id", "scale_id", "scale_name", "trans_cond", "expert", "llm_guess_code"]],
        ],
        ignore_index=True,
    )
    scored = rows["llm_guess_code"].map(score_guess).apply(pd.Series)
    rows = pd.concat([rows, scored], axis=1).dropna(subset=["hit"])
    rows["ai_first"] = rows["trans_cond"].eq("ai").astype(int)
    rows["survey"] = np.where(rows["expert"].eq(1), "expert", "citizen")
    return rows


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    groups = [("all", "all", trials)]
    for survey, part in trials.groupby("survey"):
        groups.append((survey, "all", part))
        for order, sub in part.groupby("trans_cond"):
            groups.append((survey, order, sub))
    return pd.DataFrame([_one_summary(survey, order, part) for survey, order, part in groups])


def _one_summary(survey: str, order: str, part: pd.DataFrame) -> dict[str, float | str | int]:
    hit = float(part["hit"].mean())
    point = dprime_2afc(hit)
    low, high = _bootstrap_dprime(part)
    return {
        "survey": survey,
        "trans_cond": order,
        "n_trials": int(len(part)),
        "n_respondents": int(part["respondent_id"].nunique()),
        "hit": hit,
        "miss": float(part["miss"].mean()),
        "false_alarm": float(part["false_alarm"].mean()),
        "correct_rejection": float(part["correct_rejection"].mean()),
        "dprime": point,
        "dprime_ci_low": low,
        "dprime_ci_high": high,
    }


def _bootstrap_dprime(part: pd.DataFrame) -> tuple[float, float]:
    people = part.groupby("respondent_id")["hit"].agg(["sum", "size"])
    draws = []
    rng = np.random.default_rng(_SEED)
    index = np.arange(len(people))
    hits = people["sum"].to_numpy()
    sizes = people["size"].to_numpy()
    for _ in range(_DRAWS):
        take = rng.choice(index, size=len(index), replace=True)
        draws.append(dprime_2afc(float(hits[take].sum() / sizes[take].sum())))
    values = np.asarray(draws)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _by_scale(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (survey, scale_id), part in trials.groupby(["survey", "scale_id"]):
        hit = float(part["hit"].mean())
        rows.append(
            {
                "survey": survey,
                "scale_id": scale_id,
                "scale_name": part["scale_name"].iloc[0],
                "n_trials": int(len(part)),
                "hit": hit,
                "false_alarm": 1.0 - hit,
                "dprime": dprime_2afc(hit),
            }
        )
    return pd.DataFrame(rows)


def _fit_detection(trials: pd.DataFrame, fits: Path) -> pd.DataFrame:
    fits.mkdir(parents=True, exist_ok=True)
    coef, _var, status = _fit(
        trials,
        "hit ~ expert + ai_first + (1 | respondent_id) + (1 | scale_id)",
        "binomial",
        fits / "detection",
        1,
    )
    coef.insert(0, "singular", bool(status["singular"].iloc[0]))
    return coef


def _figure_rates(summary: pd.DataFrame, path: Path) -> None:
    part = summary.loc[summary["survey"].isin(["citizen", "expert"]) & summary["trans_cond"].eq("all")]
    labels = ["Hit", "Miss", "False alarm", "Correct rejection"]
    keys = ["hit", "miss", "false_alarm", "correct_rejection"]
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(labels))
    width = 0.36
    for offset, survey, color in ((-width / 2, "citizen", "#1B1B1B"), (width / 2, "expert", "#4C78A8")):
        row = part.loc[part["survey"].eq(survey)].iloc[0]
        axis.bar(x + offset, [row[key] for key in keys], width=width, color=color, label=survey)
    axis.axhline(0.5, color="#888888", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_dprime(summary: pd.DataFrame, path: Path) -> None:
    part = summary.loc[summary["survey"].isin(["citizen", "expert"])].copy()
    part["label"] = part["survey"] + ", " + part["trans_cond"].map({"all": "both orders", "ai": "LLM first", "human": "human first"})
    part = part.iloc[::-1]
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.errorbar(
        part["dprime"],
        np.arange(len(part)),
        xerr=[part["dprime"] - part["dprime_ci_low"], part["dprime_ci_high"] - part["dprime"]],
        fmt="o",
        color="#1B1B1B",
        ecolor="#1B1B1B",
        capsize=3,
    )
    axis.set_yticks(np.arange(len(part)))
    axis.set_yticklabels(part["label"])
    axis.set_xlabel("Two-alternative d'")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report(summary: pd.DataFrame, model: pd.DataFrame) -> str:
    pooled = summary.loc[summary["survey"].eq("all")].iloc[0]
    lines = [
        "# Naming the LLM column, as a detection task",
        "",
        "The questionnaire asks which Japanese translation feels as if a generative AI made it. The choices are 翻訳1 and 翻訳2. In both surveys 翻訳1 is the human Japanese column and 翻訳2 is the LLM Japanese column. The earlier phrase \"name the LLM column\" means the respondent chose 翻訳2.",
        "",
        "That choice is a two-alternative forced choice. The LLM text is the signal, and it is always in column 2. The four detection outcomes of one choice are:",
        "",
        "|  | Respondent names 翻訳2 | Respondent names 翻訳1 |",
        "| --- | --- | --- |",
        "| 翻訳2 is the LLM text | Hit | Miss |",
        "| 翻訳1 is the human text | Correct rejection | False alarm |",
        "",
        "Naming 翻訳2 fills the hit cell and the correct-rejection cell together. Naming 翻訳1 fills the miss cell and the false-alarm cell together. The hit rate equals the correct-rejection rate, and the miss rate equals the false-alarm rate. A yes/no criterion is not a second free parameter, because the design never places the LLM text in column 1. Sensitivity is the two-alternative index d′ = √2 Z(hit rate). Zero is chance.",
        "",
        f"Pooled over {int(pooled.n_trials)} judgments from {int(pooled.n_respondents)} respondents, the hit rate is {pooled.hit:.3f}, so the false-alarm rate is {pooled.false_alarm:.3f}. d′ = {pooled.dprime:.3f}, respondent-bootstrap 95% CI [{pooled.dprime_ci_low:.3f}, {pooled.dprime_ci_high:.3f}].",
        "",
        "| Survey | Order | Trials | Hit | Miss | False alarm | Correct rejection | d′ | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.survey} | {row.trans_cond} | {int(row.n_trials)} | {row.hit:.3f} | {row.miss:.3f} | {row.false_alarm:.3f} | {row.correct_rejection:.3f} | {row.dprime:.3f} | [{row.dprime_ci_low:.3f}, {row.dprime_ci_high:.3f}] |"
        )
    intercept = model.loc[model["term"].eq("(Intercept)")].iloc[0]
    lines.extend(
        [
            "",
            "A binomial model of the hit, `hit ~ expert + ai_first + (1 | respondent_id) + (1 | scale_id)`, tests the same choice. Chance is a log-odds of 0. The intercept is the citizen, human-first log-odds of a hit.",
            "",
            "| Term | Estimate | SE | 95% CI | p |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )
    for _, row in model.iterrows():
        if row["term"] == "singular":
            continue
        lines.append(
            f"| {row.term} | {row.estimate:.3f} | {row.se:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | {_p(row.p)} |"
        )
    lines.extend(
        [
            "",
            f"At the citizen, human-first reference, the log-odds of a hit are {intercept.estimate:.3f} (95% CI [{intercept.ci_low:.3f}, {intercept.ci_high:.3f}]). An interval above 0 means d′ is above chance for that reference respondent and scale. The expert and AI-first coefficients are shifts in the log-odds of a hit.",
            "",
            "![Hit, miss, false alarm, and correct rejection](../results/sdt_guess/figures/detection_rates.png)",
            "",
            "![Two-alternative d′ by survey and administration order](../results/sdt_guess/figures/dprime.png)",
            "",
            "The hit rate is the outcome previously called `guess_correct`. Models of that outcome are models of the hit, not models with a separate false-alarm trial.",
            "",
        ]
    )
    return "\n".join(lines)


def _p(value: float) -> str:
    if pd.isna(value):
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
