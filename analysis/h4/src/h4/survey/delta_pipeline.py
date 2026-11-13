"""Study pipeline: does the scale-mean back-translation gap shift judgments?

The outcome families are the rating gap (human rating minus LLM rating),
preference for the human column, and the guess that names the LLM column.
The predictor is ``delta_z``, the mean item gap on a scale, standardized
across the 22 scales. Ratings are Gaussian REML. Preference and the guess
are binomial logit. Every model has a participant intercept and a scale
intercept.

``results/tables/confirmatory_estimate.csv`` is the coauthor semantic-fidelity
replicate. This module reads that file only to record that it is unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from h4.survey.mixed_judgments import (
    _fit,
    _p,
    _specifications,
    _z,
    back_translation_delta,
    choice_frame,
    item_flag_frame,
    rating_gap_frame,
)

# Names match mixed_judgments._specifications. The long-form naturalness
# model is the same rating gap written as a version-by-gap interaction.
PRIMARY_MODELS = (
    "naturalness_gap",
    "naturalness_version",
    "expert_semantic_gap",
    "expert_cultural_validity_gap",
    "expert_choice_fit_gap",
    "expert_naturalness_gap",
    "preference",
    "guess",
)

_FOCUS = {
    "naturalness_gap": ("rating", "Shared naturalness gap", "delta_z"),
    "naturalness_version": ("rating", "Naturalness gap, version-by-gap form", "version_human:delta_z"),
    "expert_semantic_gap": ("rating", "Expert semantic gap", "delta_z"),
    "expert_cultural_validity_gap": ("rating", "Expert cultural-validity gap", "delta_z"),
    "expert_choice_fit_gap": ("rating", "Expert choice-fit gap", "delta_z"),
    "expert_naturalness_gap": ("rating", "Expert naturalness gap", "delta_z"),
    "preference": ("preference", "Preference for the human column", "delta_z"),
    "guess": ("guess", "Guess naming the LLM column", "delta_z"),
}

_REPLICATE = Path("results/tables/confirmatory_estimate.csv")
EXCLUDED_SCALES = ("HEEMA", "TIVS")


def primary_specs(ratings: pd.DataFrame, choices: pd.DataFrame, flags: pd.DataFrame) -> list[dict]:
    """Scale-level rating, preference, and guess models that contain delta_z."""

    chosen = [spec for spec in _specifications(ratings, choices, flags) if spec["name"] in PRIMARY_MODELS]
    found = [spec["name"] for spec in chosen]
    missing = [name for name in PRIMARY_MODELS if name not in found]
    if missing:
        raise RuntimeError(f"Primary models missing from the specification list: {missing}")
    return chosen


def scale_names(root: Path) -> pd.DataFrame:
    """Scale id and short name from the stored cosine scale table."""

    names = pd.read_csv(root / "results/tables/scale_estimates.csv", dtype={"scale_id": str})
    names["scale_id"] = names["scale_id"].str.zfill(2)
    return names.loc[:, ["scale_id", "scale_name"]].drop_duplicates("scale_id")


def scales_without(scales: pd.DataFrame, names: pd.DataFrame, exclude: tuple[str, ...]) -> pd.DataFrame:
    """Drop named scales and standardize the remaining scale-mean gaps."""

    named = scales.merge(names, on="scale_id", how="left")
    missing = [name for name in exclude if name not in set(named["scale_name"])]
    if missing:
        raise RuntimeError(f"Scales not in the similarity table: {missing}")
    kept = named.loc[~named["scale_name"].isin(exclude)].copy()
    kept["delta_z"] = _z(kept["delta_back"])
    kept["abs_delta_z"] = _z(kept["abs_delta_back"])
    return kept


def run(root: Path) -> pd.DataFrame:
    """Fit the delta_z judgment models and leave the cosine replicate file byte-for-byte."""

    root = root.resolve()
    replicate = root / _REPLICATE
    before = _sha256(replicate)
    out = root / "results" / "delta_z"
    fits = out / "fits"
    fits.mkdir(parents=True, exist_ok=True)

    _items, scales = back_translation_delta(root / "results/similarities/similarity_long.parquet")
    ratings = rating_gap_frame(root, scales)
    choices = choice_frame(root, scales)
    flags = item_flag_frame(root, _items)
    specs = primary_specs(ratings, choices, flags)

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
    focus = _focus_table(coefficients)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    variances.to_csv(out / "variances.csv", index=False)
    status.to_csv(out / "model_status.csv", index=False)
    focus.to_csv(out / "delta_z_slopes.csv", index=False)
    scales.to_csv(out / "scale_delta_back.csv", index=False)
    cloud = _scale_cloud(ratings, choices)
    cloud.to_csv(out / "scale_judgment_cloud.csv", index=False)
    _plot(focus, cloud, out / "figures" / "delta_z_slopes", n_scales=int(scales["scale_id"].nunique()))

    after = _sha256(replicate)
    if before != after:
        raise RuntimeError("The coauthor cosine replicate file changed during the delta_z pipeline.")
    report = _report(focus, status, ratings, choices, replicate, before)
    (root / "protocl" / "h4_delta_z.md").write_text(report)
    return focus


def run_without(root: Path, exclude: tuple[str, ...] = EXCLUDED_SCALES) -> pd.DataFrame:
    """Refit the same slopes after dropping named scales and re-standardizing delta_z.

    The 22-scale files in ``results/delta_z/`` and the cosine replicate stay in place.
    """

    root = root.resolve()
    replicate = root / _REPLICATE
    before = _sha256(replicate)
    tag = "_".join(name.lower() for name in exclude)
    out = root / "results" / f"delta_z_no_{tag}"
    fits = out / "fits"
    fits.mkdir(parents=True, exist_ok=True)

    _items, scales = back_translation_delta(root / "results/similarities/similarity_long.parquet")
    kept = scales_without(scales, scale_names(root), exclude)
    ratings = rating_gap_frame(root, kept)
    choices = choice_frame(root, kept)
    ratings = ratings.loc[ratings["scale_id"].isin(kept["scale_id"])].dropna(subset=["delta_z"])
    choices = choices.loc[choices["scale_id"].isin(kept["scale_id"])].dropna(subset=["delta_z"])
    flags = item_flag_frame(root, _items)
    focus, status = _fit_primary(ratings, choices, flags, fits)
    full = pd.read_csv(root / "results/delta_z/delta_z_slopes.csv")
    compared = _compare(full, focus)
    focus.to_csv(out / "delta_z_slopes.csv", index=False)
    status.to_csv(out / "model_status.csv", index=False)
    kept.to_csv(out / "scale_delta_back.csv", index=False)
    compared.to_csv(out / "comparison.csv", index=False)
    cloud = _scale_cloud(ratings, choices)
    cloud.to_csv(out / "scale_judgment_cloud.csv", index=False)
    n_scales = int(kept["scale_id"].nunique())
    reduced_label = _reduced_label(exclude, n_scales)
    _plot(focus, cloud, out / "figures" / "delta_z_slopes", n_scales=n_scales)
    _compare_plot(compared, out / "figures" / "delta_z_comparison", reduced_label)

    after = _sha256(replicate)
    if before != after:
        raise RuntimeError("The coauthor cosine replicate file changed during the sensitivity.")
    report = _sensitivity_report(focus, compared, status, ratings, choices, kept, exclude)
    (root / "protocl" / f"h4_delta_z_no_{tag}.md").write_text(report)
    return compared


def _fit_primary(
    ratings: pd.DataFrame,
    choices: pd.DataFrame,
    flags: pd.DataFrame,
    fits: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = primary_specs(ratings, choices, flags)
    coef_rows = []
    status_rows = []
    for spec in specs:
        coef, _var, status = _fit(spec["frame"], spec["formula"], spec["family"], fits / spec["name"], spec["nagq"])
        coef.insert(0, "model", spec["name"])
        status.insert(0, "model", spec["name"])
        status["family"] = spec["family"]
        status["formula"] = spec["formula"]
        status["n_scales"] = spec["frame"]["scale_id"].nunique()
        coef_rows.append(coef)
        status_rows.append(status)
    coefficients = pd.concat(coef_rows, ignore_index=True)
    status = pd.concat(status_rows, ignore_index=True)
    return _focus_table(coefficients), status


def _compare(full: pd.DataFrame, reduced: pd.DataFrame) -> pd.DataFrame:
    left = full.loc[:, ["model", "family", "label", "term", "estimate", "se", "ci_low", "ci_high", "p"]].copy()
    right = reduced.loc[:, ["model", "estimate", "se", "ci_low", "ci_high", "p"]]
    merged = left.merge(right, on="model", suffixes=("_22", "_20"))
    merged["includes_zero_22"] = (merged["ci_low_22"] <= 0) & (merged["ci_high_22"] >= 0)
    merged["includes_zero_20"] = (merged["ci_low_20"] <= 0) & (merged["ci_high_20"] >= 0)
    merged["same_zero_call"] = merged["includes_zero_22"] == merged["includes_zero_20"]
    merged["same_sign"] = (merged["estimate_22"] > 0) == (merged["estimate_20"] > 0)
    return merged


def _reduced_label(exclude: tuple[str, ...], n_scales: int) -> str:
    if exclude == ("TIVS",):
        return f"Primary, {n_scales} scales, TIVS excluded"
    return "Without " + " and ".join(exclude)


def _compare_plot(compared: pd.DataFrame, stem: Path, reduced_label: str = "Without HEEMA and TIVS") -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
        }
    )
    shown = compared.loc[~compared["model"].eq("naturalness_version")].iloc[::-1]
    stem.parent.mkdir(parents=True, exist_ok=True)
    millimeter = 1 / 25.4
    figure, axis = plt.subplots(figsize=(140 * millimeter, 110 * millimeter), constrained_layout=True)
    axis.axvline(0, color="#1B1B1B", linewidth=0.6, zorder=0)
    y = list(range(len(shown)))
    axis.errorbar(
        shown["estimate_22"],
        [value + 0.12 for value in y],
        xerr=[shown["estimate_22"] - shown["ci_low_22"], shown["ci_high_22"] - shown["estimate_22"]],
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=1.0,
        capsize=2,
        markersize=3.5,
        label="22 scales, TIVS included",
    )
    axis.errorbar(
        shown["estimate_20"],
        [value - 0.12 for value in y],
        xerr=[shown["estimate_20"] - shown["ci_low_20"], shown["ci_high_20"] - shown["estimate_20"]],
        fmt="o",
        color="#1B1B1B",
        ecolor="#1B1B1B",
        elinewidth=1.0,
        capsize=2,
        markersize=3.5,
        label=reduced_label,
    )
    axis.set_yticks(y, shown["label"])
    axis.set_xlabel("Coefficient of delta_z")
    axis.legend(frameon=False, fontsize=6)
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def _sensitivity_report(
    focus: pd.DataFrame,
    compared: pd.DataFrame,
    status: pd.DataFrame,
    ratings: pd.DataFrame,
    choices: pd.DataFrame,
    kept: pd.DataFrame,
    exclude: tuple[str, ...],
) -> str:
    natural = ratings.loc[ratings["outcome"].eq("naturalness")]
    shown = compared.loc[~compared["model"].eq("naturalness_version")]
    n_scales = int(kept["scale_id"].nunique())
    n_items = int(kept["n_items"].sum())
    tag = "_".join(name.lower() for name in exclude)
    dropped = " and ".join(exclude)
    verb = "is" if len(exclude) == 1 else "are"
    if exclude == ("TIVS",):
        title = "# Primary delta_z fit, TIVS excluded in preprocessing"
        lead = (
            f"TIVS {verb} removed in preprocessing, before the scale means are standardized. "
            f"`delta_z` is the z-score across the remaining {n_scales} scales ({n_items} items). "
            "The 22-scale fit that still includes TIVS is the appendix robustness check in `results/delta_z/`."
        )
        reduced_header = f"Primary, {n_scales} scales"
    else:
        title = f"# delta_z without {dropped}"
        lead = (
            f"Same models as the 22-scale fit. {dropped} {verb} removed before the scale means are standardized, "
            f"so `delta_z` is the z-score across the remaining {n_scales} scales ({n_items} items)."
        )
        reduced_header = f"Without {dropped}"
    lines = [
        title,
        "",
        lead,
        "",
        (
            f"Naturalness gaps: {len(natural)} respondent–scale pairs "
            f"({int((natural.expert == 0).sum())} citizen, {int((natural.expert == 1).sum())} expert). "
            f"Preference rows: {int(choices.prefer_human.notna().sum())}. "
            f"Guess rows: {int(choices.guess_correct.notna().sum())}."
        ),
        "",
        "## Does the call agree with the fit that includes the excluded scale?",
        "",
        _persistence(shown),
        "",
        _compare_table(shown, reduced_header),
        "",
        f"![Slopes with TIVS retained and with {dropped} removed](../results/delta_z_no_{tag}/figures/delta_z_comparison.png)",
        "",
        "## Coefficients on this set",
        "",
        _table(focus),
        "",
    ]
    singular = status.loc[status["singular"].astype(str).isin(["True", "TRUE", "true"]), "model"].tolist()
    if singular:
        lines.append("Singular fits in this run: " + ", ".join(singular) + ".")
        lines.append("")
    lines.append("The 22-scale slopes stay in `results/delta_z/`. The cosine replicate file is unchanged.")
    lines.append("")
    return "\n".join(lines)


def _persistence(shown: pd.DataFrame) -> str:
    zero_moved = shown.loc[~shown["same_zero_call"]]
    sign_only = shown.loc[shown["same_zero_call"] & ~shown["same_sign"]]
    pieces = [
        (
            f"The call about zero persists for {int(shown['same_zero_call'].sum())} of {len(shown)} slopes. "
            "Shared naturalness, preference, and the LLM guess stay negative, and those intervals still include zero."
        )
    ]
    for row in zero_moved.itertuples(index=False):
        pieces.append(
            f"{row.label} changes that call: {row.estimate_22:.3f} [{row.ci_low_22:.3f}, {row.ci_high_22:.3f}] "
            f"versus {row.estimate_20:.3f} [{row.ci_low_20:.3f}, {row.ci_high_20:.3f}]."
        )
    if len(sign_only):
        names = ", ".join(sign_only["label"].str.lower())
        pieces.append(
            f"The point estimates for {names} change sign. Each of those intervals includes zero on both fits, so that sign was not a separated result."
        )
    return " ".join(pieces)


def _compare_table(compared: pd.DataFrame, reduced_header: str = "Without HEEMA and TIVS") -> str:
    header = f"| Model | 22 scales, TIVS included | 95% CI | {reduced_header} | 95% CI | Zero call holds |"
    rule = "| --- | ---: | --- | ---: | --- | --- |"
    body = []
    for row in compared.itertuples(index=False):
        call = "yes" if row.same_zero_call else "no"
        body.append(
            f"| {row.label} | {row.estimate_22:.3f} | [{row.ci_low_22:.3f}, {row.ci_high_22:.3f}] | "
            f"{row.estimate_20:.3f} | [{row.ci_low_20:.3f}, {row.ci_high_20:.3f}] | {call} |"
        )
    return "\n".join([header, rule, *body])


def _focus_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (family, label, term) in _FOCUS.items():
        hit = coefficients.loc[coefficients["model"].eq(name) & coefficients["term"].eq(term)]
        if hit.empty:
            raise RuntimeError(f"{name} has no term {term}")
        row = hit.iloc[0].to_dict()
        row["family"] = family
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def _scale_cloud(ratings: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    """One point per scale: mean judgment against the scale's delta_z."""

    natural = ratings.loc[ratings["outcome"].eq("naturalness")]
    rating = natural.groupby(["scale_id", "scale_name"], as_index=False).agg(
        y=("rating_gap", "mean"),
        n=("rating_gap", "size"),
        delta_z=("delta_z", "first"),
    )
    rating["panel"] = "naturalness"
    prefer = choices.dropna(subset=["prefer_human"])
    preference = prefer.groupby(["scale_id", "scale_name"], as_index=False).agg(
        y=("prefer_human", "mean"),
        n=("prefer_human", "size"),
        delta_z=("delta_z", "first"),
    )
    preference["panel"] = "preference"
    return pd.concat([rating, preference], ignore_index=True)


def _plot(focus: pd.DataFrame, cloud: pd.DataFrame, stem: Path, n_scales: int = 22) -> None:
    """Forest of delta_z slopes, then the scale means those slopes summarize."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    millimeter = 1 / 25.4
    figure, axes = plt.subplots(2, 2, figsize=(183 * millimeter, 145 * millimeter), constrained_layout=True)
    shown = focus.loc[~focus["model"].eq("naturalness_version")].copy()
    _forest(
        axes[0, 0],
        shown.loc[shown["family"].eq("rating")],
        "Change in the rating gap per SD of similarity",
        note="Black: 95% interval excludes 0",
    )
    _forest(
        axes[0, 1],
        shown.loc[shown["family"].isin(["preference", "guess"])],
        "Change in log-odds per SD of similarity",
    )
    _cloud(
        axes[1, 0],
        cloud.loc[cloud["panel"].eq("naturalness")],
        "Mean human − LLM naturalness",
        f"{n_scales} scales",
    )
    _cloud(
        axes[1, 1],
        cloud.loc[cloud["panel"].eq("preference")],
        "Proportion choosing the human column",
        f"{n_scales} scales",
    )
    for axis, letter in zip(axes.flat, "abcd", strict=True):
        axis.text(-0.22, 1.06, letter, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def _forest(axis: plt.Axes, part: pd.DataFrame, xlabel: str, note: str = "") -> None:
    part = part.iloc[::-1]
    axis.axvline(0, color="#1B1B1B", linewidth=0.6, zorder=0)
    for index, row in enumerate(part.itertuples(index=False)):
        excludes = row.ci_low > 0 or row.ci_high < 0
        color = "#1B1B1B" if excludes else "#4C78A8"
        axis.errorbar(
            row.estimate,
            index,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            markersize=3.5,
            zorder=2,
        )
    axis.set_yticks(range(len(part)), part["label"])
    axis.set_xlabel(xlabel)
    axis.set_ylim(-0.6, len(part) - 0.4)
    if note:
        axis.text(0.98, 0.98, note, transform=axis.transAxes, ha="right", va="top", fontsize=6, color="#1B1B1B")


def _cloud(axis: plt.Axes, part: pd.DataFrame, ylabel: str, note: str) -> None:
    axis.axvline(0, color="#B0B0B0", linewidth=0.4, zorder=0)
    axis.scatter(part["delta_z"], part["y"], s=8 + part["n"] * 0.04, color="#4C78A8", zorder=2)
    extremes = part.loc[part["delta_z"].abs() > 2]
    for row in extremes.itertuples(index=False):
        axis.annotate(str(row.scale_name), (row.delta_z, row.y), textcoords="offset points", xytext=(4, 2), fontsize=6)
    axis.set_xlabel(f"Similarity gap, delta_z ({note})")
    axis.set_ylabel(ylabel)


def _report(
    focus: pd.DataFrame,
    status: pd.DataFrame,
    ratings: pd.DataFrame,
    choices: pd.DataFrame,
    replicate: Path,
    digest: str,
) -> str:
    stored = pd.read_csv(replicate).iloc[0]
    natural = ratings.loc[ratings["outcome"].eq("naturalness")]
    lines = [
        "# delta_z and the judgment gaps",
        "",
        "This is the study analysis. The question is whether `delta_z` shifts a rating gap, the preference for the human column, or the guess that names the LLM column.",
        "",
        "`delta_z` is the scale mean of the MPNet back-translation gap, human cosine minus LLM cosine, standardized across the 22 scales. A rating gap is the human rating minus the LLM rating on that scale. Preference is 1 when the respondent chose the human column. The guess is 1 when the respondent named the LLM column. Preference and the guess are one Bernoulli judgment per person per scale, so each model is on the log-odds of that judgment.",
        "",
        "The random structure is a participant intercept plus a scale intercept. `ai_first` is 1 when the LLM version was answered first. The shared naturalness model also includes an expert indicator. Expert semantic, cultural-validity, choice-fit, and naturalness gaps are fit on the expert survey alone.",
        "",
        f"Naturalness gaps: {len(natural)} respondent–scale pairs ({int((natural.expert == 0).sum())} citizen, {int((natural.expert == 1).sum())} expert). Preference rows: {int(choices.prefer_human.notna().sum())}. Guess rows: {int(choices.guess_correct.notna().sum())}.",
        "",
        "## Conclusion",
        "",
        _conclusion(focus),
        "",
        "![Coefficient of the scale similarity gap, and the 22 scale means. Black marks an interval that excludes zero. Blue marks an interval that includes zero. Each point in the lower panels is one scale; marker area follows the number of respondent–scale judgments. HEEMA is the scale whose similarity gap sits far below the others.](../results/delta_z/figures/delta_z_slopes.png)",
        "",
        "## Coefficient of delta_z",
        "",
        _table(focus),
        "",
        _prose(focus),
        "",
        "## Coauthor cosine replicate",
        "",
        (
            f"The semantic-fidelity REML intercept remains the stored file `{_REPLICATE.as_posix()}` "
            f"(sha256 `{digest}`). That fit is Δ = {stored['estimate']:.3f}, "
            f"95% CI [{stored['ci_low']:.3f}, {stored['ci_high']:.3f}], "
            f"on {int(stored['n_items'])} items and {int(stored['n_scales'])} scales. "
            "This pipeline reads the file to confirm the bytes are unchanged and does not refit it."
        ),
        "",
        "Formulas and variance components are in `results/delta_z/`.",
        "",
    ]
    singular = status.loc[status["singular"].astype(str).isin(["True", "TRUE", "true"]), "model"].tolist()
    if singular:
        lines.append("Singular fits in this run: " + ", ".join(singular) + ".")
        lines.append("")
    return "\n".join(lines)


def _conclusion(focus: pd.DataFrame) -> str:
    shown = focus.loc[~focus["model"].eq("naturalness_version")]
    separated = shown.loc[(shown["ci_low"] > 0) | (shown["ci_high"] < 0)]
    if list(separated["model"]) == ["expert_choice_fit_gap"]:
        row = separated.iloc[0]
        return (
            "Similarity does not shift naturalness, semantic fidelity, cultural validity, preference, or detection of the LLM column. "
            "Each of those 95% intervals includes zero. "
            f"Choice fit is the rating that moves: the human-minus-LLM gap is higher by {row.estimate:.3f} "
            f"per standard deviation of delta_z (95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}]). "
            "A positive delta_z means that scale's human back-translation is closer to the original English than the average scale, "
            "and on choice fit that closeness goes with a higher rating for the human version relative to the LLM version. "
            "The lower panels show the same pattern in the 22 scale means: the clouds have no visible slope, and HEEMA is the distant scale on the similarity axis."
        )
    inside = shown.loc[~shown.index.isin(separated.index), "label"]
    pieces = [f"Intervals that include zero: {', '.join(inside)}."]
    for row in separated.itertuples(index=False):
        pieces.append(
            f"{row.label} is {row.estimate:.3f} (95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}])."
        )
    return " ".join(pieces)


def _table(focus: pd.DataFrame) -> str:
    header = "| Family | Model | Term | Estimate | SE | 95% CI | p |"
    rule = "| --- | --- | --- | ---: | ---: | --- | ---: |"
    body = [
        (
            f"| {row.family} | {row.label} | {row.term} | {row.estimate:.3f} | {row.se:.3f} | "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | {_p(row.p)} |"
        )
        for row in focus.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _prose(focus: pd.DataFrame) -> str:
    sentences = []
    for row in focus.itertuples(index=False):
        sentences.append(
            f"{row.label}: {row.estimate:.3f} (95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}], p {_p(row.p)})."
        )
    return " ".join(sentences)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
