"""Cut the flagged survey into the set-level tables used by the later Quarto files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from h4.survey.eligibility import mask_ineligible_ocq

_CITIZEN_KEEP = ("age", "gender", "gender_4_TEXT", "trans_cond")
_EXPERT_KEEP = ("age", "gender", "gender_4_TEXT", "highest_degree", "trans_cond")
_SETS = ("A", "B", "C", "D", "E", "F")


def analysis_rows_citizen(frame: pd.DataFrame) -> pd.DataFrame:
    """Three-rule citizen screen, kept as the sensitivity analysis.

    This cut also requires ``excl3 == 0``. That straight-lining flag was
    computed on the original item order. Presentation order was randomized
    and was not stored, so this cut is not the analysis sample.
    """

    return frame.loc[
        frame["analysis_target"].eq(1)
        & frame["excl1"].eq(0)
        & frame["excl2"].eq(0)
        & frame["excl3"].eq(0)
    ].copy()


def analysis_rows_citizen_two_rule(frame: pd.DataFrame) -> pd.DataFrame:
    """Analysis sample: finished assigned items, attention, and speeding only.

    Straight-lining (``excl3``) is not applied. The item order shown to each
    respondent was randomized and was not recorded, so a run computed on the
    original scale order is not the pre-specified screen.
    """

    return frame.loc[
        frame["analysis_target"].eq(1)
        & frame["excl1"].eq(0)
        & frame["excl2"].eq(0)
    ].copy()


def analysis_rows_expert(frame: pd.DataFrame) -> pd.DataFrame:
    """Eligible experts who finished the AI ratings and passed the attention checks."""

    return frame.loc[frame["analysis_target"].eq(1) & frame["excl1"].eq(0)].copy()


def write_set_tables(
    frame: pd.DataFrame,
    directory: Path,
    *,
    prefix: str,
    keep: tuple[str, ...],
) -> dict[str, int]:
    """Write one CSV per survey set. Counts are the rows written."""

    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for survey_set in _SETS:
        subset = frame.loc[frame["set_id"].astype("string").eq(survey_set)]
        columns = [column for column in keep if column in subset.columns]
        columns += [column for column in subset.columns if str(column).startswith(survey_set)]
        table = subset.loc[:, columns]
        table.to_csv(directory / f"{prefix}{survey_set}.csv", index=False, na_rep="NA")
        counts[survey_set] = int(len(table))
    return counts


def write_flags(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def citizen_sets(frame: pd.DataFrame, directory: Path, prefix: str) -> dict[str, int]:
    return write_set_tables(mask_ineligible_ocq(frame), directory, prefix=prefix, keep=_CITIZEN_KEEP)


def expert_sets(frame: pd.DataFrame, directory: Path, prefix: str) -> dict[str, int]:
    return write_set_tables(frame, directory, prefix=prefix, keep=_EXPERT_KEEP)
