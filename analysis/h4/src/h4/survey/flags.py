"""Exclusion flags from the citizen and expert Quarto notebooks.

The rules are the ones in ``1_分析フラグ付与_市民.qmd`` and
``2_分析フラグ付与_専門家.qmd``. A flagged respondent is kept in the flagged
table and dropped only when the analysis tables are cut.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from h4.errors import SchemaError

_ITEM = re.compile(r"^([A-F])([0-9]{2})_item([0-9]+)_(human|ai)$")
_TIMER = re.compile(r"^([A-F])([0-9]{2})_timer_(human|ai)$")
_EXPERT_DONE = re.compile(
    r"^([A-F])([0-9]{2})_(semantic|naturalness|cultural_validity|choice_fit)_ai$"
)
_CHECKS = ("dqs01_check", "dqs02_check", "dqs03_check", "ai-trap_check")
_ELIGIBILITY = (
    "eligibility_native_japanese",
    "eligibility_scale_paper",
    "eligibility_english_citation",
    "eligibility_masters_or_higher",
)


def add_citizen_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``analysis_target``, ``excl1``, ``excl2``, and ``excl3``."""

    data = frame.copy()
    data["_row_id"] = np.arange(len(data))
    items = [column for column in data.columns if _ITEM.match(str(column))]
    if not items:
        raise SchemaError("Citizen export has no scale item columns.")
    _require(data, ("set_id", "mcs_gender", *_CHECKS))
    data["analysis_target"] = _citizen_complete(data, items)
    data["excl1"] = _citizen_attention(data)
    data["excl2"] = _speeding(data)
    excl3 = _straightlining(data, items)
    for column in excl3.columns:
        data[column] = excl3[column].to_numpy()
    return data.drop(columns="_row_id")


def add_expert_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``analysis_target`` and ``excl1`` for the expert survey."""

    data = frame.copy()
    data["_row_id"] = np.arange(len(data))
    ratings = [column for column in data.columns if _EXPERT_DONE.match(str(column))]
    if not ratings:
        raise SchemaError("Expert export has no AI translation rating columns.")
    _require(data, ("set_id", *_ELIGIBILITY, *_CHECKS))
    complete = _expert_complete(data, ratings)
    eligible = np.ones(len(data), dtype=bool)
    for column in _ELIGIBILITY:
        eligible &= data[column].map(_filled).to_numpy() & data[column].astype("string").eq("1").fillna(False).to_numpy()
    data["analysis_target"] = (complete & eligible).astype(int)
    data["excl1"] = _expert_attention(data)
    return data.drop(columns="_row_id")


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError("Survey export is missing " + ", ".join(missing) + ".")


def _filled(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip() != ""


def _citizen_complete(data: pd.DataFrame, items: list[str]) -> pd.Series:
    """1 when every assigned item is answered. MCS on sets C/D is male-only."""

    set_id = data["set_id"].astype("string")
    gender = data["mcs_gender"].astype("string")
    male = gender.eq("male").fillna(False).to_numpy()
    answered = np.ones(len(data), dtype=bool)
    seen = np.zeros(len(data), dtype=bool)
    for column in items:
        match = _ITEM.match(column)
        assert match is not None
        survey_set, scale_id, _item, _condition = match.groups()
        in_set = set_id.eq(survey_set).fillna(False).to_numpy()
        mcs_optional = np.logical_and(
            np.isin(set_id.fillna("").to_numpy(), ["C", "D"]),
            np.logical_and(scale_id == "10", ~male),
        )
        required = np.logical_and(in_set, ~mcs_optional)
        values = data[column].map(_filled).to_numpy()
        answered &= ~required | values
        seen |= required
    return pd.Series(np.where(seen, answered, False).astype(int), index=data.index)


def _citizen_attention(data: pd.DataFrame) -> pd.Series:
    """1 unless every attention item is the correct code ``0``. Blank counts as wrong."""

    failed = np.zeros(len(data), dtype=bool)
    for column in _CHECKS:
        values = data[column].astype("string").str.strip()
        failed |= ~values.eq("0").fillna(False).to_numpy()
    return pd.Series(failed.astype(int), index=data.index)


def _expert_attention(data: pd.DataFrame) -> pd.Series:
    """1 when a displayed attention item is the wrong code ``1``. Blank is not a failure."""

    failed = np.zeros(len(data), dtype=bool)
    for column in _CHECKS:
        values = data[column].astype("string").str.strip()
        failed |= values.eq("1").fillna(False).to_numpy()
    return pd.Series(failed.astype(int), index=data.index)


def _expert_complete(data: pd.DataFrame, ratings: list[str]) -> np.ndarray:
    set_id = data["set_id"].astype("string")
    answered = np.ones(len(data), dtype=bool)
    seen = np.zeros(len(data), dtype=bool)
    for column in ratings:
        match = _EXPERT_DONE.match(column)
        assert match is not None
        required = set_id.eq(match.group(1)).fillna(False).to_numpy()
        answered &= ~required | data[column].map(_filled).to_numpy()
        seen |= required
    return np.where(seen, answered, False)


def _speeding(data: pd.DataFrame) -> pd.Series:
    """1 when a scale's human-plus-AI page time is below that scale's mean minus 3 SD.

    The cutoff uses only respondents with ``analysis_target == 1``. Everyone
    else receives 0. Timers are pooled by the two-digit scale id, so the same
    scale on sets C and D shares one cutoff.
    """

    timers = [column for column in data.columns if _TIMER.match(str(column))]
    if not timers:
        raise SchemaError("Citizen export has no scale timer columns.")
    complete = data["analysis_target"].eq(1)
    rows = []
    for column in timers:
        match = _TIMER.match(column)
        assert match is not None
        _survey_set, scale_id, condition = match.groups()
        rows.append(
            pd.DataFrame(
                {
                    "_row_id": data["_row_id"].to_numpy(),
                    "scale_id": scale_id,
                    "condition": condition,
                    "value": pd.to_numeric(data[column], errors="coerce").to_numpy(),
                    "complete": complete.to_numpy(),
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    long = long.loc[long["complete"]]
    grouped = long.groupby(["_row_id", "scale_id", "condition"], sort=False)["value"]
    time = grouped.apply(lambda values: np.nan if values.isna().all() else float(np.nansum(values.to_numpy())))
    wide = time.unstack("condition")
    for column in ("human", "ai"):
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.reset_index()
    wide["total_time"] = wide["human"] + wide["ai"]
    stats = wide.groupby("scale_id")["total_time"]
    cutoff = stats.transform("mean") - 3.0 * stats.transform("std")
    wide["fast"] = wide["total_time"].notna() & (wide["total_time"] < cutoff)
    hit = wide.groupby("_row_id")["fast"].any()
    flagged = data["_row_id"].map(hit).fillna(False).astype(bool)
    return pd.Series(np.where(complete.to_numpy(), flagged.to_numpy(), False).astype(int), index=data.index)


def _straightlining(data: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """Flag a scale when one response repeats for ceiling(min(10, 0.8 * n)) items."""

    pieces = []
    for column in items:
        match = _ITEM.match(column)
        assert match is not None
        _survey_set, scale_id, item_number, condition = match.groups()
        pieces.append(
            pd.DataFrame(
                {
                    "_row_id": data["_row_id"].to_numpy(),
                    "scale_prefix": f"{_survey_set}{scale_id}",
                    "scale_id": scale_id,
                    "condition": condition,
                    "item": int(item_number),
                    "value": data[column].astype("string").to_numpy(),
                }
            )
        )
    long = pd.concat(pieces, ignore_index=True)
    long = long.sort_values(["_row_id", "scale_prefix", "condition", "item"], kind="mergesort")
    records = []
    for key, group in long.groupby(["_row_id", "scale_prefix", "scale_id", "condition"], sort=False):
        row_id, _prefix, scale_id, condition = key
        values = group["value"].tolist()
        run = _longest_run(values)
        n_items = len(values)
        records.append(
            {
                "_row_id": row_id,
                "scale_id": scale_id,
                "condition": condition,
                "longest_run": run,
                "flag": run >= math.ceil(min(10.0, 0.8 * n_items)),
            }
        )
    hits = pd.DataFrame(records)
    out = pd.DataFrame({"_row_id": data["_row_id"].to_numpy()})
    if hits.empty:
        out["excl3human"] = 0
        out["excl3ai"] = 0
        out["excl3"] = 0
        out["excl3where"] = ""
        return out.drop(columns="_row_id")
    flagged = hits.loc[hits["flag"]]
    human = set(flagged.loc[flagged["condition"].eq("human"), "_row_id"])
    ai = set(flagged.loc[flagged["condition"].eq("ai"), "_row_id"])
    out["excl3human"] = out["_row_id"].isin(human).astype(int)
    out["excl3ai"] = out["_row_id"].isin(ai).astype(int)
    out["excl3"] = ((out["excl3human"] == 1) | (out["excl3ai"] == 1)).astype(int)
    for scale_id in sorted(hits["scale_id"].unique()):
        for condition in ("human", "ai"):
            rows = set(
                flagged.loc[
                    flagged["scale_id"].eq(scale_id) & flagged["condition"].eq(condition),
                    "_row_id",
                ]
            )
            out[f"excl{scale_id}{condition}"] = out["_row_id"].isin(rows).astype(int)
    where = (
        flagged.loc[:, ["_row_id", "scale_id", "condition"]]
        .drop_duplicates()
        .sort_values(["_row_id", "scale_id", "condition"], kind="mergesort")
    )
    where["label"] = where["scale_id"] + "_" + where["condition"]
    joined = where.groupby("_row_id", sort=False)["label"].agg("；".join)
    out["excl3where"] = out["_row_id"].map(joined).fillna("")
    return out.drop(columns="_row_id")


def _longest_run(values: list[object]) -> int:
    """Longest streak of the same answered code. A blank breaks the streak."""

    best = 0
    current = 0
    previous = None
    for value in values:
        if not _filled(value):
            current = 0
            previous = None
            continue
        token = str(value).strip()
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        best = max(best, current)
    return best
