"""Fixed wide-to-long reshape of the processed survey tables.

The processed CSVs stay in Qualtrics column layout. This module is the
only place that turns them into the analysis tables for expert judgment
and citizen item responses. Confirmatory cosine scoring does not read
these tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from h4.catalog import load_catalog
from h4.errors import SchemaError
from h4.survey.eligibility import eligible_long_rows, mask_ineligible_ocq

SETS = tuple("ABCDEF")
VERSION = {"human": "human", "ai": "llm"}
PROBLEM_DIMENSION = {"1": "semantic", "2": "understandability", "3": "cultural"}

_ITEM = re.compile(r"^([A-F])(\d{2})_item(\d+)_(human|ai)$")
_WORDING = re.compile(r"^([A-F])(\d{2})_wording_(human|ai)$")
_RATING = re.compile(
    r"^([A-F])(\d{2})_(semantic|naturalness|cultural_validity|choice_fit)_(human|ai)$"
)
_PROBLEM = re.compile(r"^([A-F])(\d{2})_problem_items_(human|ai)_(\d+)_([123])$")
_PREFERENCE = re.compile(r"^([A-F])(\d{2})_preference$")
_GUESS = re.compile(r"^([A-F])(\d{2})_llm_guess$")


def load_processed(directory: Path, prefix: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for survey_set in SETS:
        path = directory / f"{prefix}{survey_set}.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames[survey_set] = frame.replace({"": pd.NA, "NA": pd.NA})
    return frames


def reshape_survey(frames: dict[str, pd.DataFrame], survey: str) -> dict[str, pd.DataFrame]:
    """Return the long tables for one survey, preserving exported rating codes.

    Expert cultural-validity ratings are already reverse-coded by Qualtrics:
    disagreement with unfamiliar expressions is 5, agreement is 1. Do not
    apply ``6 - value`` here; higher exported scores mean better validity.
    The separate cultural problem checkboxes remain 0/1 problem indicators.
    """

    if survey == "citizen":
        return {
            "item_response": _melt_pattern(frames, survey, _ITEM, _item_fields),
            "scale_judgment": _citizen_judgments(frames),
        }
    if survey == "expert":
        return {
            "scale_rating": _melt_pattern(frames, survey, _RATING, _rating_fields),
            "item_problem": _melt_pattern(frames, survey, _PROBLEM, _problem_fields),
            "scale_choice": _expert_choices(frames),
        }
    raise SchemaError(f"Unknown survey: {survey}")


def expected_item_columns(items: pd.DataFrame) -> set[str]:
    columns = set()
    for row in items.itertuples(index=False):
        scale_id = _scale_key(row.scale_id)
        item_id = str(row.item_id).zfill(2)
        for survey_set in _sets(row.survey_set):
            for version in ("human", "ai"):
                columns.add(f"{survey_set}{scale_id}_item{item_id}_{version}")
    return columns


def observed_item_columns(frames: dict[str, pd.DataFrame]) -> set[str]:
    found = set()
    for frame in frames.values():
        found.update(column for column in frame.columns if _ITEM.match(str(column)))
    return found


def allowed_scores(labels: pd.DataFrame) -> dict[str, set[str]]:
    """Response codes legal for each scale.

    A filled score column is the code Qualtrics stored. A blank score column
    means the label id is the code. Anchor-only rows, such as 1, 4, and 7,
    allow every integer between the lowest and highest anchor.
    """

    allowed: dict[str, set[str]] = {}
    table = labels.copy()
    table["scale_id"] = table["scale_id"].map(_scale_key)
    for scale_id, group in table.groupby("scale_id", sort=True):
        scores = [str(value).strip() for value in group["score"].tolist() if _filled(value)]
        if scores:
            allowed[str(scale_id)] = set(scores)
            continue
        ids = []
        for value in group["label_id"].tolist():
            if _filled(value) and str(value).strip().isdigit():
                ids.append(int(str(value).strip()))
        if not ids:
            allowed[str(scale_id)] = set()
            continue
        allowed[str(scale_id)] = {str(number) for number in range(min(ids), max(ids) + 1)}
    return allowed


def response_bounds(labels: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for scale_id, codes in allowed_scores(labels).items():
        numbers = [float(code) for code in codes]
        if numbers:
            bounds[scale_id] = (min(numbers), max(numbers))
    return bounds


def check_tables(
    *,
    citizen_frames: dict[str, pd.DataFrame],
    expert_frames: dict[str, pd.DataFrame],
    citizen: dict[str, pd.DataFrame],
    expert: dict[str, pd.DataFrame],
    items: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Hard checks. A failed row means the reshape is not the analysis table."""

    rows = [
        _check_keys(citizen["item_response"], ["respondent_id", "scale_id", "item_id", "version"], "citizen item keys are unique"),
        _check_keys(citizen["scale_judgment"], ["respondent_id", "scale_id"], "citizen scale judgments are unique"),
        _check_keys(expert["scale_rating"], ["respondent_id", "scale_id", "dimension", "version"], "expert ratings are unique"),
        _check_keys(expert["item_problem"], ["respondent_id", "scale_id", "item_id", "dimension", "version"], "expert problem flags are unique"),
        _check_keys(expert["scale_choice"], ["respondent_id", "scale_id"], "expert choices are unique"),
        _check_coverage(citizen_frames, items),
        _check_roundtrip(citizen_frames, citizen["item_response"], "citizen"),
        _check_roundtrip(expert_frames, expert["scale_rating"], "expert rating"),
        _check_roundtrip(expert_frames, expert["item_problem"], "expert problem"),
        _check_copied_column(citizen_frames, citizen["scale_judgment"], _WORDING, "wording_{version}", "citizen wording"),
        _check_codes(citizen["item_response"], allowed_scores(labels)),
        _check_rating_range(expert["scale_rating"]),
        _check_flags(expert["item_problem"]),
        _check_order(citizen_frames, "citizen"),
        _check_order(expert_frames, "expert"),
        _check_counts(citizen_frames, citizen["item_response"], "citizen"),
        _check_counts(expert_frames, expert["scale_rating"], "expert"),
    ]
    return pd.DataFrame(rows)


def assert_checks(checks: pd.DataFrame) -> None:
    failed = checks.loc[~checks["passed"]]
    if failed.empty:
        return
    detail = "; ".join(f"{row.check}: {row.detail}" for row in failed.itertuples(index=False))
    raise SchemaError(f"Survey reshape checks failed. {detail}")


def _melt_pattern(frames: dict[str, pd.DataFrame], survey: str, pattern: re.Pattern[str], fields) -> pd.DataFrame:
    pieces = []
    for survey_set, frame in frames.items():
        columns = [column for column in frame.columns if pattern.match(str(column))]
        if not columns:
            continue
        base = _base_frame(frame, survey, survey_set)
        wide = pd.concat([base, frame.loc[:, columns].reset_index(drop=True)], axis=1)
        long = wide.melt(id_vars=list(base.columns), value_vars=columns, var_name="source_column", value_name="raw_value")
        parsed = long["source_column"].map(lambda column, pattern=pattern: fields(pattern.match(str(column))))
        piece = pd.concat([long, pd.DataFrame(list(parsed))], axis=1)
        if survey == "citizen":
            piece = eligible_long_rows(piece, frame, survey_set)
        pieces.append(piece)
    if not pieces:
        raise SchemaError(f"{survey} reshape found no columns for {pattern.pattern}.")
    return pd.concat(pieces, ignore_index=True)


def _item_fields(match: re.Match[str]) -> dict[str, str]:
    return {
        "scale_id": match.group(2),
        "item_id": match.group(3).zfill(2),
        "version": VERSION[match.group(4)],
    }


def _rating_fields(match: re.Match[str]) -> dict[str, str]:
    return {
        "scale_id": match.group(2),
        "dimension": match.group(3),
        "version": VERSION[match.group(4)],
    }


def _problem_fields(match: re.Match[str]) -> dict[str, str]:
    return {
        "scale_id": match.group(2),
        "version": VERSION[match.group(3)],
        "item_id": match.group(4).zfill(2),
        "dimension": PROBLEM_DIMENSION[match.group(5)],
    }


def _citizen_judgments(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for survey_set, frame in frames.items():
        stubs = sorted({match.group(2) for column in frame.columns if (match := _WORDING.match(str(column)))})
        base = _base_frame(frame, "citizen", survey_set)
        for scale_id in stubs:
            stub = f"{survey_set}{scale_id}"
            part = base.copy()
            part["scale_id"] = scale_id
            part["wording_human"] = _numeric(frame[f"{stub}_wording_human"])
            part["wording_llm"] = _numeric(frame[f"{stub}_wording_ai"])
            part["preference_code"] = frame[f"{stub}_preference"].to_numpy()
            part["llm_guess_code"] = frame[f"{stub}_llm_guess"].to_numpy()
            rows.append(eligible_long_rows(part, frame, survey_set))
    return pd.concat(rows, ignore_index=True)


def _expert_choices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for survey_set, frame in frames.items():
        stubs = sorted({match.group(2) for column in frame.columns if (match := _PREFERENCE.match(str(column)))})
        base = _base_frame(frame, "expert", survey_set)
        for scale_id in stubs:
            stub = f"{survey_set}{scale_id}"
            part = base.copy()
            part["scale_id"] = scale_id
            part["preference_code"] = frame[f"{stub}_preference"].to_numpy()
            part["llm_guess_code"] = frame[f"{stub}_llm_guess"].to_numpy()
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _base_frame(frame: pd.DataFrame, survey: str, survey_set: str) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "survey": survey,
            "respondent_id": [f"{survey}:{survey_set}:{index}" for index in frame.index],
            "set_id": survey_set,
            "trans_cond": frame["trans_cond"].to_numpy(),
        }
    )
    for column in ("age", "gender", "gender_4_TEXT", "highest_degree"):
        if column in frame.columns:
            base[column] = frame[column].to_numpy()
    return base


def _check_keys(frame: pd.DataFrame, keys: list[str], name: str) -> dict:
    duplicated = int(frame.duplicated(keys).sum())
    return {"check": name, "passed": duplicated == 0, "detail": f"duplicate keys {duplicated}; rows {len(frame)}"}


# Workbook attention rows inside SCSS. The citizen export stores set-level
# attention columns and does not repeat these three rows as item columns.
_NOT_EXPORTED = {
    "C07_item39_human",
    "C07_item39_ai",
    "C07_item40_human",
    "C07_item40_ai",
    "C07_item41_human",
    "C07_item41_ai",
}


def _check_coverage(frames: dict[str, pd.DataFrame], items: pd.DataFrame) -> dict:
    expected = expected_item_columns(items)
    observed = observed_item_columns(frames)
    missing = set(expected - observed)
    extra = sorted(observed - expected)
    unexpected = sorted(missing - _NOT_EXPORTED)
    excused = sorted(missing & _NOT_EXPORTED)
    return {
        "check": "citizen item columns match the catalog",
        "passed": not unexpected and not extra,
        "detail": (
            f"expected {len(expected)}, extra {len(extra)}, "
            f"unexplained missing {len(unexpected)}, "
            f"SCSS attention rows not exported as items {len(excused)}"
        ),
    }


def _check_roundtrip(frames: dict[str, pd.DataFrame], long: pd.DataFrame, survey: str) -> dict:
    compared = 0
    mismatched = 0
    prefix = str(long["survey"].iloc[0])
    for survey_set, frame in frames.items():
        part = long.loc[long["set_id"].eq(survey_set)]
        if "source_column" not in part.columns or part.empty:
            continue
        wide = part.pivot(index="respondent_id", columns="source_column", values="raw_value")
        for column in wide.columns:
            original = (mask_ineligible_ocq(frame) if prefix == "citizen" else frame)[column]
            rebuilt = wide[column]
            order = [f"{prefix}:{survey_set}:{index}" for index in frame.index]
            rebuilt = rebuilt.reindex(order)
            left = original.map(_blank).to_numpy()
            right = rebuilt.map(_blank).to_numpy()
            compared += len(left)
            mismatched += int((left != right).sum())
    return {
        "check": f"{survey} long table round-trips to the wide columns",
        "passed": mismatched == 0 and compared > 0,
        "detail": f"cells {compared}, mismatched {mismatched}",
    }


def _check_copied_column(
    frames: dict[str, pd.DataFrame],
    long: pd.DataFrame,
    pattern: re.Pattern[str],
    field_template: str,
    name: str,
) -> dict:
    compared = 0
    mismatched = 0
    for survey_set, frame in frames.items():
        part = long.loc[long["set_id"].eq(survey_set)]
        for column in frame.columns:
            match = pattern.match(str(column))
            if match is None:
                continue
            scale_id = match.group(2)
            version = VERSION[match.group(3)]
            field = field_template.format(version=version)
            rows = part.loc[part["scale_id"].eq(scale_id)].set_index("respondent_id")
            order = [f"{rows['survey'].iloc[0]}:{survey_set}:{index}" for index in frame.index] if len(rows) else []
            if not order:
                continue
            rebuilt = rows.reindex(order)[field].map(_numeric_blank)
            original = mask_ineligible_ocq(frame)[column].map(_numeric_blank)
            compared += len(original)
            mismatched += int((original.to_numpy() != rebuilt.to_numpy()).sum())
    return {
        "check": f"{name} matches the wide table",
        "passed": mismatched == 0 and compared > 0,
        "detail": f"cells {compared}, mismatched {mismatched}",
    }


def _check_codes(responses: pd.DataFrame, allowed: dict[str, set[str]]) -> dict:
    observed = responses.dropna(subset=["raw_value"]).copy()
    bad = []
    for scale_id, group in observed.groupby("scale_id"):
        legal = allowed.get(str(scale_id), set())
        outside = sorted(set(group["raw_value"].astype(str)) - legal)
        if outside:
            bad.append(f"{scale_id}:{','.join(outside[:6])}")
    return {
        "check": "citizen responses use the scale's response codes",
        "passed": not bad,
        "detail": "none outside the label codes" if not bad else "; ".join(bad[:8]),
    }


def _check_rating_range(ratings: pd.DataFrame) -> dict:
    observed = ratings.dropna(subset=["raw_value"])
    outside = sorted(set(observed["raw_value"].astype(str)) - {"1", "2", "3", "4", "5"})
    return {
        "check": "expert scale ratings are on 1-5",
        "passed": not outside,
        "detail": "codes " + (", ".join(outside) if outside else "1-5"),
    }


def _check_flags(problems: pd.DataFrame) -> dict:
    observed = problems.dropna(subset=["raw_value"])
    outside = sorted(set(observed["raw_value"].astype(str)) - {"1"})
    return {
        "check": "expert problem cells are blank or 1",
        "passed": not outside,
        "detail": "other codes " + (", ".join(outside) if outside else "none"),
    }


def _check_order(frames: dict[str, pd.DataFrame], survey: str) -> dict:
    values = pd.concat([frame["trans_cond"] for frame in frames.values()], ignore_index=True)
    outside = sorted(set(values.dropna().astype(str)) - {"human", "ai"})
    missing = int(values.isna().sum())
    return {
        "check": f"{survey} trans_cond is human or ai",
        "passed": not outside and missing == 0,
        "detail": f"missing {missing}, other {outside}",
    }


def _check_counts(frames: dict[str, pd.DataFrame], long: pd.DataFrame, survey: str) -> dict:
    wide_rows = sum(len(frame) for frame in frames.values())
    long_people = int(long["respondent_id"].nunique())
    return {
        "check": f"{survey} respondent count is unchanged",
        "passed": wide_rows == long_people,
        "detail": f"wide {wide_rows}, long {long_people}",
    }


def _sets(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _scale_key(value: object) -> str:
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() else text


def _filled(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() not in {"", "nan", "NA", "<NA>"}


def _numeric_blank(value: object) -> str:
    text = _blank(value)
    if text == "":
        return ""
    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return text
    if float(number).is_integer():
        return str(int(number))
    return str(number)


def _blank(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    return str(value)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def catalog_items(catalog_path: Path, scale_meta_path: Path | None = None) -> pd.DataFrame:
    catalog = load_catalog(catalog_path)
    sets = {}
    if scale_meta_path is not None and Path(scale_meta_path).exists():
        meta = pd.read_csv(scale_meta_path, dtype=str)
        meta["scale_id"] = meta["scale_id"].map(_scale_key)
        sets = dict(zip(meta["scale_id"], meta["survey_set"], strict=True))
    rows = []
    for scale in catalog["scales"]:
        if scale.get("status") != "study":
            continue
        scale_id = str(scale["scale_id"])
        for item in scale["items"]:
            rows.append(
                {
                    "scale_id": scale_id,
                    "scale_name": str(scale.get("short_name") or scale_id),
                    "item_id": str(item["item_id"]).zfill(2),
                    "survey_set": item.get("survey_set") or scale.get("survey_set") or sets.get(scale_id),
                    "reverse_keyed": bool(item.get("reverse_keyed")),
                }
            )
    return pd.DataFrame(rows)
