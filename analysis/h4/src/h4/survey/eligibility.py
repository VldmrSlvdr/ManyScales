"""Scale-specific respondent eligibility, separate from study-wide screening."""

from __future__ import annotations

import pandas as pd


def ocq_nonconsumer(frame: pd.DataFrame) -> pd.Series:
    """Code 6 means no chocolate in the past month; missing is not code 6."""
    if "C08_chocolate_frequency" not in frame:
        return pd.Series(False, index=frame.index)
    mask = pd.to_numeric(frame["C08_chocolate_frequency"], errors="coerce").eq(6)
    if "set_id" in frame:
        mask &= frame["set_id"].eq("C")
    return mask


def mask_ineligible_ocq(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask OCQ responses for non-consumers, retaining frequency and other scales.

    Preserve row order and index so positional respondent IDs remain stable.
    The raw frequency is retained as an audit/eligibility variable.
    """
    out = frame.copy()
    columns = [c for c in out if c.startswith("C08_") and c != "C08_chocolate_frequency"]
    out.loc[ocq_nonconsumer(out), columns] = pd.NA
    return out


def eligible_long_rows(long: pd.DataFrame, wide: pd.DataFrame, survey_set: str) -> pd.DataFrame:
    """Remove only scale-08 rows, matching stable wide-row respondent IDs."""
    if survey_set != "C" or "scale_id" not in long:
        return long
    excluded = {f"citizen:C:{i}" for i in wide.index[ocq_nonconsumer(wide)]}
    return long.loc[~(long["scale_id"].astype(str).str.zfill(2).eq("08") & long["respondent_id"].isin(excluded))].copy()
