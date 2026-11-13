"""Build the citizen analysis tables that do not apply straight-lining.

The analysis sample is finished citizens who pass the attention check and
the speeding check. Straight-lining is omitted because item order was
randomized and the presented order was not recorded. The earlier three-rule
tables in data/survey/3_processed and data/survey/4_analysis stay in place
as the sensitivity cut. Confirmatory cosine scoring does not read either cut.
"""

from pathlib import Path
import shutil

import pandas as pd

from h4.survey.datasets import (
    analysis_rows_citizen,
    analysis_rows_citizen_two_rule,
    citizen_sets,
)
from h4.survey.reshape import _citizen_judgments, load_processed
from h4.survey.eligibility import ocq_nonconsumer

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ROOT / "data/survey/1_flags/ManyScales_一般_analysis_flags.tsv"
PROCESSED = ROOT / "data/survey/3_processed_n1859"
ANALYSIS = ROOT / "data/survey/4_analysis_n1859"
AUDIT = ROOT / "results/exploratory_n1859/screen_audit.csv"


def _numeric_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("analysis_target", "excl1", "excl2", "excl3"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def main() -> None:
    flagged = _numeric_flags(pd.read_csv(FLAGS, sep="\t", dtype=str, low_memory=False))
    kept = analysis_rows_citizen_two_rule(flagged)
    sensitivity = analysis_rows_citizen(flagged)
    if len(kept) != 1859:
        raise RuntimeError(f"expected 1859 citizen analysis rows, found {len(kept)}")
    counts = citizen_sets(kept, PROCESSED, "ManyScales_市民_processed_")
    old_processed = ROOT / "data/survey/3_processed"
    for survey_set in "ABCDEF":
        source = old_processed / f"ManyScales_専門家_processed_{survey_set}.csv"
        shutil.copy2(source, PROCESSED / source.name)
    frames = load_processed(PROCESSED, "ManyScales_市民_processed_")
    judgments = _citizen_judgments(frames)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    judgments.to_parquet(ANALYSIS / "citizen_scale_judgment.parquet", index=False)
    people = int(judgments["respondent_id"].nunique())
    if people != 1859:
        raise RuntimeError(f"expected 1859 respondents in the long table, found {people}")
    audit = pd.DataFrame(
        [
            {"cut": "citizen_two_rule", "n_respondents": int(len(kept)), "note": "analysis_target, attention, speeding"},
            {"cut": "citizen_three_rule", "n_respondents": int(len(sensitivity)), "note": "sensitivity; also drops straight-lining"},
            {
                "cut": "citizen_removed_only_by_straightline",
                "n_respondents": int(len(kept) - len(sensitivity)),
                "note": "present in the analysis sample and absent from the sensitivity sample",
            },
            {"cut": "speeding_flags_in_export", "n_respondents": int(flagged["excl2"].eq(1).sum()), "note": "none in this export"},
            {
                "cut": "ocq_two_rule_eligible",
                "n_respondents": int((kept["set_id"].eq("C") & ~ocq_nonconsumer(kept)).sum()),
                "note": f"OCQ only; excludes {int(ocq_nonconsumer(kept).sum())} non-consumers; other scales retained",
            },
            {
                "cut": "ocq_three_rule_eligible",
                "n_respondents": int((sensitivity["set_id"].eq("C") & ~ocq_nonconsumer(sensitivity)).sum()),
                "note": f"OCQ sensitivity only; excludes {int(ocq_nonconsumer(sensitivity).sum())} non-consumers",
            },
        ]
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT, index=False)
    print(audit.to_string(index=False))
    print("set counts", counts)
    print("judgment rows", len(judgments), "people", people, "scales", judgments["scale_id"].nunique())


if __name__ == "__main__":
    main()
