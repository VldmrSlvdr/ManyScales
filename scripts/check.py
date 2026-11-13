"""Offline checks for current article outputs, citations, paths, and notices."""
from html.parser import HTMLParser
from pathlib import Path
import argparse
import csv
import subprocess
import hashlib
import re
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="store_true", help="Also check locally attached numerical results")
    args = parser.parse_args()
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    paper = citation["preferred-citation"]
    check(len(paper["authors"]) == 43, "Full paper author list")
    check(paper["doi"] == "10.51094/jxiv.2056", "Paper DOI")
    license_text = (ROOT / "LICENSE").read_text()
    for name in ["Yi-Fei Li", "Jonas R. Kunst", "Yoshihiko Kunisato", "Koji Kosugi"]:
        check(name in license_text, f"Shared MIT copyright: {name}")
    check("Permission is hereby granted" in license_text, "MIT permission text")
    if (ROOT / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        forbidden = {".csv", ".tsv", ".xls", ".xlsx", ".rds", ".rdata", ".npz", ".npy",
                     ".parquet", ".arrow", ".feather", ".zip", ".qsf"}
        present = [p for p in tracked if p and (ROOT / p).is_file()]
        check(not any(Path(p).suffix.lower() in forbidden for p in present), "Study data must not be tracked")
        check(not any(p.startswith(".github/") for p in present), ".github is local only")
        check([p for p in present if Path(p).name == "LICENSE"] == ["LICENSE"], "One shared license")
    check("Version: 0.3.0" in (ROOT / "translation/LLMTranslate/DESCRIPTION").read_text(), "Pinned R package")
    app = ROOT / "translation/LLMTranslate4Hitode/app.R"
    check(hashlib.sha256(app.read_bytes()).hexdigest() ==
          "2d654a4be0dff077abd039c796f7c10ee8d1fd94d72c29a7c4dcdaa7693a9811", "Original Hitode app")
    if args.results:
        required = ["analysis/scale_names.csv", "translation/model_assignments.csv",
                    "figures/h1_ratings.csv", "figures/h3_ratings.csv",
                    "analysis/h4/data/published/specification_curve_primary.csv"]
        missing = [name for name in required if not (ROOT / name).is_file()]
        check(not missing, "Attach the OSF data release before --results: " + ", ".join(missing))
        names = rows("analysis/scale_names.csv")
        check([r["scale_id"] for r in names] == [f"{i:02d}" for i in range(1, 23)], "Scale IDs")
        assignments = rows("translation/model_assignments.csv")
        check(len(assignments) == 22, "Scale assignments")
        for row in assignments:
            check(row["forward_model"] == row["reconciliation_model"] != row["back_model"], "Provider-stage rule")
            check(row["scale_name"] == names[int(row["scale_id"]) - 1]["scale_name"], "Scale labels")
        check(len({(r["forward_model"], r["back_model"]) for r in assignments}) == 6, "Six provider pairs")
        for h, n in [("h1", 66), ("h3", 22)]:
            data = rows(f"figures/{h}_ratings.csv")
            check(len(data) == n, f"{h} plotted rows")
            for row in data:
                if row["overall"] == "FALSE":
                    check(row["scale_name"] == names[int(row["scale_id"]) - 1]["scale_name"], f"{h} approved labels")
                    check(int(row["scale_id"]) != 11, "Primary scale exclusion")
        h3 = rows("figures/h3_ratings.csv")
        overall = [r for r in h3 if r["overall"] == "TRUE"]
        check(len(overall) == 1 and [round(float(overall[0][c]), 3) for c in ["mean", "lower95", "upper95"]]
              == [.094, .071, .116], "H3 overall result")
        scales = [r for r in h3 if r["overall"] == "FALSE"]
        check(sum(float(r["lower95"]) > 0 for r in scales) == 9 and
              sum(float(r["upper95"]) < 0 for r in scales) == 1, "H3 scale decisions")
        specs = rows("analysis/h4/data/published/specification_curve_primary.csv")
        primary = [r for r in specs if r["short_name"] == "mpnet" and r["metric"] == "cosine" and r["comparison"] == "back_translation"]
        check(len(primary) == 1 and round(float(primary[0]["estimate"]), 3) == -.041, "H4 estimate")
        check(primary[0]["n_items"] == "486" and primary[0]["n_scales"] == "21", "H4 primary sample")
        cultural = [r for r in rows("analysis/h4/data/published/raw_rating_ordinal_n1859.csv")
                    if r["audience"] == "expert" and r["outcome"] == "cultural_validity" and r["term"] == "x:sim"]
        check(len(cultural) == 1 and float(cultural[0]["ci_low"]) > 0, "Corrected cultural interaction")
        for ext in ["png", "pdf", "svg"]:
            check((ROOT / f"figures/h4_similarity.{ext}").read_bytes() ==
                  (ROOT / f"analysis/h4/figures/mean_item_gap_ja_ci.{ext}").read_bytes(), "H4 figure synchronization")
    # Maintained docs only; bundled upstream package documentation stays unchanged.
    docs = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
    docs += [ROOT / p for p in ["analysis/README.md", "analysis/ratings/README.md",
             "analysis/ratings/supplementary/README.md", "analysis/h4/README.md",
             "reports/README.md", "figures/README.md"]]
    docs += [p for p in (ROOT / "translation").rglob("README.md") if "LLMTranslate" not in p.parts]
    for doc in docs:
        for target in re.findall(r"\]\(([^\n)]+)\)", doc.read_text()):
            target = target.strip("<>").split("#")[0]
            if target and not re.match(r"[a-z]+:", target):
                check((doc.parent / unquote(target)).exists(), f"Broken link: {doc.relative_to(ROOT)} → {target}")
    class ReportLinks(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                target = dict(attrs).get("href", "")
                url = urlsplit(target)
                if url.path and not url.scheme and not url.netloc:
                    check((self.report.parent / unquote(url.path)).exists(),
                          f"Broken report link: {self.report.name} → {target}")
    for report in (ROOT / "reports").glob("*.html"):
        parser = ReportLinks()
        parser.report = report
        parser.feed(report.read_text())
    secret = re.compile(r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|AIza[0-9A-Za-z_-]{30,})")
    for base in ["scripts", "translation/LLMTranslate4Hitode"]:
        for path in (ROOT / base).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".r", ".py", ".yaml", ".yml", ".json"}:
                check(not secret.search(path.read_text()), f"Potential credential in {path.relative_to(ROOT)}")
    print("PASS: code-only repository, citation, shared MIT notice, and report links." +
          (" Local H1/H3/H4 result checks passed." if args.results else ""))


if __name__ == "__main__":
    main()
