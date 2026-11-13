#!/usr/bin/env python3
"""Extract scale introductions, items, labels, and metadata from the workbook."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import ROOT

from h4.catalog import extract_workbook, study_items_from_catalog, write_catalog, write_preanalysis_tables
from h4.preprocess import exclude_flagged_scales
from h4.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "analysis.yaml"),
        help="Analysis configuration. Workbook and catalog paths come from paths.workbook and paths.scales.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    workbook = config.paths["workbook"]
    destination = config.paths["scales"]
    catalog = extract_workbook(workbook)
    write_catalog(catalog, destination)
    write_preanalysis_tables(catalog, config.paths["processed"].parent)
    full = study_items_from_catalog(catalog)
    primary, exclusion = exclude_flagged_scales(full)
    processed_dir = config.paths["processed"].parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    with_tivs = processed_dir / "items_with_tivs.parquet"
    full.to_parquet(with_tivs, index=False)
    log_path = processed_dir / "exclusion_log.csv"
    exclusion.to_csv(log_path, index=False)
    print(f"Wrote {with_tivs} ({len(full)} items, robustness, TIVS retained)")
    print(f"Wrote {log_path}")
    print(exclusion.to_string(index=False))
    for table_path in dict.fromkeys((config.paths["items"], config.paths["processed"])):
        if table_path.suffix.lower() != ".parquet":
            continue
        table_path.parent.mkdir(parents=True, exist_ok=True)
        primary.to_parquet(table_path, index=False)
        print(f"Wrote {table_path} ({len(primary)} items, primary, TIVS excluded)")
    study = [scale for scale in catalog["scales"] if scale["status"] == "study"]
    n_items = sum(len(scale["items"]) for scale in study)
    unmatched = [scale["sheet"] for scale in study if not scale["registry_matched"]]
    print(f"Wrote {destination}")
    print(f"Study scales: {len(study)}; study items: {n_items}")
    print(f"Catalog scales: {catalog['n_scales']}")
    if unmatched:
        print("Study sheets without an overview row: " + ", ".join(unmatched))


if __name__ == "__main__":
    main(sys.argv[1:])
