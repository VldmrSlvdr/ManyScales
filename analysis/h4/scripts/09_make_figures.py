#!/usr/bin/env python3
"""Draw figures from tables that earlier stages have already written."""

import sys
from pathlib import Path

from _bootstrap import ROOT, with_config

from h4.plotting import make_figures

if __name__ == "__main__":
    args = with_config(sys.argv[1:])
    results = ROOT / "results"
    if "--results" in args:
        results = Path(args[args.index("--results") + 1])
    written = make_figures(results)
    print(f"Wrote {len(written)} figure files to {results / 'figures'}")
