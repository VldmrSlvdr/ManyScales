#!/usr/bin/env python3
"""Exploratory token geometry and distributional distance. This does not evaluate H4."""

import sys

from _bootstrap import with_config

from h4.exploratory.runner import main

if __name__ == "__main__":
    main(["--part", "geometry", *with_config(sys.argv[1:])])
