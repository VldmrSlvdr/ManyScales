#!/usr/bin/env python3
"""Fit the preregistered hierarchical model for H4."""

import sys

from _bootstrap import with_config

from h4.pipeline import main

if __name__ == "__main__":
    main(with_config(sys.argv[1:], through="confirmatory"))
