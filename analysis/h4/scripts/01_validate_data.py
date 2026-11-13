#!/usr/bin/env python3
"""Validate the canonical item table. This stage does not estimate H4."""

import sys

from _bootstrap import with_config

from h4.pipeline import main

if __name__ == "__main__":
    main(with_config(sys.argv[1:], through="validate"))
