#!/usr/bin/env python3
"""Bootstrap, alternative specifications, and the direct cross-lingual comparison."""

import sys

from _bootstrap import with_config

from h4.pipeline import main

if __name__ == "__main__":
    main(with_config(sys.argv[1:], through="robustness"))
