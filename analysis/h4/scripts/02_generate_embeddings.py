#!/usr/bin/env python3
"""Encode item texts and cache sentence embeddings."""

import sys

from _bootstrap import with_config

from h4.pipeline import main

if __name__ == "__main__":
    main(with_config(sys.argv[1:], through="embed"))
