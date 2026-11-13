#!/usr/bin/env python3
"""Score cached embeddings. This stage does not fit the H4 model."""

import sys

from _bootstrap import with_config

from h4.pipeline import main

if __name__ == "__main__":
    main(with_config(sys.argv[1:], through="similarity"))
