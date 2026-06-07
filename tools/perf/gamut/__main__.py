"""Entry point: `python -m gamut …` (run from tools/perf, or with tools/perf on PYTHONPATH)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
