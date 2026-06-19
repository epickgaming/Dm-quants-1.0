#!/usr/bin/env python3
"""Thin wrapper to run the anchored walk-forward backtest.

Examples
--------
    # offline, reproducible synthetic run (no network):
    python scripts/run_backtest.py --synthetic

    # big-data sweep over cached real H1 history:
    python scripts/run_backtest.py --no-fetch

This simply forwards to `python -m propalgo.cli backtest`.
"""

import sys

from propalgo.cli import main

if __name__ == "__main__":
    main(["backtest"] + sys.argv[1:])
