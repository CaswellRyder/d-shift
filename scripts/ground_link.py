#!/usr/bin/env python3
"""Run the stdlib-only ground-link bench CLI from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtr.ground_link import main

if __name__ == "__main__":
    raise SystemExit(main())
