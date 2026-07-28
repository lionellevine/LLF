#!/usr/bin/env python3
"""Build every raw and calibrated LLF score matrix from the bundled Parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oct_llf.calibration import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/reproduced/matrices"))
    args = parser.parse_args()
    result = summarize(args.data, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
