"""Command-line interface for the OCT LLF package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .calibration import summarize
from .scoring import score_formulas
from .validation import validate_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oct-llf", description="OCT log-probability scoring and calibration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="validate a text-free data bundle")
    validate.add_argument("data_dir", type=Path)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--scan-repository", type=Path)
    summary = commands.add_parser("summarize", help="write raw and calibrated matrices")
    summary.add_argument("data_dir", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    formulas = commands.add_parser("score-formulas", help="compute all pair-score formulas")
    formulas.add_argument("adapter_chosen", type=float)
    formulas.add_argument("base_chosen", type=float)
    formulas.add_argument("chosen_length", type=int)
    formulas.add_argument("adapter_rejected", type=float)
    formulas.add_argument("base_rejected", type=float)
    formulas.add_argument("rejected_length", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-data":
        result = validate_bundle(args.data_dir, args.manifest, args.scan_repository)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "summarize":
        result = summarize(args.data_dir, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "score-formulas":
        result = score_formulas(
            args.adapter_chosen,
            args.base_chosen,
            args.chosen_length,
            args.adapter_rejected,
            args.base_rejected,
            args.rejected_length,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
