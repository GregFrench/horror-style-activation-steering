#!/usr/bin/env python
"""Compare baseline and steered outputs with simple descriptive metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight comparison for baseline and steered generations."
    )
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--steered-file", type=Path, default=None)
    parser.add_argument("--baseline-text", default=None)
    parser.add_argument("--steered-text", default=None)
    return parser.parse_args()


def _load_text(file_path: Path | None, direct_text: str | None, label: str) -> str:
    if file_path is not None:
        return file_path.read_text(encoding="utf-8")
    if direct_text is not None:
        return direct_text
    raise SystemExit(f"Provide --{label}-file or --{label}-text.")


def main() -> None:
    args = parse_args()

    from src.evaluate import format_comparison

    baseline = _load_text(args.baseline_file, args.baseline_text, "baseline")
    steered = _load_text(args.steered_file, args.steered_text, "steered")
    print(format_comparison(baseline, steered))


if __name__ == "__main__":
    main()
