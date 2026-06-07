#!/usr/bin/env python3
"""Collect PhysGaussian ablation metrics into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_ablation import collect_metrics, write_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "ablation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    rows = collect_metrics(output_root)
    csv_path, md_path = write_summary(output_root, rows)
    print(f"[collect] runs={len(rows)}")
    print(f"[collect] {csv_path}")
    print(f"[collect] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
