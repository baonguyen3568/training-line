"""Seed demo run directories (all 8 RunStatus values) for frontend/tester development.

    uv run python scripts/dev/seed_runs.py --root /tmp/vlm-demo    # -> /tmp/vlm-demo/outputs/runs/demo-*
Refuses a non-empty ``<root>/outputs/runs``; never seed into the real project outputs by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlm_pipeline.artifacts.testing import seed_demo_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="directory that will contain outputs/runs/")
    args = parser.parse_args(argv)
    try:
        dirs = seed_demo_runs(args.root / "outputs" / "runs")
    except FileExistsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    for d in dirs:
        print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
