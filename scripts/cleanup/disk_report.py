"""Disk / memory / project-footprint report (ADR-005 D6) -> ``outputs/system/disk_report.json`` + a short table.

Thin CLI over ``vlm_pipeline.artifacts.storage.measure``. Read-only except for writing the report file.

    uv run python scripts/cleanup/disk_report.py            # write report + print table
    uv run python scripts/cleanup/disk_report.py --no-write # print only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from vlm_pipeline.artifacts.storage import measure
from vlm_pipeline.common.paths import project_root

REPORT_REL = Path("outputs") / "system" / "disk_report.json"


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    Path(fh.name).replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only, do not write the report file")
    args = parser.parse_args(argv)

    root = project_root()
    report = measure(root)
    if not args.no_write:
        write_atomic(root / REPORT_REL, report)
    for section in ("disk", "mem", "project"):
        values = report[section]
        if isinstance(values, dict):
            print(f"{section:8} " + "  ".join(f"{k}={v}" for k, v in values.items()))
    print(f"level    {report['level']}")
    reasons = report["reasons"]
    for reason in reasons if isinstance(reasons, list) else []:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
