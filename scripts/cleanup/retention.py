"""Checkpoint retention (ADR-005 D3/D7). DRY-RUN by default; nothing is deleted without ``--apply``.

Thin CLI over ``vlm_pipeline.artifacts.retention``. Only ``outputs/runs/*/checkpoints/epoch_*`` are candidates.

    uv run python scripts/cleanup/retention.py                      # dry-run table for all runs
    uv run python scripts/cleanup/retention.py --run <id> --json    # dry-run plan JSON for one run
    uv run python scripts/cleanup/retention.py --apply --by bao-cleanup [--approved-by bao-pm]
    uv run python scripts/cleanup/retention.py --items <id1,id2> --apply --by ... --approved-by ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from vlm_pipeline.artifacts.retention import apply, plan
from vlm_pipeline.common.gpu_lock import default_lock_path
from vlm_pipeline.common.paths import GIB, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", dest="runs", help="limit to this run id (repeatable)")
    parser.add_argument("--items", help="comma-separated item ids to apply (default: every actionable item)")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument("--apply", action="store_true", help="actually delete/strip (default is dry-run)")
    parser.add_argument("--by", help="who runs --apply (required with --apply)")
    parser.add_argument("--approved-by", help="PM/security approval, required above the approval threshold")
    args = parser.parse_args(argv)

    root = project_root()
    lock_path = default_lock_path()
    current = plan(root, run_ids=args.runs, lock_path=lock_path)
    if args.json:
        print(json.dumps(current.as_dict(), ensure_ascii=False, indent=2))
    else:
        for run in current.runs:
            print(f"== {run.run_id}  status={run.status}{'  KEEP' if run.keep else ''}")
            for item in run.items:
                roles = ",".join(item.roles) or "-"
                why = item.blocked_by or item.reason
                print(f"   {item.action:11} {item.reclaim_bytes / GIB:8.3f} GiB  {item.path}  [{roles}]  {why}")
        total = current.total_reclaim_bytes / GIB
        print(f"total reclaimable: {total:.3f} GiB" + ("  (needs --approved-by)" if current.needs_approval else ""))

    if not args.apply:
        if current.actionable:
            print("dry-run: nothing deleted. Re-run with --apply --by <name> to execute.", file=sys.stderr)
        return 0
    if not args.by:
        parser.error("--apply requires --by <name>")

    ids = args.items.split(",") if args.items else [i.item_id for i in current.actionable]
    try:
        result = apply(root, ids, by=args.by, approved_by=args.approved_by, lock_path=lock_path)
    except PermissionError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    for item in result.done:
        print(f"done     {item.action:11} {item.reclaim_bytes / GIB:8.3f} GiB  {item.item_id}")
    for item_id, why in result.skipped:
        print(f"skipped  {item_id}: {why}")
    freed = result.freed_bytes / GIB
    if result.done:
        log = root / ".team" / "log" / f"{time.strftime('%Y-%m-%d')}.md"
        with log.open("a", encoding="utf-8") as fh:
            approval = f", approved by {args.approved_by}" if args.approved_by else ""
            fh.write(
                f"{time.strftime('%H:%M')} [{args.by}] retention --apply: {len(result.done)} item(s), "
                f"freed {freed:.2f} GiB{approval}\n"
            )
    print(f"freed {freed:.3f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
