"""Remove gate-run git worktrees under ``.worktrees/`` (ADR-002 A1). DRY-RUN by default.

Thin CLI over ``vlm_pipeline.artifacts.worktrees``: uses ``git worktree remove`` (never --force) + ``prune``.

    uv run python scripts/cleanup/worktrees.py                       # dry-run
    uv run python scripts/cleanup/worktrees.py --apply --by bao-cleanup [--run ID ...]
"""

from __future__ import annotations

import argparse
import sys

from vlm_pipeline.artifacts.worktrees import apply, plan
from vlm_pipeline.common.gpu_lock import default_lock_path
from vlm_pipeline.common.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", dest="runs", help="limit to this run id (repeatable)")
    parser.add_argument("--apply", action="store_true", help="actually remove (default is dry-run)")
    parser.add_argument("--by", help="who runs --apply (required with --apply)")
    args = parser.parse_args(argv)

    root = project_root()
    lock_path = default_lock_path()
    items = plan(root, run_ids=args.runs, lock_path=lock_path)
    for item in items:
        print(f"{item.action:6} {item.run_id:45} status={item.status:11} {item.blocked_by or item.reason}")
    if not items:
        print("no gate-run worktrees under .worktrees/")
    if not args.apply:
        return 0
    if not args.by:
        parser.error("--apply requires --by <name>")
    result = apply(root, [i.run_id for i in items if i.action == "remove"], by=args.by, lock_path=lock_path)
    for item in result.removed:
        print(f"removed  {item.run_id}")
    for run_id, msg in result.failed:
        print(f"FAILED   {run_id}: {msg}", file=sys.stderr)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
