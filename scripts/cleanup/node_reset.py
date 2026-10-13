"""Reset one graph step (G-4): delete every attempt dir of a node and give it a new generation. DRY-RUN by default.

Thin CLI over ``vlm_pipeline.artifacts.node_reset`` for an admin (the storage panel answers DISK-002 with this command
on a shared host above the approval threshold).

    uv run python scripts/cleanup/node_reset.py --node n3-f03d17784e408456                     # dry-run
    uv run python scripts/cleanup/node_reset.py --node <node> --confirm <node> --apply --by <name> [--approved-by <n>]
"""

from __future__ import annotations

import argparse
import sys

from vlm_pipeline.artifacts.index import ArtifactIndex, default_index_path
from vlm_pipeline.artifacts.node_reset import PERMANENT_LOSS, apply_reset, plan_reset
from vlm_pipeline.artifacts.retention import cleanup_lock
from vlm_pipeline.common.gpu_lock import default_lock_path
from vlm_pipeline.common.paths import GIB, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", required=True, help="<node id>-<16 hex>, as in outputs/graph/runs")
    parser.add_argument("--apply", action="store_true", help="actually delete (default is dry-run)")
    parser.add_argument("--by", help="who runs --apply (required with --apply)")
    parser.add_argument("--confirm", help="the node name again: required when an attempt holds checkpoints")
    parser.add_argument("--approved-by", help="PM/security approval on a shared host")
    args = parser.parse_args(argv)

    root = project_root()
    index = ArtifactIndex(default_index_path(root))
    lock_path = default_lock_path()
    try:
        plan = plan_reset(root, index, args.node, lock_path=lock_path)
    except ValueError:
        parser.error("--node must be <node id>-<16 hex>")
    for attempt in plan.attempts:
        ckpt = "  checkpoints" if attempt.has_checkpoint else ""
        print(f"   {attempt.status:11} {attempt.reclaim_bytes / GIB:8.3f} GiB  {attempt.name}{ckpt}")
    print(f"total {plan.reclaim_bytes / GIB:.3f} GiB, next generation {plan.generation + 1}")
    if plan.blocked_by:
        print(f"refused: {plan.blocked_by} {plan.blocked_params}", file=sys.stderr)
        return 1
    if not args.apply:
        print("dry-run: nothing deleted. Re-run with --apply --by <name>.", file=sys.stderr)
        return 0
    if not args.by:
        parser.error("--apply requires --by <name>")
    if plan.consequence_code == PERMANENT_LOSS and args.confirm != args.node:
        parser.error("an attempt holds checkpoints (permanent loss): pass --confirm <node> to delete them on purpose")
    with cleanup_lock(root) as held:
        if not held:
            print("refused: another cleanup pass is running (outputs/system/retention.lock)", file=sys.stderr)
            return 1
        try:
            result = apply_reset(
                root, index, args.node, by=args.by, expected=plan, approved_by=args.approved_by, lock_path=lock_path
            )
        except PermissionError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
    print(f"deleted {len(result.deleted)} attempt dir(s), freed {result.freed_bytes / GIB:.3f} GiB, "
          f"generation {result.generation}" + (f", FAILED: {result.failed}" if result.failed else ""))  # fmt: skip
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
