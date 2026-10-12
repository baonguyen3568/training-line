"""Node-cache eviction (P005 §6 R1). DRY-RUN by default; this is also the admin path for DISK-002 approvals.

Thin CLI over ``vlm_pipeline.artifacts.cache_retention``. Protected entries (trained models, unreadable run.json)
are only evicted when listed explicitly with ``--items``.

    uv run python scripts/cleanup/cache.py                                  # list cache, LRU first
    uv run python scripts/cleanup/cache.py --auto --apply --by bao-cleanup  # evict down to the cap
    uv run python scripts/cleanup/cache.py --items k1,k2 --apply --by <admin> [--approved-by <admin>]
"""

from __future__ import annotations

import argparse
import sys

from vlm_pipeline.artifacts.cache_retention import evict, list_cache, plan_cache
from vlm_pipeline.artifacts.index import ArtifactIndex, default_index_path
from vlm_pipeline.common.paths import GIB, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-gb", type=float, default=200.0, help="cache cap for --auto (default 200)")
    parser.add_argument("--auto", action="store_true", help="select LRU entries down to the cap (never protected)")
    parser.add_argument("--items", help="comma-separated cache keys (may include protected entries)")
    parser.add_argument("--apply", action="store_true", help="actually evict (default is dry-run)")
    parser.add_argument("--by", help="who runs --apply (required with --apply)")
    parser.add_argument("--approved-by", help="approver recorded in the log (DISK-002 approvals)")
    args = parser.parse_args(argv)

    root = project_root()
    index = ArtifactIndex(default_index_path(root))
    for item, reason in list_cache(root, index):
        print(f"{item.reclaim_bytes / GIB:9.3f} GiB  {item.cache_key}  run={item.run_id}  {reason or 'auto-evictable'}")
    plan = plan_cache(root, index, cap_gb=args.cap_gb)
    print(f"auto plan: {len(plan.evict)} entr(y/ies), {plan.reclaim_bytes / GIB:.3f} GiB; shortfall "
          f"{plan.shortfall_bytes / GIB:.3f} GiB")  # fmt: skip
    if not args.apply:
        return 0
    if not args.by:
        parser.error("--apply requires --by <name>")
    if args.auto == bool(args.items):
        parser.error("use exactly one of --auto or --items")
    keys = [i.cache_key for i in plan.evict] if args.auto else args.items.split(",")
    confirmed = set() if args.auto else set(keys)  # explicit --items is the admin's per-item choice
    result = evict(
        root, index, keys, by=args.by, manual=not args.auto, confirmed=confirmed, approved_by=args.approved_by
    )
    for item in result.evicted:
        print(f"evicted  {item.cache_key}  {item.reclaim_bytes / GIB:.3f} GiB")
    for key, why in result.skipped:
        print(f"skipped  {key}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
