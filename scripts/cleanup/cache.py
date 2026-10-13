"""Node-cache eviction (P005 §6 R1). DRY-RUN by default; this is also the admin path for DISK-002 approvals.

Thin CLI over ``vlm_pipeline.artifacts.cache_retention``. Protected entries (trained models, unreadable run.json)
are only evicted when listed explicitly with ``--items``.

    uv run python scripts/cleanup/cache.py                                  # list cache, LRU first
    uv run python scripts/cleanup/cache.py --auto --apply --by bao-cleanup  # evict down to the cap
    uv run python scripts/cleanup/cache.py --items k1,k2 --apply --by <admin> [--approved-by <admin>]
    uv run python scripts/cleanup/cache.py --release-lease <graph_run_id> --apply --by <admin>

``--release-lease``: drop the cache leases of a graph run whose process is gone but could not be judged at app
start (it ran in another pid namespace, e.g. a host script next to the app in compose). The dry-run lists every lease
and its owner; a live owner in this pid namespace or a held GPU lock refuses, an owner in another namespace needs
``--confirm <graph run id>``. A lease protects the inputs of a running graph from eviction.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from vlm_pipeline.artifacts.cache_retention import evict, list_cache, plan_cache
from vlm_pipeline.artifacts.index import ArtifactIndex, default_index_path, owner_is_alive, pid_namespace
from vlm_pipeline.artifacts.retention import CLEANUP_LOG_REL, append_jsonl, cleanup_lock
from vlm_pipeline.common.gpu_lock import default_lock_path, lock_present
from vlm_pipeline.common.paths import GIB, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-gb", type=float, default=200.0, help="cache cap for --auto (default 200)")
    parser.add_argument("--auto", action="store_true", help="select LRU entries down to the cap (never protected)")
    parser.add_argument("--items", help="comma-separated cache keys (may include protected entries)")
    parser.add_argument("--apply", action="store_true", help="actually evict (default is dry-run)")
    parser.add_argument("--by", help="who runs --apply (required with --apply)")
    parser.add_argument("--approved-by", help="approver recorded in the log (DISK-002 approvals)")
    parser.add_argument("--release-lease", metavar="GRAPH_RUN_ID", help="drop the leases of a graph run that is over")
    parser.add_argument("--confirm", help="with --release-lease: the graph run id again (owner not checkable)")
    args = parser.parse_args(argv)

    root = project_root()
    index = ArtifactIndex(default_index_path(root))
    if args.release_lease:
        return _release_lease(parser, root, index, args)
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
    with cleanup_lock(root) as held:
        if not held:
            print("refused: another cleanup pass is running (outputs/system/retention.lock)", file=sys.stderr)
            return 1
        result = evict(
            root, index, keys, by=args.by, manual=not args.auto, confirmed=confirmed, approved_by=args.approved_by
        )
    for item in result.evicted:
        print(f"evicted  {item.cache_key}  {item.reclaim_bytes / GIB:.3f} GiB")
    for key, why in result.skipped:
        print(f"skipped  {key}: {why}")
    return 0


GRAPH_RUN_ID = re.compile(r"g-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", re.ASCII)  # executor / app / run_workflow.py


def _release_lease(parser: argparse.ArgumentParser, root: Path, index: ArtifactIndex, args: argparse.Namespace) -> int:
    """Admin path for leases no app start could judge (owner in another pid namespace). Refuses a live owner."""
    run_id = args.release_lease
    if not GRAPH_RUN_ID.fullmatch(run_id):
        parser.error("--release-lease takes a graph run id: g-YYYYMMDD-HHMMSS-<6 hex>")
    leases = index.leases_of(run_id)
    here = pid_namespace()
    unjudged = False
    for key, owner, since in leases:
        who = "legacy (no owner recorded)" if owner is None else f"pid {owner.pid} start {owner.start} ns {owner.ns}"
        print(f"lease  {key}  since {time.strftime('%Y-%m-%d %H:%M', time.localtime(since))}  {who}")
        if owner is not None and owner.ns is not None and owner.ns != here:
            unjudged = True
        elif owner is not None and owner_is_alive(owner):
            print(f"refused: the graph run's process (pid {owner.pid}) is still running", file=sys.stderr)
            return 1
    if not leases:
        print(f"graph run {run_id!r} holds no lease")
        return 0
    if not args.apply:
        print(f"dry-run: {len(leases)} lease(s) would be released. Re-run with --apply --by <name>.", file=sys.stderr)
        return 0
    if not args.by:
        parser.error("--apply requires --by <name>")
    if unjudged and args.confirm != run_id:
        parser.error("an owner runs in another pid namespace (cannot be checked): pass --confirm <graph run id>")
    if lock_present(default_lock_path()):
        print(
            "refused: the GPU lock is held (a graph may still use these inputs); retry when it is free", file=sys.stderr
        )
        return 1
    with cleanup_lock(root) as held:
        if not held:
            print("refused: another cleanup pass is running (outputs/system/retention.lock)", file=sys.stderr)
            return 1
        index.release_run(run_id)
        now = time.time()
        append_jsonl(
            root / CLEANUP_LOG_REL,
            {"event": "lease_released", "run_id": run_id, "cache_keys": [k for k, _, _ in leases],
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), "ts_epoch": now, "by": args.by,
             "manual": True, "approved_by": args.approved_by},
        )  # fmt: skip
    print(f"released {len(leases)} cache lease(s) of graph run {run_id!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
