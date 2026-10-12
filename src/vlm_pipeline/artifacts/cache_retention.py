"""Node-cache LRU eviction (P005 §6 R1, P009 §5). DRY-RUN by default via ``plan_cache``.

The cache *is* the run directories of cached nodes; ``ArtifactIndex`` (outputs/app.db) only points at them. Evicting
an entry frees its heavy bytes and keeps its history:

- deleted: ``checkpoints/``, ``engines/`` and every output path that lies inside the run dir but outside ``export/``
- kept: run.json (status -> ``evicted``), *.jsonl, logs/, export/ (user-named artifacts), anything outside the run dir
  (e.g. the shared HF cache), and pinned entries / parents still needed at load time (the index refuses those)

Trained models are expensive to recompute (PM decision, DECISIONS 2026-10-12): an entry flagged ``trained_model`` in
the index (set by the executor from node metadata) or whose run.json says ``stage == "train"`` is never chosen by the
automatic plan, exported or not; an unreadable run.json is protected too (MANIFEST_UNREADABLE).
It is listed in ``CachePlan.protected`` (TRAINED_MODEL) and only ``evict(..., manual=True)`` (the user's explicit
choice in the storage panel, after a dry-run) removes it; old non-LATEST/best checkpoints of training runs are
handled by checkpoint retention (D3), not here. When the cap cannot be met with what is evictable,
``CachePlan.shortfall_bytes`` > 0 and the caller reports a DISK error to the user instead of deleting more.

Eviction is 3-step so it never races the executor: ``begin_evict`` -> delete -> ``finish_evict`` (a failed delete
marks the entry ``invalid`` so it is never hit again). Least recently used entries go first until the cache is under
``cap_gb`` (and at least ``need_free_gb`` is freed when the project footprint hit its warn level).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from vlm_pipeline.artifacts.index import ArtifactIndex, CacheEntry
from vlm_pipeline.artifacts.retention import RUNS_REL, check_project_root
from vlm_pipeline.artifacts.storage import du_bytes
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import RunStatus, read_manifest, write_manifest

HEAVY_SUBDIRS = ("checkpoints", "engines")
KEPT_SUBDIRS = ("export", "logs")
DEFAULT_CAP_GB = 200.0  # ADR-005: min(200 GB, 10 % of free disk at install); the app passes the configured value


@dataclass(frozen=True)
class CacheItem:
    cache_key: str
    run_id: str
    node_type: str
    reclaim_bytes: int
    targets: tuple[str, ...]  # paths relative to the run dir that eviction deletes
    last_used: float

    def as_dict(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "run_id": self.run_id,
            "node_type": self.node_type,
            "reclaim_gb": round(self.reclaim_bytes / GIB, 3),
            "targets": list(self.targets),
            "last_used": self.last_used,
            "event_type": "cache_evicted",
        }


@dataclass(frozen=True)
class CachePlan:
    cache_bytes: int  # bytes held by every evictable entry
    cap_gb: float
    evict: tuple[CacheItem, ...]  # LRU first
    protected: tuple[tuple[CacheItem, str], ...] = ()  # (item, reason) never chosen automatically
    shortfall_bytes: int = 0  # still over cap / short of need_free after evicting everything allowed -> DISK error

    @property
    def reclaim_bytes(self) -> int:
        return sum(i.reclaim_bytes for i in self.evict)


@dataclass
class CacheResult:
    evicted: list[CacheItem] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def freed_bytes(self) -> int:
        return sum(i.reclaim_bytes for i in self.evicted)


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _targets(root: Path, entry: CacheEntry) -> tuple[str, ...]:
    """Paths (relative to the run dir) eviction deletes; symlinks and anything outside the run dir are skipped."""
    run_dir = (root / RUNS_REL / entry.run_id).resolve()
    rels: set[str] = {d for d in HEAVY_SUBDIRS if (run_dir / d).is_dir() and not (run_dir / d).is_symlink()}
    for ref in entry.outputs.values():
        out = Path(ref.path)
        if out.is_symlink() or not out.exists():
            continue
        resolved = out.resolve()
        if not _inside(resolved, run_dir) or resolved == run_dir:
            continue
        rel = resolved.relative_to(run_dir)
        if rel.parts[0] in KEPT_SUBDIRS or (len(rel.parts) == 1 and rel.suffix == ".jsonl") or rel.name == "run.json":
            continue
        rels.add(rel.as_posix())
    # drop paths already covered by a parent target
    return tuple(sorted(r for r in rels if not any(r != o and r.startswith(o + "/") for o in rels)))


def _item(root: Path, entry: CacheEntry) -> CacheItem:
    run_dir = root / RUNS_REL / entry.run_id
    targets = _targets(root, entry)
    return CacheItem(
        cache_key=entry.cache_key,
        run_id=entry.run_id,
        node_type=entry.node_type,
        reclaim_bytes=sum(du_bytes(run_dir / t) for t in targets),
        targets=targets,
        last_used=entry.last_hit_at or entry.completed_at,
    )


TRAINED_MODEL = "TRAINED_MODEL"
MANIFEST_UNREADABLE = "MANIFEST_UNREADABLE"


def protection_reason(root: Path, entry: CacheEntry) -> str | None:
    """Why automatic eviction must skip ``entry`` (None = may be evicted automatically). Fails closed.

    Trained-ness comes from the index flag the executor sets from node registry metadata (never from matching the
    namespaced ``node_type`` string) or from run.json ``stage``; an unreadable run.json also protects the entry.
    """
    if entry.trained_model:
        return TRAINED_MODEL
    try:
        stage = read_manifest(root / RUNS_REL / entry.run_id).stage
    except (OSError, ValueError):
        return MANIFEST_UNREADABLE
    return TRAINED_MODEL if stage == "train" else None


def plan_cache(
    root: Path, index: ArtifactIndex, *, cap_gb: float = DEFAULT_CAP_GB, need_free_gb: float = 0.0
) -> CachePlan:
    """Read-only: LRU entries to evict so the cache fits ``cap_gb`` and at least ``need_free_gb`` is freed."""
    items: list[CacheItem] = []
    protected: list[tuple[CacheItem, str]] = []
    for entry in index.evictable():  # already LRU ordered
        item = _item(root, entry)
        reason = protection_reason(root, entry)
        if reason is None:
            items.append(item)
        else:
            protected.append((item, reason))
    total = sum(i.reclaim_bytes for i in items) + sum(i.reclaim_bytes for i, _ in protected)
    cap = cap_gb * GIB
    need = need_free_gb * GIB
    chosen: list[CacheItem] = []
    freed = 0
    for item in items:
        if total - freed <= cap and freed >= need:
            break
        if item.reclaim_bytes == 0:
            continue
        chosen.append(item)
        freed += item.reclaim_bytes
    shortfall = max(total - freed - cap, need - freed, 0.0)
    return CachePlan(
        cache_bytes=total,
        cap_gb=cap_gb,
        evict=tuple(chosen),
        protected=tuple(protected),
        shortfall_bytes=int(shortfall),
    )


def _log(root: Path, run_dir: Path, record: dict[str, object]) -> None:
    for path in (run_dir / "events.jsonl", root / "outputs" / "system" / "cleanup.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def evict(root: Path, index: ArtifactIndex, cache_keys: list[str], *, by: str, manual: bool = False) -> CacheResult:
    """Evict the given entries (re-checked at ``begin_evict``); updates run.json status and logs ``cache_evicted``.

    Protected entries (see ``protection_reason``) are skipped unless ``manual`` (explicit user choice).
    """
    check_project_root(root)
    runs_dir = (root / RUNS_REL).resolve()
    result = CacheResult()
    for key in cache_keys:
        entry = index.get(key)
        reason = protection_reason(root, entry) if entry is not None else None
        if reason is not None and not manual:
            result.skipped.append((key, reason))
            continue
        if entry is None or not index.begin_evict(key):
            why = "IN_USE_BY_RUNNING_WORKFLOW" if index.leased_by(key) else "not evictable (pinned, needed or gone)"
            result.skipped.append((key, why))
            continue
        run_dir = root / RUNS_REL / entry.run_id
        if run_dir.is_symlink() or not _inside(run_dir.resolve(), runs_dir) or run_dir.resolve() == runs_dir:
            index.finish_evict(key, ok=False)
            result.skipped.append((key, "run dir outside outputs/runs"))
            continue
        item = _item(root, entry)
        try:
            for rel in item.targets:
                target = run_dir / rel
                if target.is_symlink():
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
        except OSError as exc:
            index.finish_evict(key, ok=False)
            result.skipped.append((key, f"delete failed: {exc}"))
            continue
        index.finish_evict(key, ok=True)
        if (run_dir / "run.json").is_file():
            manifest = read_manifest(run_dir)
            write_manifest(run_dir, manifest.model_copy(update={"status": RunStatus.EVICTED}))
        now = time.time()
        _log(
            root,
            run_dir,
            {
                "event": "cache_evicted",
                "run_id": entry.run_id,
                "cache_key": key,
                "bytes": item.reclaim_bytes,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
                "ts_epoch": now,
                "by": by,
                "manual": manual,
                "approved_by": None,
            },
        )
        result.evicted.append(item)
    return result
