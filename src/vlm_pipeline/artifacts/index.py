"""Node-output cache index in SQLite (``outputs/app.db``) — single source of truth (P009 §5, ADR-005 / P005 §6 R1).

The executor ``record``s every completed cacheable node, ``lookup``s before running one, ``verify``s a hit and
``mark_hit``s it (LRU input), and ``invalidate``s an entry whose files no longer match. Retention evicts in three steps
so it never races the executor: ``begin_evict`` (completed -> evicting, executor sees a miss) -> delete files ->
``finish_evict`` (-> evicted). Pinned entries and parents still needed at load time by a live child cannot be evicted.

Stdlib only (``sqlite3``), WAL mode, one connection guarded by a lock so one instance can be shared across threads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

State = Literal["completed", "evicting", "evicted", "invalid"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    cache_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('completed', 'evicting', 'evicted', 'invalid')),
    requires_parent_at_load INTEGER NOT NULL DEFAULT 0,
    trained_model INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    completed_at REAL NOT NULL,
    last_hit_at REAL,
    evicted_at REAL
);
CREATE TABLE IF NOT EXISTS artifact_parents (
    child TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
CREATE INDEX IF NOT EXISTS artifact_parents_parent ON artifact_parents (parent);
"""


@dataclass(frozen=True)
class OutputRef:
    type: str  # port type, e.g. "model", "dataset", "report"
    path: str  # absolute path of the artifact (file or directory)
    hash: str  # manifest hash from the worker (deterministic for FakeWorker)


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    run_id: str
    node_type: str
    outputs: Mapping[str, OutputRef]  # port -> artifact
    state: State
    completed_at: float
    last_hit_at: float | None
    pinned: bool
    requires_parent_at_load: bool
    trained_model: bool  # set by the executor from the node registry; never evicted automatically


def _outputs_to_json(outputs: Mapping[str, OutputRef]) -> str:
    for port, ref in outputs.items():
        if not Path(ref.path).is_absolute():
            raise ValueError(f"output {port!r} path must be absolute: {ref.path}")
    return json.dumps({p: {"type": r.type, "path": r.path, "hash": r.hash} for p, r in outputs.items()}, sort_keys=True)


def _outputs_from_json(raw: str) -> dict[str, OutputRef]:
    return {p: OutputRef(str(v["type"]), str(v["path"]), str(v["hash"])) for p, v in json.loads(raw).items()}


class ArtifactIndex:
    def __init__(self, db_path: Path, *, clock: Callable[[], float] = time.time) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
            columns = {r["name"] for r in self._db.execute("PRAGMA table_info(artifacts)")}
            if "trained_model" not in columns:  # db created before this column existed
                self._db.execute("ALTER TABLE artifacts ADD COLUMN trained_model INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    # ------------------------------------------------------------------ executor side
    def record(
        self,
        cache_key: str,
        *,
        run_id: str,
        node_type: str,
        outputs: Mapping[str, OutputRef],
        parents: Sequence[str] = (),
        requires_parent_at_load: bool = False,
        trained_model: bool = False,
    ) -> None:
        """Register a completed node output (replaces an invalid/evicted entry with the same key).

        ``trained_model``: the node produces a trained model (registry metadata, e.g. a fine-tune node); such entries
        are only ever evicted by an explicit user choice.
        """
        payload = _outputs_to_json(outputs)
        with self._tx() as db:
            db.execute(
                "INSERT INTO artifacts (cache_key, run_id, node_type, outputs_json, state, requires_parent_at_load,"
                " trained_model, completed_at) VALUES (?, ?, ?, ?, 'completed', ?, ?, ?)"
                " ON CONFLICT(cache_key) DO UPDATE SET run_id=excluded.run_id, node_type=excluded.node_type,"
                " outputs_json=excluded.outputs_json, state='completed',"
                " requires_parent_at_load=excluded.requires_parent_at_load, trained_model=excluded.trained_model,"
                " completed_at=excluded.completed_at,"
                " last_hit_at=NULL, evicted_at=NULL",
                (
                    cache_key,
                    run_id,
                    node_type,
                    payload,
                    int(requires_parent_at_load),
                    int(trained_model),
                    self._clock(),
                ),
            )
            db.execute("DELETE FROM artifact_parents WHERE child = ?", (cache_key,))
            db.executemany(
                "INSERT OR IGNORE INTO artifact_parents (child, parent) VALUES (?, ?)",
                [(cache_key, p) for p in parents],
            )

    def get(self, cache_key: str) -> CacheEntry | None:
        """Entry in any state (for retention/UI); executors use ``lookup``."""
        with self._lock:
            row = self._db.execute("SELECT * FROM artifacts WHERE cache_key = ?", (cache_key,)).fetchone()
        return None if row is None else self._entry(row)

    def lookup(self, cache_key: str) -> CacheEntry | None:
        """A usable cache hit, i.e. only ``completed`` entries (evicting/evicted/invalid count as a miss)."""
        entry = self.get(cache_key)
        return entry if entry is not None and entry.state == "completed" else None

    @staticmethod
    def verify(entry: CacheEntry, *, hasher: Callable[[OutputRef], str | None] | None = None) -> bool:
        """Every output path still exists (and matches its hash when ``hasher`` is given)."""
        for ref in entry.outputs.values():
            if not Path(ref.path).exists():
                return False
            if hasher is not None and hasher(ref) != ref.hash:
                return False
        return True

    def mark_hit(self, cache_key: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE artifacts SET last_hit_at = ? WHERE cache_key = ?", (self._clock(), cache_key))

    def invalidate(self, cache_key: str) -> None:
        """Never hit this entry again (files missing/changed); a later ``record`` revives the key."""
        with self._tx() as db:
            db.execute("UPDATE artifacts SET state = 'invalid' WHERE cache_key = ?", (cache_key,))

    # ------------------------------------------------------------------ retention side
    def set_pinned(self, cache_key: str, pinned: bool) -> None:
        with self._tx() as db:
            db.execute("UPDATE artifacts SET pinned = ? WHERE cache_key = ?", (int(pinned), cache_key))

    def live_dependents(self, cache_key: str) -> list[str]:
        """Completed children that need ``cache_key`` when they are loaded (e.g. LoRA adapter -> base model)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT a.cache_key FROM artifact_parents p JOIN artifacts a ON a.cache_key = p.child"
                " WHERE p.parent = ? AND a.requires_parent_at_load = 1 AND a.state IN ('completed', 'evicting')",
                (cache_key,),
            ).fetchall()
        return sorted(r["cache_key"] for r in rows)

    def evictable(self) -> list[CacheEntry]:
        """Completed, unpinned entries without a live dependent, least recently used first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT a.* FROM artifacts a WHERE a.state = 'completed' AND a.pinned = 0 AND NOT EXISTS ("
                " SELECT 1 FROM artifact_parents p JOIN artifacts c ON c.cache_key = p.child"
                " WHERE p.parent = a.cache_key AND c.requires_parent_at_load = 1"
                " AND c.state IN ('completed', 'evicting'))"
                " ORDER BY COALESCE(a.last_hit_at, a.completed_at) ASC, a.cache_key ASC"
            ).fetchall()
        return [self._entry(r) for r in rows]

    def begin_evict(self, cache_key: str) -> bool:
        """Atomically completed -> evicting if still evictable; False means someone else changed it (skip)."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE artifacts SET state = 'evicting' WHERE cache_key = ? AND state = 'completed' AND pinned = 0"
                " AND NOT EXISTS (SELECT 1 FROM artifact_parents p JOIN artifacts c ON c.cache_key = p.child"
                " WHERE p.parent = ? AND c.requires_parent_at_load = 1 AND c.state IN ('completed', 'evicting'))",
                (cache_key, cache_key),
            )
            return cur.rowcount == 1

    def finish_evict(self, cache_key: str, *, ok: bool = True) -> None:
        """Evicting -> evicted (files gone) or -> invalid (deletion failed half way: never hit it again)."""
        with self._tx() as db:
            if ok:
                db.execute(
                    "UPDATE artifacts SET state = 'evicted', evicted_at = ? WHERE cache_key = ? AND state = 'evicting'",
                    (self._clock(), cache_key),
                )
            else:
                db.execute(
                    "UPDATE artifacts SET state = 'invalid' WHERE cache_key = ? AND state = 'evicting'", (cache_key,)
                )

    @staticmethod
    def _entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            cache_key=row["cache_key"],
            run_id=row["run_id"],
            node_type=row["node_type"],
            outputs=_outputs_from_json(row["outputs_json"]),
            state=row["state"],
            completed_at=row["completed_at"],
            last_hit_at=row["last_hit_at"],
            pinned=bool(row["pinned"]),
            requires_parent_at_load=bool(row["requires_parent_at_load"]),
            trained_model=bool(row["trained_model"]),
        )
