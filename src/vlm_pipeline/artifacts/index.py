"""Node-output cache index in SQLite (``outputs/app.db``) — single source of truth (P009 §5, ADR-005 / P005 §6 R1).

The executor ``record``s every completed cacheable node, ``lookup``s before running one, ``verify``s a hit and
``mark_hit``s it (LRU input), and ``invalidate``s an entry whose files no longer match. Retention evicts in three steps
so it never races the executor: ``begin_evict`` (completed -> evicting, executor sees a miss) -> delete files ->
``finish_evict`` (-> evicted). Parents still needed at load time by a live child cannot be evicted. Pinning is a KEEP
file (cache_retention); the ``pinned`` column stays for old databases.

The executor ``lease``s every entry it hits or produces for the duration of a graph run and ``release_run``s in a
``finally``; leased entries are never evicted, however old the lease (fail closed: a crashed graph's lease is
released by the startup reconciliation, never by age).

Stdlib only (``sqlite3``), WAL mode, one connection guarded by a lock so one instance can be shared across threads.
"""

from __future__ import annotations

import json
import logging
import os
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
    run_dir TEXT,
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
CREATE TABLE IF NOT EXISTS artifact_leases (
    cache_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    since REAL NOT NULL,
    owner_pid INTEGER,
    owner_start TEXT,
    owner_ns TEXT,
    PRIMARY KEY (cache_key, run_id)
);
CREATE INDEX IF NOT EXISTS artifacts_lru ON artifacts (state, pinned, last_hit_at);
"""

# An entry is not evictable while a running graph holds a lease on it (it may be the input of a long job) or while
# a live child needs it at load time. Kept as full literals (no string building around SQL).
_EVICTABLE_SQL = (
    "SELECT a.* FROM artifacts a WHERE a.state = 'completed' AND a.pinned = 0"
    " AND NOT EXISTS (SELECT 1 FROM artifact_parents p JOIN artifacts c ON c.cache_key = p.child"
    " WHERE p.parent = a.cache_key AND c.requires_parent_at_load = 1 AND c.state IN ('completed', 'evicting'))"
    " AND NOT EXISTS (SELECT 1 FROM artifact_leases l WHERE l.cache_key = a.cache_key)"
    " ORDER BY COALESCE(a.last_hit_at, a.completed_at) ASC, a.cache_key ASC"
)
_BEGIN_EVICT_SQL = (
    "UPDATE artifacts SET state = 'evicting' WHERE cache_key = :k AND state = 'completed' AND pinned = 0"
    " AND NOT EXISTS (SELECT 1 FROM artifact_parents p JOIN artifacts c ON c.cache_key = p.child"
    " WHERE p.parent = :k AND c.requires_parent_at_load = 1 AND c.state IN ('completed', 'evicting'))"
    " AND NOT EXISTS (SELECT 1 FROM artifact_leases l WHERE l.cache_key = :k)"
)
INDEX_REL = Path("outputs") / "graph" / "app.db"  # same file the executor opens (state_dir / "app.db")


def default_index_path(root: Path) -> Path:
    """The project's cache index; executor, app and CLIs must all open this one file."""
    return root / INDEX_REL


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
    run_dir: str | None  # absolute host path of the node's run directory (None only for legacy rows)


def _outputs_to_json(outputs: Mapping[str, OutputRef]) -> str:
    for port, ref in outputs.items():
        if not Path(ref.path).is_absolute():
            raise ValueError(f"output {port!r} path must be absolute: {ref.path}")
    return json.dumps({p: {"type": r.type, "path": r.path, "hash": r.hash} for p, r in outputs.items()}, sort_keys=True)


def _outputs_from_json(raw: str) -> dict[str, OutputRef]:
    return {p: OutputRef(str(v["type"]), str(v["path"]), str(v["hash"])) for p, v in json.loads(raw).items()}


_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaseOwner:
    """The process holding a lease: pid, its start time (no pid reuse) and its pid namespace (compose vs host)."""

    pid: int
    start: str | None
    ns: str | None = None


def process_start(pid: int) -> str | None:
    """Start time of ``pid`` in clock ticks since boot (/proc/<pid>/stat field 22), or None if it is not running."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    fields = raw.rsplit(")", 1)[-1].split()  # after "(comm)": state is field 3, starttime field 22
    return fields[19] if len(fields) > 19 else None


def pid_namespace() -> str | None:
    """This process's pid namespace (``pid:[4026531836]``); None when /proc does not say."""
    try:
        return str(Path("/proc/self/ns/pid").readlink())
    except OSError:
        return None


def current_owner() -> LeaseOwner:
    pid = os.getpid()
    return LeaseOwner(pid, process_start(pid), pid_namespace())


def owner_is_alive(owner: LeaseOwner) -> bool:
    """The lease owner still runs: same pid and start time.

    An owner in another pid namespace (app in compose, script on the host) cannot be checked from here: it is kept
    (fail closed), only release_run frees it.
    """
    if owner.ns is not None and owner.ns != pid_namespace():
        _log.info("lease owner pid %s is in another pid namespace: kept", owner.pid)
        return True
    current = process_start(owner.pid)
    return current is not None and (owner.start is None or current == owner.start)


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
            if "run_dir" not in columns:
                self._db.execute("ALTER TABLE artifacts ADD COLUMN run_dir TEXT")
            lease_columns = {r["name"] for r in self._db.execute("PRAGMA table_info(artifact_leases)")}
            for column, kind in (("owner_pid", "INTEGER"), ("owner_start", "TEXT"), ("owner_ns", "TEXT")):
                if column not in lease_columns:  # leases from before owners were recorded: released as before
                    self._db.execute(f"ALTER TABLE artifact_leases ADD COLUMN {column} {kind}")

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
        run_dir: Path,
    ) -> None:
        """Register a completed node output (replaces an invalid/evicted entry with the same key).

        ``trained_model``: the node produces a trained model (registry metadata, e.g. a fine-tune node); such entries
        are only ever evicted by an explicit user choice. ``run_dir``: absolute host path of the node run directory
        (``outputs/runs/<id>`` or ``outputs/graph/runs/<id>``); eviction only ever deletes inside it.
        """
        payload = _outputs_to_json(outputs)
        if not run_dir.is_absolute():
            raise ValueError(f"run_dir must be absolute: {run_dir}")
        with self._tx() as db:
            db.execute(
                "INSERT INTO artifacts (cache_key, run_id, node_type, outputs_json, state, requires_parent_at_load,"
                " trained_model, run_dir, completed_at) VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?)"
                " ON CONFLICT(cache_key) DO UPDATE SET run_id=excluded.run_id, node_type=excluded.node_type,"
                " outputs_json=excluded.outputs_json, state='completed',"
                " requires_parent_at_load=excluded.requires_parent_at_load, trained_model=excluded.trained_model,"
                " run_dir=excluded.run_dir,"
                " completed_at=excluded.completed_at,"
                " last_hit_at=NULL, evicted_at=NULL",
                (
                    cache_key,
                    run_id,
                    node_type,
                    payload,
                    int(requires_parent_at_load),
                    int(trained_model),
                    str(run_dir),
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

    def lease(self, cache_keys: Sequence[str], run_id: str, *, owner: LeaseOwner | None = None) -> None:
        """Protect ``cache_keys`` from eviction while graph run ``run_id`` uses them (idempotent).

        ``owner`` = the process running that graph, this process by default: an app restart keeps the leases of
        graphs another live process runs (scripts, CLI) and drops only dead owners'.
        """
        now = self._clock()
        who = owner or current_owner()
        with self._tx() as db:
            db.executemany(
                "INSERT OR IGNORE INTO artifact_leases (cache_key, run_id, since, owner_pid, owner_start, owner_ns)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(k, run_id, now, who.pid, who.start, who.ns) for k in cache_keys],
            )

    def release_run(self, run_id: str) -> None:
        """Drop every lease held by graph run ``run_id`` (call from ``finally``)."""
        with self._tx() as db:
            db.execute("DELETE FROM artifact_leases WHERE run_id = ?", (run_id,))

    def release_all_except(
        self, active_run_ids: Sequence[str], *, owner_alive: Callable[[LeaseOwner], bool] | None = None
    ) -> int:
        """Drop leases left by dead processes (call at app startup); returns how many.

        A lease is kept when its run is in ``active_run_ids`` or its owner process is alive (``owner_is_alive``: same
        pid AND start time; an owner in another pid namespace cannot be judged and is kept). Each owner's rows go
        on their own: a live owner never loses a lease because a dead one shared its run id. Leases without an owner
        (older rows) follow the run list.
        """
        alive = owner_alive or owner_is_alive
        active = set(active_run_ids)
        with self._tx() as db:
            rows = db.execute(
                "SELECT DISTINCT run_id, owner_pid, owner_start, owner_ns FROM artifact_leases"
            ).fetchall()
            dead = [
                r for r in rows
                if r["run_id"] not in active
                and (r["owner_pid"] is None or not alive(LeaseOwner(r["owner_pid"], r["owner_start"], r["owner_ns"])))
            ]  # fmt: skip
            before = db.total_changes
            db.executemany(
                "DELETE FROM artifact_leases"
                " WHERE run_id = ? AND owner_pid IS ? AND owner_start IS ? AND owner_ns IS ?",
                [(r["run_id"], r["owner_pid"], r["owner_start"], r["owner_ns"]) for r in dead],
            )
            return db.total_changes - before

    def leases_of(self, run_id: str) -> list[tuple[str, LeaseOwner | None, float]]:
        """(cache_key, owner or None for legacy rows, since) of every lease graph run ``run_id`` holds."""
        with self._lock:
            rows = self._db.execute(
                "SELECT cache_key, since, owner_pid, owner_start, owner_ns FROM artifact_leases WHERE run_id = ?"
                " ORDER BY cache_key",
                (run_id,),
            ).fetchall()
        return [
            (
                r["cache_key"],
                LeaseOwner(r["owner_pid"], r["owner_start"], r["owner_ns"]) if r["owner_pid"] is not None else None,
                r["since"],
            )
            for r in rows
        ]

    def leased_by(self, cache_key: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT run_id FROM artifact_leases WHERE cache_key = ? ORDER BY run_id", (cache_key,)
            ).fetchall()
        return [r["run_id"] for r in rows]

    def keys_for_run_dirs(self, run_dirs: Sequence[Path]) -> list[str]:
        """Cache keys whose node run dir is one of ``run_dirs`` (absolute paths, as recorded), any state."""
        wanted = json.dumps(sorted({str(p) for p in run_dirs}))
        with self._lock:
            rows = self._db.execute(
                "SELECT cache_key FROM artifacts WHERE run_dir IN (SELECT value FROM json_each(?)) ORDER BY cache_key",
                (wanted,),
            ).fetchall()
        return [r["cache_key"] for r in rows]

    # ------------------------------------------------------------------ retention side
    def evictable(self) -> list[CacheEntry]:
        """Completed, unpinned entries without a live dependent, least recently used first."""
        with self._lock:
            rows = self._db.execute(_EVICTABLE_SQL).fetchall()
        return [self._entry(r) for r in rows]

    def begin_evict(self, cache_key: str) -> bool:
        """Atomically completed -> evicting if still evictable; False means someone else changed it (skip)."""
        with self._tx() as db:
            cur = db.execute(_BEGIN_EVICT_SQL, {"k": cache_key})
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
            run_dir=row["run_dir"],
        )
