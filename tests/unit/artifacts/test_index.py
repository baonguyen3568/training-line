"""Unit tests for ``vlm_pipeline.artifacts.index.ArtifactIndex`` (SQLite in tmp_path)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.index import ArtifactIndex, LeaseOwner, OutputRef

RUN_DIR = Path("/nonexistent/outputs/graph/runs/node")  # absolute; index tests never touch it


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        self.t += 1
        return self.t


@pytest.fixture
def idx(tmp_path: Path) -> ArtifactIndex:
    return ArtifactIndex(tmp_path / "state" / "app.db", clock=Clock())


def _out(tmp_path: Path, name: str, *, create: bool = True) -> dict[str, OutputRef]:
    path = tmp_path / "runs" / name / "export"
    if create:
        path.mkdir(parents=True)
    return {"model": OutputRef("model", str(path), f"h-{name}")}


def test_record_lookup_roundtrip(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("k1", run_dir=RUN_DIR, run_id="r1", node_type="train.lora", outputs=_out(tmp_path, "r1"))
    hit = idx.lookup("k1")
    assert hit is not None
    assert hit.run_id == "r1"
    assert hit.outputs["model"].hash == "h-r1"
    assert idx.verify(hit)
    assert idx.lookup("missing") is None


def test_relative_output_path_rejected(idx: ArtifactIndex) -> None:
    with pytest.raises(ValueError, match="absolute"):
        idx.record("k", run_dir=RUN_DIR, run_id="r", node_type="t", outputs={"m": OutputRef("model", "rel/path", "h")})


def test_verify_detects_missing_files_and_hash_mismatch(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("k1", run_dir=RUN_DIR, run_id="r1", node_type="t", outputs=_out(tmp_path, "r1", create=False))
    entry = idx.lookup("k1")
    assert entry is not None
    assert not idx.verify(entry)
    idx.record("k2", run_dir=RUN_DIR, run_id="r2", node_type="t", outputs=_out(tmp_path, "r2"))
    e2 = idx.lookup("k2")
    assert e2 is not None
    assert idx.verify(e2, hasher=lambda ref: ref.hash)
    assert not idx.verify(e2, hasher=lambda _ref: "other")


def test_invalidate_then_record_revives(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("k1", run_dir=RUN_DIR, run_id="r1", node_type="t", outputs=_out(tmp_path, "r1"))
    idx.invalidate("k1")
    assert idx.lookup("k1") is None
    idx.record("k1", run_dir=RUN_DIR, run_id="r1b", node_type="t", outputs=_out(tmp_path, "r1b"))
    hit = idx.lookup("k1")
    assert hit is not None
    assert hit.run_id == "r1b"


def test_lru_order_uses_last_hit(idx: ArtifactIndex, tmp_path: Path) -> None:
    for k in ("a", "b", "c"):
        idx.record(k, run_dir=RUN_DIR, run_id=k, node_type="t", outputs=_out(tmp_path, k))
    idx.mark_hit("a")  # a is now the most recently used
    assert [e.cache_key for e in idx.evictable()] == ["b", "c", "a"]


def test_evict_three_steps_and_executor_sees_miss(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("k", run_dir=RUN_DIR, run_id="r", node_type="t", outputs=_out(tmp_path, "r"))
    assert idx.begin_evict("k")
    assert idx.lookup("k") is None  # evicting = miss for the executor
    assert not idx.begin_evict("k")  # second evictor loses the race
    idx.finish_evict("k")
    entry = idx.get("k")
    assert entry is not None
    assert entry.state == "evicted"
    assert idx.evictable() == []


def test_failed_eviction_becomes_invalid(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("k", run_dir=RUN_DIR, run_id="r", node_type="t", outputs=_out(tmp_path, "r"))
    assert idx.begin_evict("k")
    idx.finish_evict("k", ok=False)
    entry = idx.get("k")
    assert entry is not None
    assert entry.state == "invalid"


def test_required_parents_are_not_evictable(idx: ArtifactIndex, tmp_path: Path) -> None:
    idx.record("base", run_dir=RUN_DIR, run_id="base", node_type="train.sft", outputs=_out(tmp_path, "base"))
    idx.record("other", run_dir=RUN_DIR, run_id="o", node_type="t", outputs=_out(tmp_path, "o"))
    idx.record(
        "lora",
        run_dir=RUN_DIR,
        run_id="l",
        node_type="train.lora",
        outputs=_out(tmp_path, "l"),
        parents=["base"],
        requires_parent_at_load=True,
    )
    idx.record(
        "quant", run_dir=RUN_DIR, run_id="q", node_type="quantize", outputs=_out(tmp_path, "q"), parents=["lora"]
    )
    assert {e.cache_key for e in idx.evictable()} == {"lora", "quant", "other"}  # quant is self-contained
    assert not idx.begin_evict("base")
    # once the adapter is gone, its base becomes evictable
    assert idx.begin_evict("lora")
    idx.finish_evict("lora")
    assert idx.begin_evict("base")


def test_concurrent_use_from_threads(idx: ArtifactIndex, tmp_path: Path) -> None:
    def work(i: int) -> None:
        idx.record(f"k{i}", run_dir=RUN_DIR, run_id=f"r{i}", node_type="t", outputs=_out(tmp_path, f"r{i}"))
        idx.mark_hit(f"k{i}")
        assert idx.lookup(f"k{i}") is not None

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(idx.evictable()) == 16


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    first = ArtifactIndex(db)
    first.record("k", run_dir=RUN_DIR, run_id="r", node_type="t", outputs=_out(tmp_path, "r"))
    first.close()
    second = ArtifactIndex(db)
    assert second.lookup("k") is not None


def test_trained_model_flag_roundtrip_and_old_db_migration(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.execute(
        "CREATE TABLE artifacts (cache_key TEXT PRIMARY KEY, run_id TEXT NOT NULL, node_type TEXT NOT NULL,"
        " outputs_json TEXT NOT NULL, state TEXT NOT NULL, requires_parent_at_load INTEGER NOT NULL DEFAULT 0,"
        " pinned INTEGER NOT NULL DEFAULT 0, completed_at REAL NOT NULL, last_hit_at REAL, evicted_at REAL)"
    )
    old.commit()
    old.close()
    idx = ArtifactIndex(db)
    idx.record(
        "k", run_dir=RUN_DIR, run_id="r", node_type="vlm/train.lora", outputs=_out(tmp_path, "r"), trained_model=True
    )
    entry = idx.lookup("k")
    assert entry is not None
    assert entry.trained_model


def test_leased_entries_are_not_evictable_until_released(idx: ArtifactIndex, tmp_path: Path) -> None:
    for k in ("in-use", "free"):
        idx.record(k, run_dir=RUN_DIR, run_id=k, node_type="t", outputs=_out(tmp_path, k))
    idx.lease(["in-use"], run_id="graph-1")
    idx.lease(["in-use"], run_id="graph-1")  # idempotent
    idx.lease(["in-use"], run_id="graph-2")
    assert [e.cache_key for e in idx.evictable()] == ["free"]
    assert not idx.begin_evict("in-use")
    assert idx.leased_by("in-use") == ["graph-1", "graph-2"]
    idx.release_run("graph-1")
    assert not idx.begin_evict("in-use")  # graph-2 still holds it
    idx.release_run("graph-2")
    assert idx.begin_evict("in-use")


def test_old_leases_still_block(tmp_path: Path) -> None:
    clock = Clock()
    idx = ArtifactIndex(tmp_path / "app.db", clock=clock)
    idx.record("k", run_dir=RUN_DIR, run_id="r", node_type="t", outputs=_out(tmp_path, "r"))
    idx.lease(["k"], run_id="crashed-graph")
    clock.t += 25 * 3600
    assert not idx.begin_evict("k")  # fail closed: only release_run frees it
    assert not idx.begin_evict("k")


def test_release_all_except_drops_leases_of_dead_runs(idx: ArtifactIndex, tmp_path: Path) -> None:
    for k in ("a", "b", "c"):
        idx.record(k, run_dir=RUN_DIR, run_id=k, node_type="t", outputs=_out(tmp_path, k))
    idx.lease(["a", "b"], run_id="crashed", owner=LeaseOwner(999_999_999, "1"))  # a pid that cannot exist
    idx.lease(["c"], run_id="alive", owner=LeaseOwner(999_999_999, "1"))
    assert idx.release_all_except(["alive"]) == 2
    assert idx.leased_by("a") == []
    assert idx.leased_by("c") == ["alive"]
    assert idx.release_all_except(["alive"]) == 0


def test_a_restart_keeps_leases_of_graphs_another_live_process_runs(idx: ArtifactIndex, tmp_path: Path) -> None:
    """PM: a graph run by a script (run_workflow.py) is not in the app's run list; its live owner keeps its lease."""
    import os

    from vlm_pipeline.artifacts.index import process_start

    for k in ("ours", "script", "reused", "legacy"):
        idx.record(k, run_dir=RUN_DIR, run_id=k, node_type="t", outputs=_out(tmp_path, k))
    idx.lease(["ours"], run_id="g-dead-app", owner=LeaseOwner(999_999_999, "1"))
    idx.lease(["script"], run_id="g-script")  # this test process: alive
    idx.lease(
        ["reused"], run_id="g-reused", owner=LeaseOwner(os.getpid(), "1")
    )  # same pid, another start time: not the owner
    with idx._tx() as db:  # a lease written before owners were recorded
        db.execute("INSERT INTO artifact_leases (cache_key, run_id, since) VALUES ('legacy', 'g-legacy', 0)")
    assert idx.release_all_except([]) == 3
    assert idx.leased_by("script") == ["g-script"]
    assert [idx.leased_by(k) for k in ("ours", "reused", "legacy")] == [[], [], []]
    assert process_start(os.getpid()) is not None


def test_leases_of_an_index_created_before_owners_get_the_columns(tmp_path: Path) -> None:
    import sqlite3

    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE artifact_leases (cache_key TEXT NOT NULL, run_id TEXT NOT NULL, since REAL NOT NULL,"
               " PRIMARY KEY (cache_key, run_id))")  # fmt: skip
    db.execute("INSERT INTO artifact_leases VALUES ('k', 'g-old', 0)")
    db.commit()
    db.close()
    idx = ArtifactIndex(tmp_path / "old.db")
    assert idx.leased_by("k") == ["g-old"]
    assert idx.release_all_except([]) == 1


def test_an_owner_in_another_pid_namespace_is_kept(idx: ArtifactIndex, tmp_path: Path) -> None:
    """cleanup R1: app in compose, script on the host: the pid cannot be checked from here, so the lease stays."""
    idx.record("k", run_dir=RUN_DIR, run_id="k", node_type="t", outputs=_out(tmp_path, "k"))
    idx.lease(["k"], run_id="g-host-script", owner=LeaseOwner(999_999_999, "1", "pid:[1]"))
    assert idx.release_all_except([]) == 0
    assert idx.leased_by("k") == ["g-host-script"]
    idx.release_run("g-host-script")
    assert idx.leased_by("k") == []


def test_a_live_owner_keeps_its_lease_when_a_dead_one_shared_the_run_id(idx: ArtifactIndex, tmp_path: Path) -> None:
    """cleanup R2: rows go per owner."""
    for k in ("a", "b"):
        idx.record(k, run_dir=RUN_DIR, run_id=k, node_type="t", outputs=_out(tmp_path, k))
    idx.lease(["a"], run_id="g-1", owner=LeaseOwner(999_999_999, "1"))
    idx.lease(["b"], run_id="g-1")  # this process, alive
    assert idx.release_all_except([]) == 1
    assert idx.leased_by("a") == []
    assert idx.leased_by("b") == ["g-1"]
