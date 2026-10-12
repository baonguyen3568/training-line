"""Unit tests for ``vlm_pipeline.artifacts.cache_retention`` (node-cache LRU eviction)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.cache_retention import evict, plan_cache
from vlm_pipeline.artifacts.index import ArtifactIndex, OutputRef
from vlm_pipeline.artifacts.testing import FakeRunSpec, make_fake_run
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import RunStatus, read_manifest

KB = 1024


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        self.t += 1
        return self.t


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    return tmp_path


@pytest.fixture
def idx(root: Path) -> ArtifactIndex:
    return ArtifactIndex(root / "outputs" / "app.db", clock=Clock())


def _cached_run(
    root: Path,
    idx: ArtifactIndex,
    run_id: str,
    *,
    out_kb: int = 64,
    exported: bool = True,
    train: bool = False,
    **record: object,
) -> Path:
    """A cached node run; by default a cheap, regenerable eval node (train=True: a fine-tune run)."""
    stage, task, node_type = ("train", "lora", "train.lora") if train else ("eval", "eval", "eval.benchmark")
    spec = FakeRunSpec(run_id, stage=stage, task=task, n_steps=10, checkpoint_every=10)  # type: ignore[arg-type]
    run = make_fake_run(root / "outputs" / "runs", spec)
    out = run / "artifacts" / "model"
    out.mkdir(parents=True)
    (out / "weights.bin").write_bytes(b"\0" * out_kb * KB)
    (run / "export").mkdir()
    if exported:
        (run / "export" / "named.safetensors").write_bytes(b"\0" * KB)
    idx.record(
        run_id,
        run_id=run_id,
        node_type=node_type,
        outputs={"model": OutputRef("model", str(out), "h"), "named": OutputRef("model", str(run / "export"), "h2")},
        trained_model=train,
        **record,  # type: ignore[arg-type]
    )
    return run


def test_plan_is_lru_and_stops_under_cap(root: Path, idx: ArtifactIndex) -> None:
    for k in ("a", "b", "c"):
        _cached_run(root, idx, k)
    idx.mark_hit("a")  # a most recently used
    each = plan_cache(root, idx, cap_gb=1e9).cache_bytes / 3
    p = plan_cache(root, idx, cap_gb=(2 * each + 1) / GIB)
    assert [i.cache_key for i in p.evict] == ["b"]
    assert plan_cache(root, idx, cap_gb=0).evict[-1].cache_key == "a"
    assert plan_cache(root, idx, cap_gb=1e9).evict == ()


def test_need_free_forces_eviction_under_cap(root: Path, idx: ArtifactIndex) -> None:
    _cached_run(root, idx, "a")
    _cached_run(root, idx, "b")
    p = plan_cache(root, idx, cap_gb=1e9, need_free_gb=1 / GIB)
    assert [i.cache_key for i in p.evict] == ["a"]


def test_evict_frees_heavy_bytes_and_keeps_history(root: Path, idx: ArtifactIndex) -> None:
    run = _cached_run(root, idx, "a")
    p = plan_cache(root, idx, cap_gb=0)
    assert p.evict[0].targets == ("artifacts/model", "checkpoints")
    res = evict(root, idx, ["a"], by="app")
    assert [i.cache_key for i in res.evicted] == ["a"]
    assert not (run / "checkpoints").exists()
    assert not (run / "artifacts" / "model").exists()
    for kept in ("run.json", "metrics.jsonl", "events.jsonl", "logs", "export/named.safetensors"):
        assert (run / kept).exists(), kept
    assert read_manifest(run).status is RunStatus.EVICTED
    entry = idx.get("a")
    assert entry is not None
    assert entry.state == "evicted"
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "cache_evicted"
    assert events[-1]["cache_key"] == "a"


def test_pinned_and_needed_parents_are_not_evicted(root: Path, idx: ArtifactIndex) -> None:
    _cached_run(root, idx, "base")
    _cached_run(root, idx, "lora", parents=["base"], requires_parent_at_load=True)
    _cached_run(root, idx, "pinned")
    idx.set_pinned("pinned", pinned=True)
    assert {i.cache_key for i in plan_cache(root, idx, cap_gb=0).evict} == {"lora"}
    res = evict(root, idx, ["base", "pinned"], by="app")
    assert res.evicted == []
    assert len(res.skipped) == 2


def test_outputs_outside_run_dir_are_never_deleted(root: Path, idx: ArtifactIndex) -> None:
    shared = root / ".cache" / "hf" / "models--x"
    shared.mkdir(parents=True)
    (shared / "w.bin").write_bytes(b"\0" * KB)
    make_fake_run(root / "outputs" / "runs", FakeRunSpec("dl", n_steps=1, checkpoint_every=0))
    idx.record("dl", run_id="dl", node_type="model.hf_pick", outputs={"model": OutputRef("model", str(shared), "h")})
    evict(root, idx, ["dl"], by="app")
    assert (shared / "w.bin").exists()


def test_symlinked_output_is_not_followed(
    root: Path, idx: ArtifactIndex, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "keep.bin").write_bytes(b"x")
    run = make_fake_run(root / "outputs" / "runs", FakeRunSpec("s", n_steps=1, checkpoint_every=0))
    (run / "artifacts").mkdir()
    (run / "artifacts" / "link").symlink_to(outside)
    idx.record("s", run_id="s", node_type="t", outputs={"m": OutputRef("model", str(run / "artifacts" / "link"), "h")})
    evict(root, idx, ["s"], by="app")
    assert (outside / "keep.bin").exists()


def test_cache_evictions_do_not_eat_the_checkpoint_quota(root: Path, idx: ArtifactIndex) -> None:
    from vlm_pipeline.artifacts.retention import _unapproved_bytes_since

    _cached_run(root, idx, "a", out_kb=512)
    evict(root, idx, ["a"], by="app")
    assert _unapproved_bytes_since(root / "outputs" / "system" / "cleanup.jsonl", 0) == 0


@pytest.mark.parametrize("exported", [False, True])
def test_trained_models_are_never_evicted_automatically(root: Path, idx: ArtifactIndex, exported: bool) -> None:
    run = _cached_run(root, idx, "train-run", exported=exported, train=True)  # hours of GPU
    make_fake_run(root / "outputs" / "runs", FakeRunSpec("prep", stage="data", task="prep", checkpoint_every=0))
    prep_out = root / "outputs" / "runs" / "prep" / "artifacts" / "shards"
    prep_out.mkdir(parents=True)
    (prep_out / "s0.tar").write_bytes(b"\0" * 4 * KB)
    idx.record("prep", run_id="prep", node_type="data.prepare", outputs={"d": OutputRef("dataset", str(prep_out), "h")})
    p = plan_cache(root, idx, cap_gb=0)
    assert [i.cache_key for i in p.evict] == ["prep"]
    assert [(i.cache_key, why) for i, why in p.protected] == [("train-run", "TRAINED_MODEL")]
    assert p.shortfall_bytes > 0  # cap 0 cannot be met without touching the trained model -> caller reports DISK
    res = evict(root, idx, ["train-run"], by="app")
    assert res.skipped == [("train-run", "TRAINED_MODEL")]
    assert (run / "checkpoints").exists()
    manual = evict(root, idx, ["train-run"], by="user", confirmed={"train-run"})  # explicit choice in the storage panel
    assert [i.cache_key for i in manual.evicted] == ["train-run"]
    assert not (run / "checkpoints").exists()


def test_no_shortfall_when_cap_is_met(root: Path, idx: ArtifactIndex) -> None:
    _cached_run(root, idx, "a")
    assert plan_cache(root, idx, cap_gb=1e9).shortfall_bytes == 0


def test_unreadable_manifest_is_protected(root: Path, idx: ArtifactIndex) -> None:
    run = _cached_run(root, idx, "broken")
    (run / "run.json").write_text("{not json")
    p = plan_cache(root, idx, cap_gb=0)
    assert p.evict == ()
    assert [(i.cache_key, why) for i, why in p.protected] == [("broken", "MANIFEST_UNREADABLE")]
    assert evict(root, idx, ["broken"], by="app").skipped == [("broken", "MANIFEST_UNREADABLE")]


def test_namespaced_train_node_with_broken_manifest_is_protected(root: Path, idx: ArtifactIndex) -> None:
    run = make_fake_run(root / "outputs" / "runs", FakeRunSpec("ns", stage="eval", task="eval", checkpoint_every=10))
    (run / "run.json").unlink()
    idx.record(
        "ns",
        run_id="ns",
        node_type="vlm/train.lora",
        outputs={"m": OutputRef("model", str(run / "checkpoints"), "h")},
        trained_model=True,
    )
    p = plan_cache(root, idx, cap_gb=0)
    assert [(i.cache_key, why) for i, why in p.protected] == [("ns", "TRAINED_MODEL")]
    assert (run / "checkpoints").exists()


def test_leased_entry_is_skipped_as_in_use(root: Path, idx: ArtifactIndex) -> None:
    run = _cached_run(root, idx, "input-of-long-job")
    idx.lease(["input-of-long-job"], run_id="graph-7")
    assert plan_cache(root, idx, cap_gb=0).evict == ()
    res = evict(root, idx, ["input-of-long-job"], by="user", confirmed={"input-of-long-job"})
    assert res.skipped == [("input-of-long-job", "IN_USE_BY_RUNNING_WORKFLOW")]
    assert (run / "checkpoints").exists()


def _graph_node(root: Path, idx: ArtifactIndex, name: str, *, run_dir: Path | None = None) -> Path:
    """Executor layout: outputs/graph/runs/<node dir> with run.json, run_id = node dir, graph id in parent."""
    node_dir = root / "outputs" / "graph" / "runs" / name
    make_fake_run(node_dir.parent, FakeRunSpec(name, stage="eval", task="eval", checkpoint_every=10,
                                               parent_run_id="g-20261012-180000-abc123"))  # fmt: skip
    out = node_dir / "predictions"
    out.mkdir()
    (out / "p.jsonl").write_bytes(b"\0" * KB)
    idx.record(
        name,
        run_id=name,
        node_type="vlm/eval.benchmark",
        outputs={"pred": OutputRef("report", str(out), "h")},
        run_dir=run_dir if run_dir is not None else node_dir,
    )
    return node_dir


def test_executor_layout_under_outputs_graph_runs_is_evicted(root: Path, idx: ArtifactIndex) -> None:
    node = _graph_node(root, idx, "e-0123456789abcdef")
    p = plan_cache(root, idx, cap_gb=0)
    assert [i.cache_key for i in p.evict] == ["e-0123456789abcdef"]
    assert p.evict[0].targets == ("checkpoints", "predictions")
    res = evict(root, idx, ["e-0123456789abcdef"], by="app")
    assert [i.cache_key for i in res.evicted] == ["e-0123456789abcdef"]
    assert res.freed_bytes > 0
    assert not (node / "predictions").exists()
    assert (node / "run.json").exists()
    assert read_manifest(node).status is RunStatus.EVICTED


def test_run_dir_outside_known_roots_is_never_touched(
    root: Path, idx: ArtifactIndex, tmp_path_factory: pytest.TempPathFactory
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "x"
    node = _graph_node(root, idx, "e-1", run_dir=elsewhere)  # index points outside the project roots
    elsewhere.mkdir()
    (elsewhere / "precious.bin").write_bytes(b"x")
    p = plan_cache(root, idx, cap_gb=0)
    assert p.evict == ()
    assert [(i.cache_key, why) for i, why in p.protected] == [("e-1", "MANIFEST_UNREADABLE")]
    res = evict(root, idx, ["e-1"], by="admin", confirmed={"e-1"})
    assert res.evicted == []
    assert (elsewhere / "precious.bin").exists()
    assert (node / "predictions").exists()


def test_relative_run_dir_is_rejected(idx: ArtifactIndex) -> None:
    with pytest.raises(ValueError, match="run_dir must be absolute"):
        idx.record("k", run_id="k", node_type="t", outputs={}, run_dir=Path("outputs/graph/runs/k"))
