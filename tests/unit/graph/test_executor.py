"""Executor behaviour beyond the spec: events, cancel, resume step, broken UI callback (CPU, FakeWorker)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vlm_pipeline.graph.errors import InterruptedByMemguard
from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.graph.testing import FakeWorker, make_test_registry
from vlm_pipeline.graph.worker import NOT_RESUMABLE_MARKER, NodeJob, NodeResult


def _wf() -> dict[str, Any]:
    def n(i: str, t: str, **p: Any) -> dict[str, Any]:
        return {"id": i, "type": t, "version": "1.0.0", "params": p}

    return {
        "format": "vlm-workflow",
        "schema_version": 1,
        "name": "t",
        "nodes": [n("d", "test.dataset"), n("m", "test.model"), n("t", "test.train", max_steps=10)],
        "edges": [{"from": ["d", "data"], "to": ["t", "data"]}, {"from": ["m", "model"], "to": ["t", "base_model"]}],
    }


def test_events_carry_run_id_and_timestamp(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    ex = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append)
    run = ex.run(_wf(), graph_run_id="g-test")
    assert run.graph_run_id == "g-test"
    assert all(e["graph_run_id"] == "g-test" and isinstance(e["ts"], float) for e in events)
    assert events[0] == {**events[0], "type": "run_status", "status": "running"}
    assert events[-1]["type"] == "run_status"
    assert events[-1]["status"] == "completed"
    statuses = [(e["node_id"], e["status"]) for e in events if e["type"] == "node_status"]
    assert ("t", "running") in statuses
    assert ("t", "completed") in statuses


def test_invalid_graph_emits_issues_and_no_node_events(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    wf = _wf()
    wf["edges"] = wf["edges"][:1]  # base_model missing
    run = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append).run(wf)
    assert run.state == "invalid"
    assert [e["type"] for e in events] == ["run_status"]
    assert events[0]["status"] == "invalid"
    assert events[0]["issues"][0]["code"] == "CFG-003"


def test_broken_event_callback_never_stops_a_run(tmp_path: Path) -> None:
    def boom(_: dict[str, Any]) -> None:
        raise RuntimeError("websocket closed")

    run = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=boom).run(_wf())
    assert run.state == "completed"


class _CancelOnTrain(FakeWorker):
    """Cancels the run when the train node starts, then behaves like a job stopped by SIGTERM."""

    def __init__(self, executor_ref: list[Executor]) -> None:
        super().__init__()
        self.executor_ref = executor_ref

    def run(self, job: NodeJob) -> NodeResult:
        if job.node_id == "t":
            self.calls.append(job.node_id)
            assert self.executor_ref[0].cancel("g-cancel")
            assert job.cancel.is_set()
            raise RuntimeError("terminated")
        return super().run(job)


def test_cancel_interrupts_without_error_and_skips_rest(tmp_path: Path) -> None:
    ref: list[Executor] = []
    ex = Executor(make_test_registry(), _CancelOnTrain(ref), tmp_path)
    ref.append(ex)
    run = ex.run(_wf(), graph_run_id="g-cancel")
    assert run.status["t"] == "interrupted"
    assert "t" not in run.errors
    assert not ex.cancel("g-cancel")  # run finished: nothing left to cancel


def test_resume_passes_checkpoint_step_to_worker(tmp_path: Path) -> None:
    Executor(make_test_registry(), FakeWorker(fail={"t": InterruptedByMemguard()}, checkpoints={"t"}), tmp_path).run(
        _wf()
    )
    seen: list[int | None] = []

    class Recorder(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                seen.append(job.resume_from_step)
            return super().run(job)

    run = Executor(make_test_registry(), Recorder(), tmp_path).run(_wf())
    assert run.status["t"] == "completed"
    assert seen == [1]


def test_not_resumable_run_dir_is_not_resumable_and_warns(tmp_path: Path) -> None:
    class NotResumable(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                (job.run_dir / NOT_RESUMABLE_MARKER).touch()
            return super().run(job)

    first = Executor(
        make_test_registry(), NotResumable(fail={"t": InterruptedByMemguard()}, checkpoints={"t"}), tmp_path
    ).run(_wf())
    assert first.errors["t"].resumable is False  # a checkpoint exists, but the worker said it cannot be resumed
    assert "t" not in first.warnings
    seen: list[int | None] = []

    class Recorder(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                seen.append(job.resume_from_step)
            return super().run(job)

    run = Executor(make_test_registry(), Recorder(), tmp_path).run(_wf())
    assert seen == [None]
    assert [w.code for w in run.warnings["t"]] == ["CACHE-001"]  # "this step runs again"
    assert run.status["t"] == "completed"


def test_missing_output_file_is_cache_warning_and_rerun(tmp_path: Path) -> None:
    ex = Executor(make_test_registry(), FakeWorker(), tmp_path)
    first = ex.run(_wf())
    Path(first.outputs["d"]["data"].path).unlink()  # artifact deleted behind the index's back
    worker = FakeWorker()
    run = Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert run.warnings["d"][0].code == "CACHE-001"
    assert worker.calls == ["d"]  # same content -> same hash -> downstream stays cached
    assert run.status["t"] == "cached"


def test_concurrent_runs_have_separate_cancel_flags(tmp_path: Path) -> None:
    ex = Executor(make_test_registry(), FakeWorker(), tmp_path)
    results: list[str] = []

    def go(gid: str) -> None:
        results.append(ex.run(_wf(), graph_run_id=gid).state)

    threads = [threading.Thread(target=go, args=(f"g{i}",)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results == ["completed"] * 4


def test_artifacts_are_leased_during_the_run_and_released_after(tmp_path: Path) -> None:
    """Retention must not evict d/m outputs while the long train node reads them (lease), and must be able to after."""
    from vlm_pipeline.artifacts.index import ArtifactIndex

    index = ArtifactIndex(tmp_path / "app.db")
    keys: dict[str, str] = {}
    leased_while_training: list[bool] = []

    def on_event(e: dict[str, Any]) -> None:
        if e.get("cache_key"):
            keys[e["node_id"]] = e["cache_key"]

    class Spy(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                leased_while_training.append(all(index.leased_by(keys[k]) for k in ("d", "m")))
            return super().run(job)

    Executor(make_test_registry(), Spy(), tmp_path, index=index, on_event=on_event).run(_wf(), graph_run_id="g-l")
    assert leased_while_training == [True]
    assert not any(index.leased_by(k) for k in keys.values())


def test_training_outputs_are_marked_as_trained_models(tmp_path: Path) -> None:
    from vlm_pipeline.graph.executor import produces_trained_model

    registry = make_test_registry()
    assert produces_trained_model(registry.get("test.train"))  # type: ignore[arg-type]
    assert not produces_trained_model(registry.get("test.dataset"))  # type: ignore[arg-type]


def test_long_chain_validates_without_recursion_error() -> None:
    """bao-cleancode review: a 3000-node chain must not hit RecursionError (would be misreported as CFG-007)."""
    from vlm_pipeline.graph import validate

    n = 3000
    nodes = [{"id": "m", "type": "test.model", "version": "1.0.0", "params": {}}]
    nodes += [{"id": f"q{i}", "type": "test.quant", "version": "1.0.0", "params": {}} for i in range(n)]
    edges = [{"from": ["m", "model"], "to": ["q0", "model"]}]
    edges += [{"from": [f"q{i}", "model"], "to": [f"q{i + 1}", "model"]} for i in range(n - 1)]
    wf = {"format": "vlm-workflow", "schema_version": 1, "name": "chain", "nodes": nodes, "edges": edges}
    assert [i.code for i in validate(wf, make_test_registry())] == []
    edges.append({"from": [f"q{n - 1}", "model"], "to": ["q0", "model"]})  # close a loop -> CFG-002, not CFG-007
    assert "CFG-002" in {i.code for i in validate(wf, make_test_registry())}


def test_one_executor_many_runs_each_with_its_own_event_sink(tmp_path: Path) -> None:
    """The app keeps ONE Executor (so cancel() finds the run) and gives every run its own WebSocket sink."""
    ex = Executor(make_test_registry(), FakeWorker(), tmp_path)
    a: list[dict[str, Any]] = []
    b: list[dict[str, Any]] = []
    ex.run(_wf(), graph_run_id="ga", on_event=a.append)
    ex.run(_wf(), graph_run_id="gb", on_event=b.append)
    assert a
    assert b
    assert {e["graph_run_id"] for e in a} == {"ga"}
    assert {e["graph_run_id"] for e in b} == {"gb"}


def test_workflow_file_model_matches_validator_and_round_trips() -> None:
    """graph.workflow models are the single source: a valid workflow parses, unknown keys are rejected by both."""
    import pytest
    from pydantic import ValidationError

    from vlm_pipeline.graph import validate
    from vlm_pipeline.graph.workflow import WorkflowFile

    wf = _wf()
    wf["nodes"][0]["ui"] = {"x": 1, "y": 2}
    parsed = WorkflowFile.model_validate(wf)
    assert parsed.model_dump(by_alias=True, exclude_none=True)["edges"][0]["from"] == ("d", "data")
    assert validate(wf, make_test_registry()) == []
    for where, bad in (("top", {**wf, "description": "x"}), ("edge", {**wf, "edges": [{**wf["edges"][0], "ui": {}}]})):
        with pytest.raises(ValidationError):
            WorkflowFile.model_validate(bad)
        assert any(i.code == "CFG-007" for i in validate(bad, make_test_registry())), where


def test_checkpoint_marker_formats(tmp_path: Path) -> None:
    """FakeWorker writes a text step; NeMo AutoModel symlinks LATEST -> epoch_<E>_step_<S> (bao-backend)."""
    from vlm_pipeline.graph.worker import read_checkpoint_step

    text = tmp_path / "text"
    (text / "checkpoints").mkdir(parents=True)
    (text / "checkpoints" / "LATEST").write_text("7")
    assert read_checkpoint_step(text) == 7

    am = tmp_path / "automodel"
    ckpts = am / "checkpoints"
    (ckpts / "epoch_0_step_19").mkdir(parents=True)
    (ckpts / "LATEST").symlink_to("epoch_0_step_19", target_is_directory=True)
    assert read_checkpoint_step(am) == 19
    (ckpts / "epoch_0_step_19" / ".incomplete").touch()  # save still in progress -> not resumable
    assert read_checkpoint_step(am) is None
    assert read_checkpoint_step(tmp_path / "none") is None


class _RealishWorker(FakeWorker):
    """Same outputs as FakeWorker but a different identity, like the container worker of a real run."""

    def identity(self, spec: Any) -> str:
        return f"container/vlm-automodel@sha256:{'8' * 64}/{spec.type_id}"


def test_fake_artifacts_never_satisfy_a_real_run(tmp_path: Path) -> None:
    """M1 E2E bug (bao-backend): FakeWorker outputs in a shared cache were reused by a RealWorker run."""
    Executor(make_test_registry(), FakeWorker(), tmp_path).run(_wf())  # demo run fills the cache
    real = _RealishWorker()
    run = Executor(make_test_registry(), real, tmp_path).run(_wf())
    assert sorted(real.calls) == ["d", "m", "t"]  # nothing taken from the fake run
    assert "cached" not in set(run.status.values())
    again = _RealishWorker()
    assert set(Executor(make_test_registry(), again, tmp_path).run(_wf()).status.values()) == {"cached"}
    assert again.calls == []  # real artifacts do cache for real runs


def test_runtime_image_is_part_of_the_cache_key(tmp_path: Path) -> None:
    class NewImage(_RealishWorker):
        def identity(self, spec: Any) -> str:
            return f"container/vlm-automodel@sha256:{'9' * 64}/{spec.type_id}"

    Executor(make_test_registry(), _RealishWorker(), tmp_path).run(_wf())
    worker = NewImage()
    Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert sorted(worker.calls) == ["d", "m", "t"]  # new image -> recompute


def test_every_node_run_dir_has_a_run_json_linked_to_the_graph_run(tmp_path: Path) -> None:
    """Dashboard + retention read run.json per node run dir (bao-cleanup); the cache index points at that dir."""
    from vlm_pipeline.artifacts.index import ArtifactIndex
    from vlm_pipeline.common.runinfo import RunStatus, read_manifest

    index = ArtifactIndex(tmp_path / "app.db")
    keys: dict[str, str] = {}
    run = Executor(make_test_registry(), FakeWorker(), tmp_path, index=index, on_event=lambda e: (
        keys.__setitem__(e["node_id"], e["cache_key"]) if e.get("cache_key") else None
    )).run(_wf(), graph_run_id="g-m")  # fmt: skip
    for nid in ("d", "m", "t"):
        entry = index.lookup(keys[nid])
        assert entry is not None
        run_dir = Path(run.outputs[nid][next(iter(run.outputs[nid]))].path).parent
        assert entry.run_id == run_dir.name
        assert entry.run_dir == str(run_dir.resolve())
        manifest = read_manifest(run_dir)
        assert manifest.run_id == run_dir.name
        assert manifest.parent_run_id == "g-m"
        assert manifest.status == RunStatus.COMPLETED
    assert read_manifest(run_dir).stage == "train"


def test_failed_node_run_json_carries_the_error_code(tmp_path: Path) -> None:
    from vlm_pipeline.common.runinfo import RunStatus, read_manifest
    from vlm_pipeline.graph.errors import OutOfMemoryUMA

    Executor(make_test_registry(), FakeWorker(fail={"t": OutOfMemoryUMA()}), tmp_path).run(_wf())
    [t_dir] = list((tmp_path / "runs").glob("t-*"))
    manifest = read_manifest(t_dir)
    assert manifest.status == RunStatus.FAILED
    assert manifest.error_code == "MEM-003"


def test_container_path_outputs_are_rejected(tmp_path: Path) -> None:
    """A worker returning /workspace/... (container path) must not poison the cache index (bao-cleanup)."""
    from vlm_pipeline.graph.worker import ArtifactRef

    class ContainerPaths(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            res = super().run(job)
            return NodeResult(
                {p: ArtifactRef(r.type, "/workspace/outputs/graph/runs/x/" + p, r.hash) for p, r in res.outputs.items()}
            )

    index_db = tmp_path / "app.db"
    run = Executor(make_test_registry(), ContainerPaths(), tmp_path).run(_wf())
    assert run.status["d"] == "failed"
    assert run.errors["d"].code == "SYS-000"
    assert "non-host" in run.errors["d"].details or "host path" in run.errors["d"].details
    assert index_db.exists()


def test_memory_wait_is_reported_as_node_state(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    class Waits(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                assert job.on_wait is not None
                job.on_wait({"need_gb": 22.0, "avail_gb": 9.5, "waited_s": 30.0, "since": 1.0})
            return super().run(job)

    Executor(make_test_registry(), Waits(), tmp_path, on_event=events.append).run(_wf())
    [state] = [e for e in events if e["type"] == "node_state"]
    assert {k: state[k] for k in ("node_id", "state", "need_gb", "avail_gb", "since")} == {
        "node_id": "t", "state": "waiting_memory", "need_gb": 22.0, "avail_gb": 9.5, "since": 1.0,
    }  # fmt: skip
