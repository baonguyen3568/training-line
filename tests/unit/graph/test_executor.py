"""Executor behaviour beyond the spec: events, cancel, resume step, broken UI callback (CPU, FakeWorker)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.graph import executor as executor_mod
from vlm_pipeline.graph.errors import InterruptedByMemguard
from vlm_pipeline.graph.executor import Executor, _save_warnings, _saved_warnings
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


def test_a_completed_node_warning_reaches_the_events(tmp_path: Path) -> None:
    """QNT-002 style: the node completes, its output is used, and the card shows the worker's catalog warning."""

    class Warns(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            result = super().run(job)
            if job.node_id != "t":
                return result
            return NodeResult(outputs=result.outputs, warnings=(("QNT-002", {"node": "t", "reason": "top1"}),))

    events: list[dict[str, Any]] = []
    run = Executor(make_test_registry(), Warns(), tmp_path, on_event=events.append).run(_wf())
    assert run.status["t"] == "completed"
    [warning] = [e for e in events if e.get("type") == "warning"]
    assert warning["node_id"] == "t"
    assert (warning["error"]["code"], warning["error"]["params"]["reason"]) == ("QNT-002", "top1")
    assert "actions" in warning["error"]


def test_a_cached_node_shows_its_warning_again(tmp_path: Path) -> None:
    """QNT-002 on a reused quantized model: the result did not change, so neither did the warning."""

    class Warns(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            result = super().run(job)
            if job.node_id != "t":
                return result
            return NodeResult(outputs=result.outputs, warnings=(("QNT-002", {"node": "t", "reason": "top1"}),))

    Executor(make_test_registry(), Warns(), tmp_path).run(_wf())
    events: list[dict[str, Any]] = []
    again = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append).run(_wf())
    assert again.status["t"] == "cached"
    [warning] = [e for e in events if e.get("type") == "warning"]
    assert (warning["node_id"], warning["error"]["code"], warning["error"]["params"]["reason"]) == (
        "t",
        "QNT-002",
        "top1",
    )
    assert [w.code for w in again.warnings["t"]] == ["QNT-002"]


def test_a_cached_node_without_warnings_reports_none(tmp_path: Path) -> None:
    Executor(make_test_registry(), FakeWorker(), tmp_path).run(_wf())
    events: list[dict[str, Any]] = []
    again = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append).run(_wf())
    assert set(again.status.values()) == {"cached"}
    assert not [e for e in events if e.get("type") == "warning"]


def test_kept_warnings_live_outside_the_run_dir_and_only_catalog_codes_return(tmp_path: Path) -> None:
    """fixcode B1: a job writes in its run dir, so nothing it plants there can surface as a warning."""

    class Plants(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            (job.run_dir / "node_warnings.json").write_text(json.dumps([{"code": "QNT-002", "params": {}}]))
            return super().run(job)

    Executor(make_test_registry(), Plants(), tmp_path).run(_wf())
    events: list[dict[str, Any]] = []
    again = Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append).run(_wf())
    assert set(again.status.values()) == {"cached"}
    assert not [e for e in events if e.get("type") == "warning"]
    for kept in (tmp_path / executor_mod.WARNINGS_DIR).glob("*.json"):
        kept.write_text(json.dumps([{"code": "NOT-A-CODE", "params": {}}, {"code": "SYS-000", "params": {}}]))
    events.clear()
    Executor(make_test_registry(), FakeWorker(), tmp_path, on_event=events.append).run(_wf())
    assert not [e for e in events if e.get("type") == "warning"]  # unknown and error-severity codes are dropped


def test_a_fresh_result_without_warnings_clears_the_kept_ones(tmp_path: Path) -> None:
    _save_warnings(tmp_path, "k", (("QNT-002", {"node": "t", "reason": "top1"}),))
    assert [w.code for w in _saved_warnings(tmp_path, "k")] == ["QNT-002"]
    _save_warnings(tmp_path, "k", ())
    assert _saved_warnings(tmp_path, "k") == []


def _load_oom_worker(fail_times: int, **manifest: Any) -> Any:
    """A train job that dies with a CUDA allocation failure while loading, the first ``fail_times`` tries."""
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest
    from vlm_pipeline.graph.errors import ContainerExit

    class LoadOom(FakeWorker):
        tries = 0

        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id != "t":
                return super().run(job)
            LoadOom.tries += 1
            if LoadOom.tries > fail_times:
                return super().run(job)
            fields = {"peak_mem_gb": 3.0, "est_peak_gb": 40.0, "container_started_at": 100.0, "ended_at": 160.0,
                      "error_code": "MEM-003", "oom_killed": False, **manifest}  # fmt: skip
            write_manifest(job.run_dir, RunManifest(run_id=job.run_dir.name, name="t", stage="train", task="lora",
                                                    status=RunStatus.FAILED, **fields))  # fmt: skip
            raise ContainerExit(1, log_tail="torch.AcceleratorError: CUDA error: out of memory")

    return LoadOom()


def test_a_cuda_allocation_failure_at_load_is_retried_once(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(executor_mod, "ALLOC_RETRY_WAIT_S", 0.0)
    marked: list[dict[str, Any]] = []  # the failed try's run.json as marked (tests reuse the dir: no workspace)
    mark = executor_mod._mark_retried

    def record(run_dir: Path, reason: str) -> None:
        mark(run_dir, reason)
        marked.append(json.loads((run_dir / "run.json").read_text()))

    monkeypatch.setattr(executor_mod, "_mark_retried", record)
    worker = _load_oom_worker(fail_times=1)
    events: list[dict[str, Any]] = []
    run = Executor(make_test_registry(), worker, tmp_path, on_event=events.append).run(_wf())
    assert run.status["t"] == "completed"
    assert type(worker).tries == 2
    [pause] = [e for e in events if e.get("reason") == executor_mod.ALLOC_RETRY_REASON]
    assert (pause["type"], pause["state"], pause["node_id"]) == ("node_state", "waiting_memory", "t")
    # frontend: the retried try never reports "failed" (no error card for something the app is handling)
    seen = [e.get("status") or e.get("reason") for e in events if e.get("node_id") == "t" and e["type"] != "warning"]
    assert seen == ["running", executor_mod.ALLOC_RETRY_REASON, "running", "completed"]
    assert [(m["status"], m["retried_after"]) for m in marked] == [("failed", "gpu_alloc_transient")]


def test_a_second_allocation_failure_stands(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(executor_mod, "ALLOC_RETRY_WAIT_S", 0.0)
    worker = _load_oom_worker(fail_times=5)
    run = Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert run.status["t"] == "failed"
    assert run.errors["t"].code == "MEM-003"
    assert type(worker).tries == 2  # one retry, never a loop


@pytest.mark.parametrize(
    "manifest",
    [
        {"oom_killed": True},  # the kernel killed it: real host pressure
        {"peak_mem_gb": 30.0},  # reached most of its working set: a real "too big" OOM
        {"peak_mem_gb": None},  # unknown: fail closed
        {"est_peak_gb": None},
        {"ended_at": 100.0 + 600},  # ran 10 minutes: not a load failure
        {"error_code": "SYS-003"},
    ],
)
def test_other_failures_are_never_retried(tmp_path: Path, monkeypatch: Any, manifest: dict[str, Any]) -> None:
    monkeypatch.setattr(executor_mod, "ALLOC_RETRY_WAIT_S", 0.0)
    worker = _load_oom_worker(fail_times=5, **manifest)
    run = Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert run.status["t"] == "failed"
    assert type(worker).tries == 1


def test_a_load_failure_after_training_started_is_not_retried(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(executor_mod, "ALLOC_RETRY_WAIT_S", 0.0)
    worker = _load_oom_worker(fail_times=5)
    real_run = type(worker).run

    def run_with_metrics(self: Any, job: NodeJob) -> NodeResult:
        if job.node_id == "t":
            (job.run_dir / "metrics.jsonl").write_text('{"step": 1}\n')
        return real_run(self, job)

    monkeypatch.setattr(type(worker), "run", run_with_metrics)
    assert Executor(make_test_registry(), worker, tmp_path).run(_wf()).status["t"] == "failed"
    assert type(worker).tries == 1


def test_a_cancel_during_the_pause_ends_the_node(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(executor_mod, "ALLOC_RETRY_WAIT_S", 30.0)
    worker = _load_oom_worker(fail_times=1)
    ex = Executor(make_test_registry(), worker, tmp_path)
    started = time.time()
    holder: dict[str, Any] = {}

    def on_event(e: dict[str, Any]) -> None:
        if e.get("reason") == executor_mod.ALLOC_RETRY_REASON:
            ex.cancel(e["graph_run_id"])

    ex.on_event = on_event
    holder["run"] = ex.run(_wf())
    assert time.time() - started < 10
    assert holder["run"].status["t"] in ("failed", "interrupted", "cancelled")
    assert type(worker).tries == 1


def test_a_rerun_host_node_does_not_keep_an_earlier_tries_failure(tmp_path: Path) -> None:
    """E2E 2026-10-14: model.hf_pick failed yesterday (NET-001), succeeded today in the same dir, still read failed."""
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest, write_manifest
    from vlm_pipeline.graph.executor import _write_node_manifest
    from vlm_pipeline.nodes import REGISTRY

    spec = REGISTRY.get("model.hf_pick")
    assert spec is not None
    old = RunManifest(run_id=tmp_path.name, name="m", stage="data", task="model.hf_pick", status=RunStatus.FAILED,
                      error_code="NET-001", started_at=1000.0, ended_at=1001.0, tags=["graph:g-old"])  # fmt: skip
    write_manifest(tmp_path, old)
    _write_node_manifest(tmp_path, node_id="m", spec=spec, gid="g-new", status=RunStatus.COMPLETED, error_code=None,
                         started=2000.0)  # fmt: skip
    now = read_manifest(tmp_path)
    assert (now.status, now.error_code, now.parent_run_id) == (RunStatus.COMPLETED, None, "g-new")
    assert now.started_at == 2000.0
    assert "graph:g-old" not in now.tags

    # A manifest the worker wrote during this try (a container job via jobrun) keeps its own details.
    mine = old.model_copy(update={"started_at": 2001.0, "status": RunStatus.FAILED, "error_code": "MEM-003"})
    write_manifest(tmp_path, mine)
    _write_node_manifest(tmp_path, node_id="m", spec=spec, gid="g-new", status=RunStatus.FAILED, error_code="X",
                         started=2000.0)  # fmt: skip
    kept = read_manifest(tmp_path)
    assert (kept.status, kept.error_code) == (RunStatus.FAILED, "MEM-003")
    assert "graph:g-new" in kept.tags


def test_rerunning_a_failed_node_in_its_old_dir_reports_the_new_success(tmp_path: Path) -> None:
    """PM 2026-10-14: a user who re-runs a node that failed yesterday must not keep seeing yesterday's error."""
    from vlm_pipeline.common.runinfo import read_manifest

    class FailsOnce(FakeWorker):
        tries = 0

        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "m":
                FailsOnce.tries += 1
                if FailsOnce.tries == 1:
                    raise RuntimeError("hub unreachable")
            return super().run(job)

    worker = FailsOnce()
    first = Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert first.status["m"] == "failed"
    [run_dir] = list((tmp_path / "runs").glob("m-*"))
    assert read_manifest(run_dir).status.value == "failed"
    time.sleep(0.01)  # the second try starts strictly later
    second = Executor(make_test_registry(), worker, tmp_path).run(_wf())
    assert second.status["m"] == "completed"
    assert list((tmp_path / "runs").glob("m-*")) == [run_dir]  # the same dir was reused (no workspace claims)
    now = read_manifest(run_dir)
    assert (now.status.value, now.error_code) == ("completed", None)


def test_an_unclassified_failure_logs_its_traceback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Slot #2 2026-10-14: a node failed SYS-000 and neither the app log nor run.json said why."""

    class Broken(FakeWorker):
        def run(self, job: NodeJob) -> NodeResult:
            if job.node_id == "t":
                raise RuntimeError("runner said inputs_nested")
            return super().run(job)

    with caplog.at_level("ERROR", logger="vlm_pipeline.graph.executor"):
        run = Executor(make_test_registry(), Broken(), tmp_path).run(_wf())
    assert run.status["t"] == "failed"
    assert run.errors["t"].code == "SYS-000"  # the UI: the catalog error only
    assert "node t failed (SYS-000)" in caplog.text
    assert "runner said inputs_nested" in caplog.text  # the log: the cause
