"""Executor behaviour beyond the spec: events, cancel, resume step, broken UI callback (CPU, FakeWorker)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vlm_pipeline.graph.errors import InterruptedByMemguard
from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.graph.testing import FakeWorker, make_test_registry
from vlm_pipeline.graph.worker import NodeJob, NodeResult


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
