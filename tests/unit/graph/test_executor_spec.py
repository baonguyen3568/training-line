"""Spec tests for the graph executor with a fake worker (P009 §5). CPU only, no containers.

ASSUMED INTERFACE (draft — agree with bao-backend):
- ``vlm_pipeline.graph.testing.FakeWorker(fail: dict[node_id, BaseException] | None = None)``;
  records ``calls: list[str]`` (node ids in execution order); produces deterministic artifact hashes.
- ``vlm_pipeline.graph.executor.Executor(registry, worker, state_dir: Path)``;
  ``run(workflow: dict) -> GraphRun`` with ``status: dict[node_id, str]`` (pending|queued|running|completed|
  failed|interrupted|cached|skipped) and ``errors: dict[node_id, UserError]`` (``code``, ``params``, ``details`` — traceback only in
  ``details``; user text resolved from i18n keys at the API layer, like ``Issue``).
- ``vlm_pipeline.graph.testing.ContainerExit(exit_code, oom_killed=False, events=())``: raised by FakeWorker to
  simulate a container job ending (P008). Classification order (bao-fixcode): exit 75 → MEM-002 (interrupted);
  exit 137 + OOMKilled → MEM-003; worker error event with a catalog code → that code; non-zero exit without
  event → SYS-003; unknown exception → SYS-000.
- ``vlm_pipeline.graph.errors``: ``InterruptedByMemguard`` (exit 75) and catalog exceptions
  (``OutOfMemoryUMA`` → ``MEM-003``, memguard pause → ``MEM-002``); anything unknown → ``SYS-000``.
  Codes = ``src/vlm_pipeline/common/error_catalog.yaml``; display rules = P010 §6–§7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.graph.executor (P009) in P2"
)


def _wf() -> dict[str, Any]:
    # d ─┐
    #    ├─► t ─► (adapter)       and independent branch  m2 ─► q
    # m ─┘
    def n(i: str, t: str, **p: Any) -> dict[str, Any]:
        return {"id": i, "type": t, "version": "1.0.0", "params": p}

    return {
        "format": "vlm-workflow",
        "schema_version": 1,
        "name": "t",
        "nodes": [n("d", "test.dataset"), n("m", "test.model"), n("t", "test.train", max_steps=10),
                  n("m2", "test.model"), n("q", "test.quant")],
        "edges": [{"from": ["d", "data"], "to": ["t", "data"]}, {"from": ["m", "model"], "to": ["t", "base_model"]},
                  {"from": ["m2", "model"], "to": ["q", "model"]}],
    }


def _executor(tmp_path: Path, **worker_kw: Any) -> tuple[Any, Any]:
    from vlm_pipeline.graph.executor import Executor
    from vlm_pipeline.graph.testing import FakeWorker, make_test_registry

    worker = FakeWorker(**worker_kw)
    return Executor(make_test_registry(), worker, tmp_path), worker


def test_topological_order(tmp_path: Path) -> None:
    ex, worker = _executor(tmp_path)
    run = ex.run(_wf())
    assert set(run.status.values()) == {"completed"}
    assert worker.calls.index("t") > worker.calls.index("d")
    assert worker.calls.index("t") > worker.calls.index("m")
    assert worker.calls.index("q") > worker.calls.index("m2")


def test_invalid_graph_never_executes(tmp_path: Path) -> None:
    wf = _wf()
    wf["edges"].append({"from": ["d", "data"], "to": ["q", "model"]})  # type mismatch
    ex, worker = _executor(tmp_path)
    ex.run(wf)
    assert worker.calls == []


def test_second_run_is_fully_cached(tmp_path: Path) -> None:
    ex, worker = _executor(tmp_path)
    ex.run(_wf())
    n_first = len(worker.calls)
    run2 = ex.run(_wf())
    assert len(worker.calls) == n_first
    assert set(run2.status.values()) == {"cached"}


def test_param_change_reruns_only_node_and_downstream(tmp_path: Path) -> None:
    ex, worker = _executor(tmp_path)
    ex.run(_wf())
    worker.calls.clear()
    wf = _wf()
    wf["nodes"][2]["params"]["max_steps"] = 20  # node t
    ex.run(wf)
    assert worker.calls == ["t"]


def test_upstream_change_invalidates_downstream(tmp_path: Path) -> None:
    ex, worker = _executor(tmp_path)
    ex.run(_wf())
    worker.calls.clear()
    wf = _wf()
    wf["nodes"][3]["params"]["revision"] = "f" * 40  # node m2
    ex.run(wf)
    assert sorted(worker.calls) == ["m2", "q"]


def test_failure_isolated_to_its_branch(tmp_path: Path) -> None:
    ex, _ = _executor(tmp_path, fail={"m": RuntimeError("boom")})
    run = ex.run(_wf())
    assert run.status["m"] == "failed"
    assert run.status["t"] == "skipped"
    assert run.status["q"] == "completed"


def test_resume_after_interrupt_does_not_rerun_completed(tmp_path: Path) -> None:
    from vlm_pipeline.graph.errors import InterruptedByMemguard

    ex, worker = _executor(tmp_path, fail={"t": InterruptedByMemguard()})
    run = ex.run(_wf())
    assert run.status["t"] == "interrupted"  # UI shows "⏸ Tạm dừng" + persistent banner, not a toast (P010)
    assert run.errors["t"].code == "MEM-002"

    # New process: fresh executor on the same state dir, worker no longer fails.
    ex2, worker2 = _executor(tmp_path)
    run2 = ex2.run(_wf())
    assert worker2.calls == ["t"]
    assert run2.status["t"] == "completed"


def test_known_error_maps_to_catalog_code_without_traceback(tmp_path: Path) -> None:
    from vlm_pipeline.graph.errors import OutOfMemoryUMA

    ex, _ = _executor(tmp_path, fail={"t": OutOfMemoryUMA()})
    err = ex.run(_wf()).errors["t"]
    assert err.code == "MEM-003"
    assert "Traceback" not in repr(err.params)


def test_unknown_error_is_unclassified_and_hides_internals(tmp_path: Path) -> None:
    secret = "hf_" + "x" * 34
    ex, _ = _executor(tmp_path, fail={"t": ValueError(f"/home/ntcai/secret/path token={secret}")})
    err = ex.run(_wf()).errors["t"]
    assert err.code == "SYS-000"
    assert "/home/" not in repr(err.params)
    assert secret not in repr(err.params)
    assert secret not in (err.details or "")  # details are shown on demand: still must be redacted


def test_moving_nodes_on_canvas_keeps_cache(tmp_path: Path) -> None:
    ex, worker = _executor(tmp_path)
    ex.run(_wf())
    worker.calls.clear()
    wf = _wf()
    for i, node in enumerate(wf["nodes"]):
        node["ui"] = {"x": 100 * i, "y": 50}
    run = ex.run(wf)
    assert worker.calls == []
    assert set(run.status.values()) == {"cached"}


@pytest.mark.parametrize(
    ("exit_kwargs", "expected_code", "expected_status"),
    [
        ({"exit_code": 75}, "MEM-002", "interrupted"),
        ({"exit_code": 137, "oom_killed": True}, "MEM-003", "failed"),
        ({"exit_code": 1, "events": ({"type": "error", "code": "DATA-001", "params": {"idx": 3}},)}, "DATA-001", "failed"),
        ({"exit_code": 1}, "SYS-003", "failed"),
        ({"exit_code": 137, "oom_killed": False}, "SYS-003", "failed"),  # SIGKILL but not OOM: do not guess OOM
    ],
)
def test_container_exit_classification(
    tmp_path: Path, exit_kwargs: dict[str, Any], expected_code: str, expected_status: str
) -> None:
    from vlm_pipeline.graph.testing import ContainerExit

    ex, _ = _executor(tmp_path, fail={"t": ContainerExit(**exit_kwargs)})
    run = ex.run(_wf())
    assert run.status["t"] == expected_status
    assert run.errors["t"].code == expected_code
    assert run.status["q"] == "completed"  # independent branch unaffected
