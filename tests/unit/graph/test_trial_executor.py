"""MODEL-005 quick trial: the step count is a run option the executor carries, with its own cache key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.graph.testing import FakeWorker, make_test_registry
from vlm_pipeline.graph.worker import NodeJob, NodeResult


def _wf() -> dict[str, Any]:
    def n(i: str, t: str, **p: Any) -> dict[str, Any]:
        return {"id": i, "type": t, "version": "1.0.0", "params": p}

    return {"format": "vlm-workflow", "schema_version": 1, "name": "t",
            "nodes": [n("d", "test.dataset"), n("m", "test.model"), n("t", "test.train", max_steps=10)],
            "edges": [{"from": ["d", "data"], "to": ["t", "data"]},
                      {"from": ["m", "model"], "to": ["t", "base_model"]}]}  # fmt: skip


class _Jobs(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.trials: dict[str, int | None] = {}

    def run(self, job: NodeJob) -> NodeResult:
        self.trials[job.node_id] = job.trial_max_steps
        return super().run(job)


def test_a_trial_reaches_only_its_step_and_never_stands_for_a_real_run(tmp_path: Path) -> None:
    keys: dict[str, str] = {}

    def on_event(e: dict[str, Any]) -> None:
        if e.get("cache_key"):
            keys[e["node_id"]] = e["cache_key"]

    worker = _Jobs()
    Executor(make_test_registry(), worker, tmp_path, on_event=on_event).run(_wf(), trial_max_steps={"t": 5})
    assert worker.trials == {"d": None, "m": None, "t": 5}
    trial_keys = dict(keys)
    real = _Jobs()
    Executor(make_test_registry(), real, tmp_path, on_event=on_event).run(_wf())
    assert real.trials == {"t": None}  # data + model reused from the trial; the 5-step train is not
    assert keys["t"] != trial_keys["t"]
    assert keys["d"] == trial_keys["d"]
