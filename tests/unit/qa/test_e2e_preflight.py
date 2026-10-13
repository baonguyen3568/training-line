from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.runinfo import RunManifest, write_manifest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "e2e_preflight.py"


def _pre() -> Any:
    spec = importlib.util.spec_from_file_location("e2e_preflight", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_is_16_or_the_holders_abort_plus_4() -> None:
    pre = _pre()
    assert pre.required_gb(None) == 16.0
    assert pre.required_gb(8.0) == 16.0
    assert pre.required_gb(14.5) == 18.5


def test_holder_abort_level_comes_from_its_run_json(tmp_path: Path) -> None:
    pre = _pre()
    manifest = RunManifest(run_id="20261013-1015-r1", name="t", stage="train", task="sft", memguard_abort_gb=14.5)
    write_manifest(tmp_path / "runs" / "20261013-1015-r1", manifest)
    status = {"held": True, "who": "bao-backend", "job": "train", "run_id": "20261013-1015-r1"}
    assert pre.holder_abort_gb(status, tmp_path) == (14.5, "bao-backend (train)")


@pytest.mark.parametrize("where", ["graph/runs/n4-0123456789abcdef", "graph/runs/n5-abcdef0123456789/jobs/model"])
def test_graph_node_jobs_are_found_by_their_run_id(tmp_path: Path, where: str) -> None:
    """App runs: the lock holds a node run id, or "<node run>-<job>" for an eval/bench container (tester 2026-10-13)."""
    pre = _pre()
    run_id = "n4-0123456789abcdef" if where.count("/") == 2 else "n5-abcdef0123456789-model"
    write_manifest(
        tmp_path / where, RunManifest(run_id=run_id, name="t", stage="eval", task="eval", memguard_abort_gb=9)
    )
    write_manifest(
        tmp_path / "graph/runs/n5-abcdef0123456789/jobs/base",
        RunManifest(run_id="n5-abcdef0123456789-base", name="t", stage="eval", task="eval", memguard_abort_gb=30),
    )
    status = {"held": True, "who": "vlm-app", "job": "j", "run_id": run_id}
    assert pre.holder_abort_gb(status, tmp_path) == (9.0, "vlm-app (j)")


def test_a_run_id_cannot_walk_out_of_outputs(tmp_path: Path) -> None:
    write_manifest(tmp_path / "x", RunManifest(run_id="x", name="t", stage="eval", task="eval", memguard_abort_gb=1))
    status = {"held": True, "who": "w", "job": "j", "run_id": "../x"}
    assert _pre().holder_abort_gb(status, tmp_path / "outputs") == (None, "w (j)")


def test_free_lock_or_unknown_run_falls_back_to_the_floor(tmp_path: Path) -> None:
    pre = _pre()
    assert pre.holder_abort_gb({"held": False}, tmp_path) == (None, "")
    level, who = pre.holder_abort_gb({"held": True, "who": "x", "job": "j", "run_id": "missing"}, tmp_path)
    assert level is None
    assert who == "x (j)"


def test_unknown_level_of_a_running_job_assumes_the_worst_case() -> None:
    assert _pre().worst_case_abort_gb(121.7) == pytest.approx(0.2 * 121.7)  # A0 can be at most MemTotal
