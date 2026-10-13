"""P017: one job per run dir; a retry runs in a new attempt dir and resumes from a host-chosen checkpoint (C1-C3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.run_claims import CLAIMS_REL, claim_run
from vlm_pipeline.common.runinfo import RunManifest, write_manifest
from vlm_pipeline.graph.errors import InterruptedByMemguard
from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.graph.testing import FakeWorker, make_test_registry
from vlm_pipeline.graph.worker import MAX_ATTEMPTS, NodeJob, NodeResult

DIGEST = "sha256:" + "ab" * 32


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


class Recorder(FakeWorker):
    """FakeWorker that remembers the train node's job and, like run_job, claims its run dir."""

    def __init__(self, root: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self.root = root
        self.jobs: list[NodeJob] = []

    def identity(self, spec: Any) -> str:
        return f"container/vlm-x:1@{DIGEST}"

    def run(self, job: NodeJob) -> NodeResult:
        if job.node_id == "t":
            self.jobs.append(job)
            claim_run(self.root, job.run_dir.name, by="test")
        return super().run(job)


def _executor(root: Path, worker: FakeWorker) -> Executor:
    return Executor(make_test_registry(), worker, root / "outputs" / "graph", workspace=root)


def _ckpt(run_dir: Path, step: int, *, digest: str = DIGEST) -> Path:
    ckpt = run_dir / "checkpoints" / f"epoch_0_step_{step}"
    (ckpt / "model").mkdir(parents=True)
    (ckpt / "model" / "weights.safetensors").write_bytes(b"w")
    write_manifest(run_dir, RunManifest(run_id=run_dir.name, name="t", stage="train", task="sft",
                                        container_digest=digest))  # fmt: skip
    return ckpt


def _failed_once(root: Path) -> Path:
    """A first try of node t that a container job ran and that stopped; returns its run dir."""
    worker = Recorder(root, fail={"t": InterruptedByMemguard()})
    _executor(root, worker).run(_wf())
    return worker.jobs[0].run_dir


def test_a_retry_gets_a_new_attempt_dir_and_resumes_from_the_checkpoint(tmp_path: Path) -> None:
    first = _failed_once(tmp_path)
    ckpt = _ckpt(first, 9)
    worker = Recorder(tmp_path)
    run = _executor(tmp_path, worker).run(_wf())
    assert run.status["t"] == "completed"
    job = worker.jobs[0]
    assert job.run_dir.name == f"{first.name}-a2"  # never the dir the first job wrote
    assert (job.resume_from, job.resume_from_step) == (ckpt, 9)


def test_a_dir_from_before_claims_counts_as_used(tmp_path: Path) -> None:
    first = _failed_once(tmp_path)
    for claim in (tmp_path / CLAIMS_REL).iterdir():
        claim.unlink()  # a legacy dir: no claim, but a container job wrote its logs there
    (first / "logs").mkdir(exist_ok=True)
    worker = Recorder(tmp_path)
    _executor(tmp_path, worker).run(_wf())
    assert worker.jobs[0].run_dir.name == f"{first.name}-a2"


@pytest.mark.parametrize("plant", ["latest_symlink", "symlink_inside", "incomplete", "other_image", "not_resumable"])
def test_a_checkpoint_the_host_cannot_trust_is_not_resumed(tmp_path: Path, plant: str) -> None:
    first = _failed_once(tmp_path)
    secret = tmp_path / "secret"
    secret.mkdir()
    if plant == "latest_symlink":  # C2: a pointer the job wrote is never followed
        (first / "checkpoints").mkdir(parents=True)
        (first / "checkpoints" / "LATEST").symlink_to(secret)
    elif plant == "symlink_inside":  # C3: the next container would follow it
        ckpt = _ckpt(first, 9)
        (ckpt / "model" / "leak").symlink_to(secret)
    elif plant == "incomplete":
        ckpt = _ckpt(first, 9)
        (ckpt / ".incomplete").touch()
    elif plant == "other_image":  # C1
        _ckpt(first, 9, digest="sha256:" + "cd" * 32)
    else:
        _ckpt(first, 9)
        (first / ".not-resumable").touch()
    worker = Recorder(tmp_path)
    _executor(tmp_path, worker).run(_wf())
    assert worker.jobs[0].resume_from is None
    assert worker.jobs[0].run_dir != first


def test_the_newest_complete_checkpoint_wins(tmp_path: Path) -> None:
    first = _failed_once(tmp_path)
    _ckpt(first, 9)
    newer = first / "checkpoints" / "epoch_1_step_19"
    (newer / "model").mkdir(parents=True)
    (newer / "model" / "w").write_bytes(b"w")
    worker = Recorder(tmp_path)
    _executor(tmp_path, worker).run(_wf())
    assert worker.jobs[0].resume_from_step == 19


def test_an_eleventh_try_is_a_clear_error(tmp_path: Path) -> None:
    first = _failed_once(tmp_path)
    for n in range(2, MAX_ATTEMPTS + 1):
        claim_run(tmp_path, f"{first.name}-a{n}", by="test")
    worker = Recorder(tmp_path)
    run = _executor(tmp_path, worker).run(_wf())
    assert worker.jobs == []
    assert run.errors["t"].code == "SYS-005"
    assert run.errors["t"].params == {"node": "t", "attempt_count": MAX_ATTEMPTS}


def test_jobs_of_a_node_mark_its_dir_used(tmp_path: Path) -> None:
    """eval/bench claim ``<node dir>-model``: their node dir is used too (a later attempt is not)."""
    from vlm_pipeline.common.run_claims import node_dir_used

    claim_run(tmp_path, "n5-0123456789abcdef-model", by="test")
    assert node_dir_used(tmp_path, "n5-0123456789abcdef")
    claim_run(tmp_path, "n6-0123456789abcdef-a2", by="test")
    assert not node_dir_used(tmp_path, "n6-0123456789abcdef")
    assert not os.path.lexists(tmp_path / CLAIMS_REL / "n6-0123456789abcdef")


def test_an_attempt_that_never_saved_falls_back_to_the_one_before(tmp_path: Path) -> None:
    """cleanup Q2: attempt 2 crashed before its first checkpoint; attempt 3 resumes from attempt 1."""
    first = _failed_once(tmp_path)
    ckpt = _ckpt(first, 9)
    second = Recorder(tmp_path, fail={"t": InterruptedByMemguard()})
    _executor(tmp_path, second).run(_wf())
    assert second.jobs[0].run_dir.name == f"{first.name}-a2"
    third = Recorder(tmp_path)
    _executor(tmp_path, third).run(_wf())
    assert third.jobs[0].run_dir.name == f"{first.name}-a3"
    assert third.jobs[0].resume_from == ckpt


def test_an_unverifiable_image_is_not_resumed(tmp_path: Path) -> None:
    """security N1: run.json without the image id (lost, damaged) fails closed."""
    first = _failed_once(tmp_path)
    _ckpt(first, 9)
    (first / "run.json").unlink()
    worker = Recorder(tmp_path)
    _executor(tmp_path, worker).run(_wf())
    assert worker.jobs[0].resume_from is None


def test_an_unreadable_checkpoint_dir_is_not_trusted(tmp_path: Path) -> None:
    """security N2: a sub dir that cannot be listed cannot be verified."""
    from vlm_pipeline.graph.worker import checkpoint_tree_problem

    ckpt = _ckpt(tmp_path / "run", 9)
    locked = ckpt / "optim"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert checkpoint_tree_problem(ckpt) is not None
    finally:
        locked.chmod(0o700)
