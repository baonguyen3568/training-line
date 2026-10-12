"""Host post-step provenance of smoke runs (ADR-002 A1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest, write_manifest
from vlm_pipeline.train import smoke


def test_rerun_replaces_post_step_and_keeps_superseded_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "epoch_0_step_9" / "model" / "consolidated").mkdir(parents=True)
    steps = [
        {"name": "complete_from_base", "git_commit": "a" * 40, "git_dirty": True, "checkpoints": [{"restored": ["n"]}]},
        {"name": "other", "git_commit": "b" * 40, "git_dirty": False},
    ]
    manifest = RunManifest(
        run_id="r", name="r", stage="train", task="sft", status=RunStatus.COMPLETED, post_steps=steps
    )
    write_manifest(run_dir, manifest)
    monkeypatch.setattr("vlm_pipeline.common.jobrun.git_state", lambda _root, _dirs=(): ("c" * 40, False))
    monkeypatch.setattr("vlm_pipeline.train.ckpt_complete.complete_from_base", lambda _base, _ckpt: ["x"])

    smoke._complete_checkpoints(run_dir, tmp_path / "base", tmp_path)

    steps = read_manifest(run_dir).post_steps
    assert [s["name"] for s in steps] == ["other", "complete_from_base"]
    assert steps[1]["git_commit"] == "c" * 40
    assert steps[1]["git_dirty"] is False
    assert steps[1]["supersedes"] == [
        {"name": "complete_from_base", "git_commit": "a" * 40, "git_dirty": True, "checkpoints": [{"restored": ["n"]}]}
    ]
    assert steps[1]["checkpoints"] == [
        {"checkpoint": "checkpoints/epoch_0_step_9/model/consolidated", "restored": ["x"]}
    ]


def test_complete_refuses_unfinished_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    write_manifest(run_dir, RunManifest(run_id="r", name="r", stage="train", task="sft", status=RunStatus.RUNNING))
    monkeypatch.setattr(smoke, "project_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exc:
        smoke.main(["complete", str(run_dir), "--prepared", "p"])
    assert exc.value.code == 2
