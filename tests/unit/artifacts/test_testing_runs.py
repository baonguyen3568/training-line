"""Fake run dirs must satisfy the same contracts as real ones (runinfo + retention + metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.retention import plan
from vlm_pipeline.artifacts.testing import DEMO_RUNS, FakeRunSpec, make_fake_run, seed_demo_runs
from vlm_pipeline.common.runinfo import RunStatus, read_manifest


def test_seed_demo_runs_are_valid_contract_runs(tmp_path: Path) -> None:
    runs = tmp_path / "outputs" / "runs"
    dirs = seed_demo_runs(runs)
    assert {d.name for d in dirs} == {s.run_id for s in DEMO_RUNS}
    statuses = set()
    for d in dirs:
        manifest = read_manifest(d)
        statuses.add(manifest.status)
        for name in ("metrics.jsonl", "events.jsonl", "system.jsonl"):
            for line in (d / name).read_text().splitlines():
                rec = json.loads(line)  # strict JSON
                assert "schema_version" in rec
    assert statuses == set(RunStatus)


def test_seed_refuses_non_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "something").mkdir()
    with pytest.raises(FileExistsError):
        seed_demo_runs(tmp_path)


def test_fake_run_is_deterministic_and_has_latest(tmp_path: Path) -> None:
    spec = FakeRunSpec("r", n_steps=20, checkpoint_every=10)
    a = make_fake_run(tmp_path / "a", spec)
    b = make_fake_run(tmp_path / "b", spec)
    for name in ("run.json", "metrics.jsonl", "events.jsonl"):
        assert (a / name).read_text() == (b / name).read_text()
    assert (a / "checkpoints" / "LATEST").resolve().name == "epoch_0_step_19"
    assert len((a / "metrics.jsonl").read_text().splitlines()) == 20


def test_retention_understands_fixture_runs(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    seed_demo_runs(tmp_path / "outputs" / "runs")
    p = plan(tmp_path, lock_reader=lambda _p: None)
    blocked = {r.run_id: {i.blocked_by for i in r.items} for r in p.runs if r.items}
    assert blocked["demo-lora-running"] == {"RUN_NOT_FINISHED"}
    assert blocked["demo-lora-interrupted"] == {"RUN_INTERRUPTED"}  # resumable for 7 days (ADR-005, PM 2026-10-13)
    assert "UNKNOWN_STATUS" not in {b for v in blocked.values() for b in v}
    assert not (tmp_path / "outputs" / "runs" / "demo-lora-evicted" / "checkpoints").exists()
