from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vlm_pipeline.common.runinfo import (
    RunManifest,
    RunStatus,
    automodel_to_metrics,
    finite_json_line,
    read_manifest,
    write_manifest,
)

FIXTURE = (
    '{"step": 3, "epoch": 0, "timestamp": "2026-10-12T09:33:45.462Z", "loss": NaN, "grad_norm": 11.5, "lr": 0.0001,'
    ' "mem": 8.13, "tps": 13.4, "tps_per_gpu": 13.4, "num_tokens_per_step": 2293, "num_label_tokens": 1164}'
)


def test_status_vocabulary_is_shared_and_stable() -> None:
    assert {s.value for s in RunStatus} == {
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "evicted",
        "discarded",
    }
    assert not RunStatus.RUNNING.is_terminal
    assert not RunStatus.QUEUED.is_terminal
    assert RunStatus.COMPLETED.is_terminal


def test_manifest_roundtrip_atomic(tmp_path: Path) -> None:
    m = RunManifest(run_id="r1", name="smoke", stage="train", task="lora", base_model="org/m", seed=1234)
    write_manifest(tmp_path, m)
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    back = read_manifest(tmp_path)
    assert back.status is RunStatus.RUNNING
    assert back == m
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["status"] == "running"
    assert data["schema_version"] == 1


def test_manifest_keeps_unknown_keys_on_read(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"run_id": "r", "name": "n", "stage": "eval", "task": "eval", "future_key": 1, "status": "evicted"})
    )
    m = read_manifest(tmp_path)
    assert m.status is RunStatus.EVICTED
    assert m.model_extra == {"future_key": 1}


def test_finite_json_line_replaces_nonfinite() -> None:
    line = finite_json_line({"loss": float("nan"), "lr": float("inf"), "step": 1})
    assert json.loads(line) == {"loss": None, "lr": None, "step": 1, "nonfinite": ["loss", "lr"]}


def test_automodel_adapter_maps_keys_and_is_strict_json() -> None:
    (line,) = list(automodel_to_metrics([FIXTURE, ""]))
    assert "NaN" not in line
    rec = json.loads(line)
    assert rec["schema_version"] == 1
    assert rec["step"] == 3
    assert rec["split"] == "train"
    assert rec["mem_max_alloc_gb"] == 8.13
    assert rec["tokens_per_s"] == 13.4
    assert rec["label_tokens"] == 1164
    assert rec["loss"] is None
    assert rec["nonfinite"] == ["loss"]
    expected = datetime(2026, 10, 12, 9, 33, 45, 462000, tzinfo=UTC).timestamp()
    assert abs(rec["ts"] - expected) < 1e-3
    assert "tps_per_gpu" not in rec


def test_concurrent_manifest_writes_from_threads(tmp_path: Path) -> None:
    import threading

    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            for _ in range(50):
                write_manifest(tmp_path, RunManifest(run_id="r", name=f"n{i}", stage="train", task="lora"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert read_manifest(tmp_path).run_id == "r"
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_post_steps_from_dirty_code_are_reproducibility_gaps() -> None:
    from vlm_pipeline.common.runinfo import reproducibility_gaps

    base = {"run_id": "r", "name": "n", "stage": "train", "task": "sft", "git_commit": "a" * 40, "git_dirty": False,
            "container_digest": "sha256:x", "config_hash": "c", "seed": 1, "base_model_revision": "b" * 40}  # fmt: skip
    clean = RunManifest(**base, post_steps=[{"name": "complete_from_base", "git_commit": "c" * 40, "git_dirty": False}])
    assert reproducibility_gaps(clean) == []
    dirty = RunManifest(**base, post_steps=[{"name": "complete_from_base", "git_commit": "c" * 40, "git_dirty": True}])
    assert reproducibility_gaps(dirty) == ["post_steps.complete_from_base"]
