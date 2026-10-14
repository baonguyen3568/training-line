from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vlm_pipeline.common.runinfo import (
    RunManifest,
    RunStatus,
    automodel_to_metrics,
    finite_json_line,
    read_manifest,
    reproducibility_gaps,
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
    base = {"run_id": "r", "name": "n", "stage": "train", "task": "sft", "git_commit": "a" * 40, "git_dirty": False,
            "container_digest": "sha256:x", "config_hash": "c", "seed": 1, "base_model_revision": "b" * 40}  # fmt: skip
    clean = RunManifest(**base, post_steps=[{"name": "complete_from_base", "git_commit": "c" * 40, "git_dirty": False}])
    assert reproducibility_gaps(clean) == []
    dirty = RunManifest(**base, post_steps=[{"name": "complete_from_base", "git_commit": "c" * 40, "git_dirty": True}])
    assert reproducibility_gaps(dirty) == ["post_steps.complete_from_base"]


@pytest.mark.parametrize(
    ("dataset", "revision", "gaps"),
    [
        ("naver-clova-ix/cord-v2@" + "d" * 40, "d" * 40, []),
        ("naver-clova-ix/cord-v2@" + "d" * 40 + "#limit=20", "d" * 40, []),
        ("naver-clova-ix/cord-v2", None, ["dataset_revision"]),
        ("naver-clova-ix/cord-v2@main", None, ["dataset_revision"]),
        # TD-2: local data (data.local_folder, its splits) is pinned by its records' content digest
        ("local:" + "f" * 64 + "#split=train#val=0.1#seed=42", "sha256:" + "f" * 64, []),
        ("local:" + "f" * 64 + "#split=all#limit=50", "sha256:" + "f" * 64, []),
        ("local:" + "f" * 63, None, ["dataset_revision"]),  # not a digest
        ("local:/home/me/folder", None, ["dataset_revision"]),  # a path is never a revision
        (None, None, []),
    ],
)
def test_dataset_revision_derived_from_pinned_ref_else_gap(
    dataset: str | None, revision: str | None, gaps: list[str]
) -> None:
    m = RunManifest(run_id="r", name="n", stage="train", task="sft", git_commit="a" * 40, git_dirty=False,
                    container_digest="sha256:x", config_hash="c", seed=1, base_model_revision="b" * 40,
                    dataset=dataset)  # fmt: skip
    assert m.dataset_revision == revision
    assert reproducibility_gaps(m) == gaps


def test_the_dataset_ref_of_local_data_pins_the_run() -> None:
    """The real producer: DatasetRef.hash of a split local folder -> a reproducible run."""
    from vlm_pipeline.common.dataset_ref import DatasetRef

    ref = DatasetRef(path="/workspace/x", split="train", records="/workspace/r.jsonl", records_sha256="a" * 64,
                     val_fraction=0.1, seed=42)  # fmt: skip
    m = RunManifest(run_id="r", name="n", stage="train", task="sft", dataset=ref.hash)
    assert m.dataset_revision == "sha256:" + "a" * 64
    assert "dataset_revision" not in reproducibility_gaps(m)  # the point of TD-2: the run is reproducible


def test_dataset_revision_contradicting_dataset_ref_is_rejected() -> None:
    with pytest.raises(ValueError, match="dataset_revision"):
        RunManifest(
            run_id="r", name="n", stage="train", task="sft", dataset="x/y@" + "d" * 40, dataset_revision="e" * 40
        )


@pytest.mark.parametrize("kind", ["fifo", "symlink", "big"])
def test_a_job_written_run_json_never_hangs_or_floods(tmp_path: Path, kind: str) -> None:
    """security: run.json is job-writable; read_manifest refuses a fifo, a symlink or > 1 MB at once."""
    import os

    from vlm_pipeline.common.runinfo import RUN_JSON_MAX, read_manifest

    path = tmp_path / "run.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        target = tmp_path / "elsewhere.json"
        target.write_text("{}")
        path.symlink_to(target)
    else:
        path.write_text('{"run_id": "' + "x" * (RUN_JSON_MAX + 10) + '"}')
    with pytest.raises((OSError, ValueError)):
        read_manifest(tmp_path)
