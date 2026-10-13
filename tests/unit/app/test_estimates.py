"""Duration estimates come from measured runs only (PM 2026-10-13); anything unknown is null.

Component model after bao-tester's review: a validation pause is not part of seconds-per-step.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app.estimates import workflow_duration
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.estimate import estimate_train, measured_timings, run_timing, train_steps

PARSE = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
IMAGE = "vlm-automodel:26.08-6"
CORD = "naver-clova-ix/cord-v2@7f0115a4b758a71d6473b8d085751692da2fef98"
TEMPLATE = json.loads((project_root() / "configs" / "templates" / "invoice_to_json.json").read_text())["workflow"]
T0 = 1_791_860_000.0
STEP, VAL, CKPT, FINAL, STARTUP = 20.0, 234.0, 26.0, 257.0, 60.0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def _templates(tmp_path: Path) -> None:
    (tmp_path / "configs" / "train").mkdir(parents=True)
    (tmp_path / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")


def _run(root: Path, name: str, *, steps: int, every: int, step_s: float = STEP, **run: Any) -> Path:
    """A finished run whose logs follow the component model exactly (what the engine writes)."""
    run_dir = root / "outputs" / "runs" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"step_scheduler:\n  global_batch_size: 8\n  local_batch_size: 1\n  ckpt_every_steps: {every}\n"
    )
    train, val, ts = [], [], T0 + STARTUP  # startup holds the first step: its row is logged at T0 + STARTUP
    for s in range(steps):
        train.append({"step": s, "timestamp": _iso(ts)})
        if s < steps - 1 and (s + 1) % every == 0:  # validate + save between steps
            val.append({"step": s, "timestamp": _iso(ts + VAL)})
            ts += VAL + CKPT
        if s < steps - 1:
            ts += step_s
    val.append({"step": steps - 1, "timestamp": _iso(ts + VAL)})  # the engine validates after the last step too
    ended = ts + FINAL
    (run_dir / "checkpoints" / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (run_dir / "checkpoints" / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    manifest = {
        "run_id": name, "status": "completed", "stage": "train", "task": "sft", "base_model": PARSE,
        "container_image": IMAGE, "dataset": f"{CORD}#limit=100", "started_at": T0, "ended_at": ended, **run,
    }  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(manifest))
    return run_dir


def _timings(root: Path) -> list[Any]:
    return measured_timings(root, base_model=PARSE, image=IMAGE, task="sft", batch=(8, 1), dataset=CORD)


def test_steps_follow_samples_batch_and_epochs() -> None:
    assert train_steps(800, 8, 3) == 300
    assert train_steps(100, 8, 1) == 13  # a partial last batch is a step


def test_replaying_a_run_gives_its_own_duration(tmp_path: Path) -> None:
    """The range must contain what really happened for the run it is based on (tester: P-3b was 116 min)."""
    run_dir = _run(tmp_path, "p3b", steps=300, every=100)
    run = json.loads((run_dir / "run.json").read_text())
    timing = run_timing(run_dir, run)
    assert timing is not None
    assert (timing.t_step, timing.t_val, round(timing.t_ckpt), timing.t_final) == (STEP, VAL, CKPT, FINAL)
    assert timing.predict_s(300, 100) == pytest.approx(run["ended_at"] - run["started_at"])
    est = estimate_train(300, 100, [timing])
    assert est is not None
    assert est.minutes_lo * 60 <= run["ended_at"] - run["started_at"] + 3 <= est.minutes_hi * 60 + 6  # rounding


def test_a_short_run_and_a_long_run_predict_the_same(tmp_path: Path) -> None:
    """No validation between steps in a 13-step smoke, three in a 300-step run: same data, same answer."""
    _run(tmp_path, "smoke", steps=13, every=25)
    _run(tmp_path, "long", steps=300, every=100)
    est = estimate_train(300, 25, _timings(tmp_path))
    assert est is not None
    assert est.minutes_lo == est.minutes_hi  # the smoke measures t_val on its final validation, borrows t_ckpt
    assert est.based_on == ("long", "smoke")


def test_a_needed_checkpoint_time_nobody_measured_is_null(tmp_path: Path) -> None:
    _run(tmp_path, "smoke", steps=13, every=25)  # never saved between steps
    assert estimate_train(300, 25, _timings(tmp_path)) is None  # 11 saves needed, none measured
    assert estimate_train(20, 25, _timings(tmp_path)) is not None  # no save between steps needed


def test_only_matching_completed_runs_count(tmp_path: Path) -> None:
    _run(tmp_path, "ok", steps=20, every=25)
    _run(tmp_path, "other-image", steps=20, every=25, container_image="vlm-automodel:26.08-5")
    _run(tmp_path, "lora", steps=20, every=25, task="lora")
    _run(tmp_path, "failed", steps=20, every=25, status="failed")
    _run(tmp_path, "too-short", steps=4, every=25)
    _run(tmp_path, "qwen", steps=20, every=25, base_model="Qwen/Qwen3.5-0.8B")
    _run(tmp_path, "other-data", steps=20, every=25, dataset="someone/receipts@" + "0" * 40)
    big = _run(tmp_path, "gbs16", steps=20, every=25)
    (big / "config.yaml").write_text("step_scheduler:\n  global_batch_size: 16\n  local_batch_size: 1\n")
    assert [t.run_id for t in _timings(tmp_path)] == ["ok"]


def test_a_pause_inside_a_run_is_not_a_step(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "resumed", steps=30, every=100)
    rows = [json.loads(line) for line in (run_dir / "checkpoints" / "training.jsonl").read_text().splitlines()]
    for row in rows[15:]:  # stopped for an hour after step 14, then resumed
        row["timestamp"] = _iso(datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp() + 3600)
    (run_dir / "checkpoints" / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    timing = run_timing(run_dir, json.loads((run_dir / "run.json").read_text()))
    assert timing is not None
    assert timing.t_step == STEP


def test_no_measurement_is_null(tmp_path: Path) -> None:
    assert estimate_train(300, 25, _timings(tmp_path)) is None
    out = workflow_duration(tmp_path, TEMPLATE, IMAGE)
    assert out["duration_min"] is None
    assert out["per_node"]["n4"]["steps"] == 300
    assert out["per_node"]["n4"]["minutes"] is None
    assert out["per_node"]["n4"]["by_epochs"] == {"1": None, "2": None, "3": None, "5": None}
    assert out["unmeasured"] == ["n5"]


def test_the_default_template_validates_once_per_epoch(tmp_path: Path) -> None:
    """CORD num_train 800 (its own validation split), 3 epochs, GBS 8 -> 300 steps; the node recipe saves and
    validates at epoch ends only (build_recipe) -> 2 pauses between steps, the third is the final one."""
    _run(tmp_path, "p3b", steps=300, every=100)
    out = workflow_duration(tmp_path, TEMPLATE, IMAGE)
    n4 = out["per_node"]["n4"]
    assert n4["steps"] == 300
    expected = round((STARTUP + 299 * STEP + 2 * (VAL + CKPT) + FINAL) / 60, 1)
    assert out["duration_min"] == {"lo": expected, "hi": expected}
    assert n4["based_on"] == ["p3b"]
    assert set(n4["by_epochs"]) == {"1", "2", "3", "5"}  # every option of the epochs param
    assert n4["by_epochs"]["3"] == n4["minutes"]
    one = round((STARTUP + 99 * STEP + FINAL) / 60, 1)  # one epoch: no pause between steps
    assert n4["by_epochs"]["1"] == {"lo": one, "hi": one}


@pytest.mark.parametrize(
    ("change", "steps"),
    [
        (lambda w: w["nodes"][1]["params"].update(max_samples=100), 39),  # quick run: 100 samples x 3 epochs
        (lambda w: w["nodes"][3]["params"].update(epochs=1), 100),
        (lambda w: w["nodes"][3]["params"].update(epochs="lots"), None),  # junk param: no estimate, validate says why
    ],
)
def test_node_params_change_the_step_count(tmp_path: Path, change: Any, steps: int | None) -> None:
    workflow = json.loads(json.dumps(TEMPLATE))
    change(workflow)
    assert workflow_duration(tmp_path, workflow, IMAGE)["per_node"]["n4"]["steps"] == steps


def test_a_local_folder_has_no_size_before_it_is_read(tmp_path: Path) -> None:
    workflow = json.loads(json.dumps(TEMPLATE))
    workflow["nodes"][1] = {"id": "n2", "type": "data.local_folder", "version": "1.0.0", "params": {"path": "data/x"}}
    _run(tmp_path, "p3b", steps=300, every=100)
    assert workflow_duration(tmp_path, workflow, IMAGE)["duration_min"] is None


def test_a_run_without_validation_never_makes_validation_free(tmp_path: Path) -> None:
    """tester nit: no val rows -> t_val unknown (borrowed or null), never 0."""
    run_dir = _run(tmp_path, "noval", steps=13, every=25)
    (run_dir / "checkpoints" / "validation.jsonl").unlink()
    timing = run_timing(run_dir, json.loads((run_dir / "run.json").read_text()))
    assert timing is not None
    assert timing.t_val is None
    assert estimate_train(300, 25, _timings(tmp_path)) is None  # 11 validations needed, none measured
    _run(tmp_path, "long", steps=300, every=100)
    est = estimate_train(300, 25, _timings(tmp_path))
    assert est is not None  # now borrowed from the run that measured it
