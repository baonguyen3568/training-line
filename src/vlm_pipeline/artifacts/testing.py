"""Deterministic fake run directories for frontend, tester and executor development (M1-8).

``seed_demo_runs`` covers all 8 ``RunStatus`` values; ``graph.testing.FakeWorker`` should call ``make_fake_run`` so
the dashboard reads the same files a real run writes.

Writes exactly what a real run writes (P006 contract, ``common.runinfo``): ``run.json`` via ``write_manifest``,
``metrics.jsonl`` via ``finite_json_line``, ``events.jsonl``, ``system.jsonl`` and, optionally, AutoModel-style
``checkpoints/epoch_*`` with ``LATEST``. No randomness from the clock: same arguments -> same files (except mtimes).
Never point these helpers at the real ``outputs/`` unless you mean to; ``seed_demo_runs`` refuses non-empty targets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.common.runinfo import (
    METRICS_SCHEMA_VERSION,
    RunManifest,
    RunStatus,
    Stage,
    finite_json_line,
    write_manifest,
)

FAKE_IMAGE = "vlm-automodel:26.08-1"
FAKE_DIGEST = "sha256:" + "0" * 64
FAKE_COMMIT = "0" * 40
BASE_TS = 1_760_000_000.0  # fixed epoch so fixtures are reproducible


@dataclass(frozen=True)
class FakeRunSpec:
    run_id: str
    status: RunStatus = RunStatus.COMPLETED
    stage: Stage = "train"
    task: str = "lora"
    n_steps: int = 20
    checkpoint_every: int = 10  # 0 = no checkpoints
    base_model: str = "Qwen/Qwen3.5-0.8B"
    dataset: str = "naver-clova-ix/cord-v2"
    parent_run_id: str | None = None
    error_code: str | None = None  # set for failed runs, e.g. "MEM-003"
    start_ts: float = BASE_TS
    step_s: float = 15.0


def _metric(step: int, n_steps: int, ts: float) -> dict[str, object]:
    progress = step / max(n_steps, 1)
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "ts": ts,
        "step": step,
        "epoch": round(progress, 4),
        "split": "train",
        "loss": round(2.5 * math.exp(-3 * progress) + 0.3, 4),
        "lr": 1e-4,
        "grad_norm": round(1.0 + 0.5 * math.cos(step), 4),
        "tokens_per_s": 1800.0,
        "mem_max_alloc_gb": 14.2,
    }


def _jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(finite_json_line(r) + "\n" for r in records), encoding="utf-8")


def make_fake_run(runs_dir: Path, spec: FakeRunSpec) -> Path:
    """Create ``runs_dir/<run_id>`` for ``spec`` and return it. Overwrites files of the same run."""
    run_dir = runs_dir / spec.run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    finished = spec.status not in {RunStatus.QUEUED, RunStatus.RUNNING}
    # a running/interrupted/failed run stops half way; queued has no steps yet
    done_steps = {
        RunStatus.QUEUED: 0,
        RunStatus.RUNNING: spec.n_steps // 2,
        RunStatus.INTERRUPTED: spec.n_steps // 2,
        RunStatus.FAILED: spec.n_steps // 3,
        RunStatus.CANCELLED: spec.n_steps // 2,
    }.get(spec.status, spec.n_steps)
    ts_of = [spec.start_ts + i * spec.step_s for i in range(done_steps + 1)]
    end_ts = ts_of[-1] + spec.step_s if finished else None

    write_manifest(
        run_dir,
        RunManifest(
            run_id=spec.run_id,
            name=spec.run_id,
            stage=spec.stage,
            task=spec.task,
            status=spec.status,
            base_model=spec.base_model,
            base_model_revision="0" * 40,
            dataset=spec.dataset,
            dataset_revision="0" * 40,
            parent_run_id=spec.parent_run_id,
            config_hash="sha256:" + "1" * 64,
            git_commit=FAKE_COMMIT,
            git_dirty=False,
            seed=42,
            container_image=FAKE_IMAGE,
            container_digest=FAKE_DIGEST,
            exit_code=None if not finished else (0 if spec.status is RunStatus.COMPLETED else 1),
            error_code=spec.error_code,
            started_at=spec.start_ts,
            ended_at=end_ts,
            tags=["fixture"],
        ),
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(finite_json_line(_metric(s, spec.n_steps, ts_of[s])) + "\n" for s in range(1, done_steps + 1)),
        encoding="utf-8",
    )
    _jsonl(
        run_dir / "system.jsonl",
        [
            {"schema_version": 1, "ts": ts, "host_mem_total_gb": 121.7, "host_mem_avail_gb": 30.0, "job_rss_gb": 14.0}
            for ts in ts_of[:: max(1, len(ts_of) // 5)]
        ],
    )
    events: list[dict[str, object]] = [{"schema_version": 1, "ts": spec.start_ts, "event": "started"}]
    ckpt_root = run_dir / "checkpoints"
    latest: str | None = None
    # evicted: heavy bytes were freed by retention, history kept
    if spec.status is not RunStatus.EVICTED and spec.checkpoint_every > 0 and done_steps >= spec.checkpoint_every:
        for step in range(spec.checkpoint_every, done_steps + 1, spec.checkpoint_every):
            name = f"epoch_0_step_{step - 1}"
            ckpt = ckpt_root / name
            for sub in ("model", "optim", "dataloader", "rng"):
                (ckpt / sub).mkdir(parents=True, exist_ok=True)
            (ckpt / "model" / "adapter_model.safetensors").write_bytes(b"\0" * 4096)
            (ckpt / "optim" / "__0_0.distcp").write_bytes(b"\0" * 8192)
            latest = name
            events.append({"schema_version": 1, "ts": ts_of[step], "event": "checkpoint_saved", "path": name})
        link = ckpt_root / "LATEST"
        if link.is_symlink():
            link.unlink()
        if latest is not None:
            link.symlink_to(latest)
    if finished:
        final: dict[str, object] = {"schema_version": 1, "ts": end_ts, "event": str(spec.status)}
        if spec.error_code:
            final["code"] = spec.error_code
        events.append(final)
    _jsonl(run_dir / "events.jsonl", events)
    (run_dir / "logs" / "stdout.log").write_text(f"fixture run {spec.run_id}: {spec.status}\n", encoding="utf-8")
    return run_dir


DEMO_RUNS: tuple[FakeRunSpec, ...] = (
    FakeRunSpec("demo-lora-completed", RunStatus.COMPLETED, n_steps=40),
    FakeRunSpec("demo-lora-running", RunStatus.RUNNING, n_steps=40, start_ts=BASE_TS + 3600),
    FakeRunSpec("demo-lora-failed", RunStatus.FAILED, n_steps=30, error_code="MEM-003"),
    FakeRunSpec("demo-lora-cancelled", RunStatus.CANCELLED, n_steps=30),
    FakeRunSpec("demo-lora-interrupted", RunStatus.INTERRUPTED, n_steps=30),
    FakeRunSpec(
        "demo-quant-fp8",
        RunStatus.COMPLETED,
        stage="quantize",
        task="quantize",
        n_steps=1,
        checkpoint_every=0,
        parent_run_id="demo-lora-completed",
    ),
    FakeRunSpec("demo-queued", RunStatus.QUEUED, n_steps=20, checkpoint_every=0),
    FakeRunSpec("demo-lora-evicted", RunStatus.EVICTED, n_steps=20, start_ts=BASE_TS - 86_400),
    FakeRunSpec("demo-lora-discarded", RunStatus.DISCARDED, n_steps=20, start_ts=BASE_TS - 2 * 86_400),
)


def seed_demo_runs(runs_dir: Path, specs: Sequence[FakeRunSpec] = DEMO_RUNS) -> list[Path]:
    """Create one run per spec in an empty (or missing) ``runs_dir``; refuses a non-empty directory."""
    if runs_dir.exists() and any(runs_dir.iterdir()):
        raise FileExistsError(f"{runs_dir} is not empty; seed into a fresh directory")
    return [make_fake_run(runs_dir, spec) for spec in specs]
