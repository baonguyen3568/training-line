"""Run directory contract shared by runner, retention, dashboard and tests (P006 contract, P008 §2.3, ADR-005 D2).

``outputs/runs/<run_id>/run.json`` is written by the job runner only (never by plugins/workers). ``RunStatus`` is the
single status vocabulary; retention, frontend and runner import it from here.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUN_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    QUEUED = "queued"  # waiting for the GPU lock / its turn
    RUNNING = "running"
    COMPLETED = "completed"  # finished successfully -> retention may clean
    FAILED = "failed"  # error -> retention cleans after 3 days (ADR-005)
    CANCELLED = "cancelled"  # stopped by the user; may be resumable -> retention cleans after 3 days (ADR-005)
    INTERRUPTED = "interrupted"  # memguard (exit 75) / crash after a checkpoint -> resumable -> retention never touches
    EVICTED = "evicted"  # heavy bytes freed by retention, history kept
    DISCARDED = "discarded"  # user discarded the run -> retention may clean

    @property
    def is_terminal(self) -> bool:
        return self not in {RunStatus.QUEUED, RunStatus.RUNNING}


Stage = Literal["data", "train", "quantize", "eval", "bench", "serve", "export"]


class RunManifest(BaseModel):
    """``run.json``. Unknown keys are kept on read (forward compatible) but never written by this version."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = RUN_SCHEMA_VERSION
    run_id: str
    name: str
    stage: Stage
    task: str  # sft | lora | grpo | dpo | quantize | eval | ...
    status: RunStatus = RunStatus.RUNNING
    base_model: str | None = None
    base_model_revision: str | None = None
    dataset: str | None = None
    dataset_revision: str | None = None
    parent_run_id: str | None = None
    config_path: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    seed: int | None = None
    max_steps: int | None = None  # for "step / max_steps" progress in the dashboard
    container_image: str | None = None
    container_digest: str | None = None
    pid: int | None = None  # host PID of the runner holding the GPU lock
    exit_code: int | None = None
    oom_killed: bool | None = None
    error_code: str | None = None  # catalog code when status is failed
    started_at: float = Field(default_factory=time.time)  # unix seconds UTC
    ended_at: float | None = None
    peak_mem_gb: float | None = None  # host MemAvailable drop during the job (UMA: includes CUDA)
    host_min_available_gb: float | None = None
    trainable_pct: float | None = None  # "Trainable parameters percentage" from the engine log
    memguard_fast_fall: bool | None = None
    memguard_abort_gb: float | None = None  # watchdog killed the job because memory fell too fast
    tags: list[str] = Field(default_factory=list)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(run_dir / "run.json", manifest.model_dump_json(indent=2) + "\n")


def read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))


_REPRO_REQUIRED = ("git_commit", "container_digest", "config_hash", "seed", "base_model_revision")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def reproducibility_gaps(manifest: RunManifest) -> list[str]:
    """Field names that make a run not reproducible (empty = OK). Drives the dashboard badge (ADR-002 B2)."""
    gaps = [name for name in _REPRO_REQUIRED if getattr(manifest, name) is None]
    if manifest.git_commit is not None and not _SHA40.match(manifest.git_commit) and "git_commit" not in gaps:
        gaps.append("git_commit")
    if manifest.git_dirty:
        gaps.append("git_dirty")
    return gaps


class MetricsPage(BaseModel):
    """Response of ``GET /api/runs/{id}/metrics?after=&limit=[&downsample_to=]`` (P009 §6, frontend contract)."""

    model_config = ConfigDict(extra="forbid")

    points: list[dict[str, float | int | str | list[str] | None]]  # metrics.jsonl records (flat)
    next_cursor: str  # opaque to the client (byte offset)
    has_more: bool
    downsampled: bool


# ------------------------------------------------------------------------------------------------ metrics.jsonl
def finite_json_line(record: Mapping[str, Any]) -> str:
    """One strict-JSON line: non-finite floats become ``null`` and are listed under ``nonfinite`` (P006 rule 5)."""
    clean: dict[str, Any] = {}
    nonfinite: list[str] = []
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            clean[key] = None
            nonfinite.append(key)
        else:
            clean[key] = value
    if nonfinite:
        clean["nonfinite"] = nonfinite
    return json.dumps(clean, allow_nan=False, separators=(",", ":"))


# AutoModel training.jsonl key -> contract key (P006). Keys not listed are dropped unless numeric.
_AUTOMODEL_KEYS = {
    "loss": "loss",
    "lr": "lr",
    "grad_norm": "grad_norm",
    "mem": "mem_max_alloc_gb",
    "tps": "tokens_per_s",
    "num_tokens_per_step": "tokens_per_step",
    "num_label_tokens": "label_tokens",
}


def _iso_to_unix(ts: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def automodel_to_metrics(lines: Iterable[str], *, split: Literal["train", "val"] = "train") -> Iterator[str]:
    """Convert NeMo AutoModel ``training.jsonl``/``validation.jsonl`` lines into contract ``metrics.jsonl`` lines."""
    for raw in lines:
        if not raw.strip():
            continue
        src = json.loads(raw)  # Python's json accepts NaN/Infinity literals written upstream
        out: dict[str, Any] = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "ts": _iso_to_unix(src["timestamp"]) if "timestamp" in src else None,
            "step": int(src["step"]),
            "epoch": float(src.get("epoch", 0)),
            "split": split,
        }
        for key, value in src.items():
            if key in {"step", "epoch", "timestamp", "tps_per_gpu"}:
                continue
            target = _AUTOMODEL_KEYS.get(key, key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[target] = value
        yield finite_json_line(out)
