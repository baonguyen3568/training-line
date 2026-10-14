"""Run directory contract shared by runner, retention, dashboard and tests (P006 contract, P008 §2.3, ADR-005 D2).

``outputs/runs/<run_id>/run.json`` is written by the job runner only (never by plugins/workers). ``RunStatus`` is the
single status vocabulary; retention, frontend and runner import it from here.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


_DATASET_REF = re.compile(r"@([0-9a-f]{40})(?:#|$)")
# Local data (data.local_folder and its splits, DatasetRef.hash): the records' content digest IS the revision. Each
# record carries its image's sha256, so the digest pins the labels AND the images.
_LOCAL_REF = re.compile(r"^local:([0-9a-f]{64})(?:#|$)")


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
    # When the job's container started (after waiting for memory and the GPU lock): durations measured from here
    # exclude queueing (train.estimate). None for host jobs and runs recorded before this field.
    container_started_at: float | None = None
    # P017 C6: the earlier attempt's checkpoint this run resumed from, as names only (same runs root as this run):
    # {"run": "<run dir name>", "checkpoint": "epoch_<E>_step_<S>"}. Retention keeps that run dir while this one may
    # still need it. None: not a resume.
    resumed_from: dict[str, str] | None = None
    peak_mem_gb: float | None = None  # host MemAvailable drop during the job (UMA: includes CUDA)
    est_peak_gb: float | None = None  # what the job was expected to need (preflight / memory queue input)
    # This attempt failed in a way the executor retried once in a new attempt dir, e.g. "gpu_alloc_transient".
    retried_after: str | None = None
    host_min_available_gb: float | None = None
    trainable_pct: float | None = None  # "Trainable parameters percentage" from the engine log
    # Known-bad engine warnings (jobrun.ENGINE_WARNINGS). None = log never scanned (older runs, runner failure).
    engine_warnings: list[str] | None = None
    memguard_fast_fall: bool | None = None  # watchdog killed the job because memory fell too fast
    memguard_abort_gb: float | None = None
    post_steps: list[dict[str, Any]] = Field(default_factory=list)  # host steps after the job (code sha recorded)
    tags: list[str] = Field(default_factory=list)
    # H-prompt (P021): the one question every training sample was asked (the trial and the model card use it); None
    # when they differ or it is not known (a catalog dataset converted inside the engine).
    train_prompt: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _revision_from_dataset_ref(self) -> Self:
        """``dataset`` given as ``repo@<sha>[#...]`` (smoke CLI, app artifacts) also pins ``dataset_revision``.

        Local data (``local:<sha256>[#...]``) pins it as ``sha256:<content digest>`` (TD-2).
        """
        if not self.dataset:
            return self
        if m := _DATASET_REF.search(self.dataset):
            revision = m.group(1)
        elif m := _LOCAL_REF.match(self.dataset):
            revision = f"sha256:{m.group(1)}"
        else:
            return self
        if self.dataset_revision is None:
            self.dataset_revision = revision
        elif self.dataset_revision != revision:  # a self-contradicting manifest is worse than a missing field
            raise ValueError(f"dataset_revision {self.dataset_revision} != revision in dataset {self.dataset}")
        return self


def _atomic_write(path: Path, text: str) -> None:
    # Unique per process AND thread: the app writes run.json from several threads (graph runs in parallel).
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(run_dir / "run.json", manifest.model_dump_json(indent=2) + "\n")


RUN_JSON_MAX = 1 << 20  # run.json is written by the job too: never read more than this, never a fifo or a symlink


def read_small_json(
    path: Path, max_bytes: int = RUN_JSON_MAX, *, loads: Callable[[bytes], object] | None = None
) -> object:
    """JSON of a small regular file a job may have written (run.json). Never blocks, never follows a symlink.

    ``loads``: a stricter parser for untrusted content (download.resolver: no NaN, bounded depth); default json.

    Raises:
        FileNotFoundError: no file.
        OSError: a symlink (ELOOP) or unreadable.
        ValueError: not a regular file (fifo, device...), larger than ``max_bytes``, or not JSON.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            raise ValueError("not a small regular file")
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("file grew past the limit")
    return loads(data) if loads is not None else json.loads(data.decode("utf-8"))


def read_manifest(run_dir: Path) -> RunManifest:
    """The run.json of ``run_dir`` (a job may write it: bounded, no symlink, no fifo - see ``read_small_json``)."""
    return RunManifest.model_validate(read_small_json(run_dir / "run.json"))


_REPRO_REQUIRED = ("git_commit", "container_digest", "config_hash", "seed", "base_model_revision")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_REVISION = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")  # hub commit, or local content digest


def reproducibility_gaps(manifest: RunManifest) -> list[str]:
    """Field names that make a run not reproducible (empty = OK). Drives the dashboard badge (ADR-002 B2)."""
    gaps = [name for name in _REPRO_REQUIRED if getattr(manifest, name) is None]
    if manifest.git_commit is not None and not _SHA40.match(manifest.git_commit) and "git_commit" not in gaps:
        gaps.append("git_commit")
    if manifest.git_dirty:
        gaps.append("git_dirty")
    if manifest.dataset is not None and not _CONTENT_REVISION.match(manifest.dataset_revision or ""):
        gaps.append("dataset_revision")  # a dataset without a pinned revision cannot be re-read identically
    for step in manifest.post_steps:  # host post-processing must also come from committed code
        if step.get("git_dirty") or not _SHA40.match(str(step.get("git_commit") or "")):
            gaps.append(f"post_steps.{step.get('name', '?')}")
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
