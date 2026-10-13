"""How long a training node will take, from MEASURED runs only (PM 2026-10-13: never a guess).

Model (bao-tester review 2026-10-13: mixing validation pauses into seconds-per-step under-estimated by 16%)::

    T = startup + (steps - 1) * t_step + n_mid * (t_val + t_ckpt) + t_final

* ``t_step``: median gap between consecutive logged train steps, leaving out gaps that hold a validation or a
  checkpoint and outliers (> 5x the median: an interrupted / resumed run);
* ``t_val``: validation time (validation.jsonl row - the train step it follows; the engine also validates after the
  last step, so even a short run measures it);
* ``t_ckpt``: what remains of a between-steps gap after the step and the validation (measured only by runs that
  saved between steps);
* a run lacking ``t_val`` or ``t_ckpt`` borrows the median of the runs that measured it; with none at all, an
  estimate that needs it is None (never "free");
* ``n_mid``: validations/checkpoints BETWEEN steps: ``floor((steps - 1) / every)`` (the engine validates and saves
  every ``ckpt_every_steps``; the recipe sets val_every = ckpt_every);
* ``t_final``: last logged step -> job end (final validation + checkpoint + consolidation), nearly constant;
* ``startup``: container start (``container_started_at``; older runs: ``started_at``, which may include queueing)
  -> first logged step (so it holds the first step too: ``steps - 1`` more follow).

A measurement is used only for the SAME base model, container image, task (sft / lora), batch shape (global + local)
and training dataset (``repo@revision``, quick-run limit ignored: per-step cost does not depend on it). Up to
``MAX_RUNS`` newest runs give the range (min .. max of their predictions). Unknown -> None ("chưa có số đo").
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MIN_STEPS = 10
MAX_RUNS = 3
OUTLIER = 5.0
RUN_ROOTS = (Path("outputs") / "runs", Path("outputs") / "graph" / "runs")


@dataclass(frozen=True)
class RunTiming:
    run_id: str
    t_step: float
    t_val: float | None  # None: this run logged no validation (val off / no val set): never read as "free"
    t_ckpt: float | None  # None: this run never saved between steps
    t_final: float
    startup: float
    ended_at: float

    def predict_s(
        self, steps: int, every: int, t_ckpt: float | None = None, t_val: float | None = None
    ) -> float | None:
        """Seconds for ``steps`` at cadence ``every``; ``t_ckpt``/``t_val`` fill this run's own when it has none."""
        n_mid = (steps - 1) // every if every > 0 else 0
        ckpt = self.t_ckpt if self.t_ckpt is not None else t_ckpt
        val = self.t_val if self.t_val is not None else t_val
        if n_mid and (ckpt is None or val is None):
            return None
        return self.startup + (steps - 1) * self.t_step + n_mid * ((val or 0.0) + (ckpt or 0.0)) + self.t_final


@dataclass(frozen=True)
class TrainEstimate:
    steps: int
    minutes_lo: float
    minutes_hi: float
    based_on: tuple[str, ...]  # run ids of the measurements


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _rows(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[int(row["step"])] = _ts(row["timestamp"])
    return out


def _schedule(run_dir: Path) -> tuple[int, int, int] | None:
    """(global batch, local batch, ckpt_every_steps) from the run's config.yaml."""
    import yaml

    try:
        sched = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["step_scheduler"]
        return int(sched["global_batch_size"]), int(sched["local_batch_size"]), int(sched.get("ckpt_every_steps", 0))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def run_timing(run_dir: Path, run: dict[str, Any]) -> RunTiming | None:
    """The components of one finished run, or None when its logs cannot give them."""
    sched = _schedule(run_dir)
    try:
        train = _rows(run_dir / "checkpoints" / "training.jsonl")
        val_path = run_dir / "checkpoints" / "validation.jsonl"
        val = _rows(val_path) if val_path.is_file() else {}
        ended = float(run["ended_at"])
        start = float(run.get("container_started_at") or run["started_at"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    steps = sorted(train)
    if sched is None or len(steps) < MIN_STEPS:
        return None
    every = sched[2]
    events = {s for s in steps[:-1] if (every and (s + 1) % every == 0) or s in val}
    pure = [train[b] - train[a] for a, b in itertools.pairwise(steps) if b == a + 1 and a not in events]
    if not pure:
        return None
    first = statistics.median(pure)
    t_step = statistics.median([g for g in pure if g <= OUTLIER * first])  # a pause / resume is not a step
    t_vals = [val[s] - train[s] for s in val if s in train and val[s] > train[s]]
    t_val = statistics.median(t_vals) if t_vals else None
    mids = [s for s in events if s + 1 in train]
    extras = [train[s + 1] - train[s] - t_step - (val[s] - train[s] if s in val else 0.0) for s in mids]
    return RunTiming(
        run_id=run_dir.name,
        t_step=t_step,
        t_val=t_val,
        t_ckpt=max(0.0, statistics.median(extras)) if extras else None,
        t_final=max(0.0, ended - train[steps[-1]]),
        startup=max(0.0, train[steps[0]] - start),
        ended_at=ended,
    )


def _dataset_key(value: object) -> str | None:
    return value.split("#", 1)[0] if isinstance(value, str) and value else None


def measured_timings(
    root: Path, *, base_model: str, image: str, task: str, batch: tuple[int, int], dataset: str
) -> list[RunTiming]:
    """Newest first, at most MAX_RUNS: completed train runs matching model, image, task, batch and dataset."""
    found: list[RunTiming] = []
    for run_root in RUN_ROOTS:
        for manifest in (root / run_root).glob("*/run.json"):
            try:
                run = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(run, dict) or run.get("status") != "completed" or run.get("stage") != "train":
                continue
            key = (run.get("base_model"), run.get("container_image"), run.get("task"), _dataset_key(run.get("dataset")))
            if key != (base_model, image, task, dataset):
                continue
            sched = _schedule(manifest.parent)
            if sched is None or sched[:2] != batch:
                continue
            timing = run_timing(manifest.parent, run)
            if timing is not None:
                found.append(timing)
    return sorted(found, key=lambda t: t.ended_at, reverse=True)[:MAX_RUNS]


def train_steps(samples: int, global_batch_size: int, epochs: int) -> int:
    return math.ceil(samples / global_batch_size) * epochs


def estimate_train(steps: int, every: int, timings: Iterable[RunTiming]) -> TrainEstimate | None:
    """``every``: the ckpt/val cadence the new run will use (its recipe), not the measured runs' own."""
    timings = list(timings)
    ckpts = [t.t_ckpt for t in timings if t.t_ckpt is not None]
    vals = [t.t_val for t in timings if t.t_val is not None]
    pooled_ckpt = statistics.median(ckpts) if ckpts else None
    pooled_val = statistics.median(vals) if vals else None
    totals = [s for t in timings if (s := t.predict_s(steps, every, pooled_ckpt, pooled_val)) is not None]
    if not totals or steps <= 0:
        return None
    lo, hi = round(min(totals) / 60, 1), round(max(totals) / 60, 1)
    return TrainEstimate(steps, lo, hi, tuple(t.run_id for t in timings))


__all__ = [
    "MIN_STEPS",
    "RunTiming",
    "TrainEstimate",
    "estimate_train",
    "measured_timings",
    "run_timing",
    "train_steps",
]
