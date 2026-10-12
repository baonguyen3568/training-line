"""Read side of ``outputs/runs`` for the app (GET /api/runs, GET /api/runs/{id}/metrics; P009 §6, P006 contract).

Read-only. ``list_runs`` skips unreadable run.json files (logged, never fatal). ``read_metrics_page`` pages through
``metrics.jsonl`` with an opaque byte-offset cursor and never returns a half-written last line, so tailing a running
job is safe; ``downsample_to`` (first load only) keeps the first/last point and the min/max ``loss`` per bucket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_pipeline.common.runinfo import MetricsPage, RunManifest, read_manifest

logger = logging.getLogger(__name__)

RUNS_REL = Path("outputs") / "runs"
MAX_LIMIT = 10_000
MIN_DOWNSAMPLE = 2
DOWNSAMPLE_KEY = "loss"


def list_runs(root: Path) -> list[RunManifest]:
    """Manifests of every run under ``<root>/outputs/runs``, newest ``started_at`` first."""
    runs_dir = root / RUNS_REL
    manifests: list[RunManifest] = []
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if run_dir.is_symlink() or not run_dir.is_dir() or not (run_dir / "run.json").is_file():
            continue
        try:
            manifests.append(read_manifest(run_dir))
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("skipping %s: unreadable run.json (%s)", run_dir.name, type(exc).__name__)
    return sorted(manifests, key=lambda m: m.started_at, reverse=True)


def _parse_cursor(after: str | None) -> int:
    if after is None or after == "":
        return 0
    if not after.isdigit():
        raise ValueError(f"invalid cursor {after!r}")
    return int(after)


def _complete_lines(data: bytes) -> tuple[list[bytes], int]:
    """Complete (newline-terminated) lines and the number of bytes they span."""
    end = data.rfind(b"\n") + 1
    return data[:end].split(b"\n")[:-1], end


def _downsample(points: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Keep first, last and per-bucket min/max of ``loss`` (or evenly spaced points without it); order kept."""
    if len(points) <= n:
        return points
    keep = {0, len(points) - 1}
    inner = list(range(1, len(points) - 1))
    buckets = max(1, (n - 2) // 2)
    size = len(inner) / buckets
    for b in range(buckets):
        chunk = inner[int(b * size) : int((b + 1) * size)]
        if not chunk:
            continue
        values = [(points[i].get(DOWNSAMPLE_KEY), i) for i in chunk]
        numeric = [(v, i) for v, i in values if isinstance(v, int | float)]
        if numeric:
            keep.add(min(numeric)[1])
            keep.add(max(numeric)[1])
        else:
            keep.add(chunk[0])
    return [points[i] for i in sorted(keep)]


def read_metrics_page(run_dir: Path, after: str | None, limit: int, downsample_to: int | None = None) -> MetricsPage:
    """One page of ``metrics.jsonl`` starting at cursor ``after`` (``None`` = start of file).

    Raises:
        ValueError: bad cursor/limit/downsample_to, or ``downsample_to`` combined with a cursor.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be in 1..{MAX_LIMIT}")
    if downsample_to is not None and (downsample_to < MIN_DOWNSAMPLE or after not in (None, "")):
        raise ValueError("downsample_to (>= 2) is only allowed on the first page")
    offset = _parse_cursor(after)
    path = run_dir / "metrics.jsonl"
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read()  # TODO: read in chunks once runs log enough steps for this to matter (v1: few MB)
    except FileNotFoundError:
        return MetricsPage(points=[], next_cursor=str(offset), has_more=False, downsampled=False)

    lines, span = _complete_lines(data)
    if downsample_to is not None:
        everything = [json.loads(line) for line in lines if line.strip()]
        sampled = _downsample(everything, downsample_to)
        return MetricsPage(
            points=sampled, next_cursor=str(offset + span), has_more=False, downsampled=len(sampled) < len(everything)
        )

    points: list[dict[str, Any]] = []
    consumed = 0
    for line in lines:
        if len(points) >= limit:
            break
        consumed += len(line) + 1
        if line.strip():
            points.append(json.loads(line))
    return MetricsPage(points=points, next_cursor=str(offset + consumed), has_more=consumed < span, downsampled=False)
