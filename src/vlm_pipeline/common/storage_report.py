"""Build ``outputs/system/disk_report.json`` (ADR-005 D6, schema_version 1) from already-measured numbers.

Pure functions only: measuring (statvfs, /proc/meminfo, du, docker) happens in ``scripts/cleanup/disk_report.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

Level = Literal["ok", "warn", "critical"]

SCHEMA_VERSION = 1
_RANK: dict[Level, int] = {"ok": 0, "warn": 1, "critical": 2}


@dataclass(frozen=True)
class Thresholds:
    """ADR-005 D6 defaults (user-confirmed 2026-10-12)."""

    mem_available_warn_gb: float = 20.0
    mem_available_critical_gb: float = 10.0
    disk_free_warn_gb: float = 500.0
    disk_free_critical_gb: float = 200.0
    project_warn_gb: float = 500.0
    project_critical_gb: float = 800.0


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Reason:
    metric: str
    value: float
    threshold: float
    level: Level

    def as_dict(self) -> dict[str, object]:
        return {"metric": self.metric, "value": round(self.value, 1), "threshold": self.threshold, "level": self.level}


def _below(metric: str, value: float, warn: float, critical: float) -> Reason | None:
    if value < critical:
        return Reason(metric, value, critical, "critical")
    if value < warn:
        return Reason(metric, value, warn, "warn")
    return None


def _above(metric: str, value: float, warn: float, critical: float) -> Reason | None:
    if value > critical:
        return Reason(metric, value, critical, "critical")
    if value > warn:
        return Reason(metric, value, warn, "warn")
    return None


def evaluate(
    *,
    disk_free_gb: float,
    mem_available_gb: float,
    project_total_gb: float,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[Level, list[Reason]]:
    t = thresholds
    reasons = [
        r
        for r in (
            _below("mem_available_gb", mem_available_gb, t.mem_available_warn_gb, t.mem_available_critical_gb),
            _below("disk_free_gb", disk_free_gb, t.disk_free_warn_gb, t.disk_free_critical_gb),
            _above("project_total_gb", project_total_gb, t.project_warn_gb, t.project_critical_gb),
        )
        if r is not None
    ]
    level: Level = max((r.level for r in reasons), key=_RANK.__getitem__, default="ok")
    return level, reasons


def build_report(
    *,
    ts: str,
    disk: Mapping[str, float],
    mem: Mapping[str, float],
    project: Mapping[str, float],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """``disk`` needs total_gb/used_gb/free_gb, ``mem`` needs available_gb, ``project`` maps component -> GB."""
    project_total = sum(project.values())
    level, reasons = evaluate(
        disk_free_gb=disk["free_gb"],
        mem_available_gb=mem["available_gb"],
        project_total_gb=project_total,
        thresholds=thresholds,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": ts,
        "disk": {k: round(v, 1) for k, v in disk.items()},
        "mem": {k: round(v, 1) for k, v in mem.items()},
        "project": {**{k: round(v, 1) for k, v in project.items()}, "total_gb": round(project_total, 1)},
        "level": level,
        "reasons": [r.as_dict() for r in reasons],
        "thresholds": asdict(thresholds),
    }
