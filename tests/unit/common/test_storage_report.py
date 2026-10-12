"""Unit tests for ``vlm_pipeline.common.storage_report`` (ADR-005 D6 thresholds, disk_report.json schema v1)."""

from __future__ import annotations

import json

import pytest

from vlm_pipeline.common.storage_report import SCHEMA_VERSION, Thresholds, build_report, evaluate


def test_all_ok() -> None:
    level, reasons = evaluate(disk_free_gb=2000, mem_available_gb=43, project_total_gb=100)
    assert level == "ok"
    assert reasons == []


@pytest.mark.parametrize(
    ("kwargs", "metric", "level"),
    [
        ({"mem_available_gb": 15}, "mem_available_gb", "warn"),
        ({"mem_available_gb": 9}, "mem_available_gb", "critical"),
        ({"disk_free_gb": 400}, "disk_free_gb", "warn"),
        ({"disk_free_gb": 150}, "disk_free_gb", "critical"),
        ({"project_total_gb": 600}, "project_total_gb", "warn"),
        ({"project_total_gb": 900}, "project_total_gb", "critical"),
    ],
)
def test_single_metric_levels(kwargs: dict[str, float], metric: str, level: str) -> None:
    base = {"disk_free_gb": 2000.0, "mem_available_gb": 43.0, "project_total_gb": 100.0} | kwargs
    got_level, reasons = evaluate(**base)
    assert got_level == level
    assert [(r.metric, r.level) for r in reasons] == [(metric, level)]


def test_worst_level_wins_and_all_reasons_listed() -> None:
    level, reasons = evaluate(disk_free_gb=400, mem_available_gb=9, project_total_gb=600)
    assert level == "critical"
    assert {r.metric for r in reasons} == {"mem_available_gb", "disk_free_gb", "project_total_gb"}


def test_boundaries_are_not_alerts() -> None:
    t = Thresholds()
    level, _ = evaluate(
        disk_free_gb=t.disk_free_warn_gb, mem_available_gb=t.mem_available_warn_gb, project_total_gb=t.project_warn_gb
    )
    assert level == "ok"


def test_build_report_schema_is_json_serializable() -> None:
    report = build_report(
        ts="2026-10-12T17:00:00+07:00",
        disk={"total_gb": 3700.0, "used_gb": 1600.0, "free_gb": 2000.0},
        mem={"total_gb": 121.0, "available_gb": 15.0, "swap_used_gb": 8.0},
        project={"cache_hf_gb": 36.04, "outputs_gb": 1.0, "docker_vlm_gb": 29.4},
    )
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["level"] == "warn"
    assert report["project"] == {"cache_hf_gb": 36.0, "outputs_gb": 1.0, "docker_vlm_gb": 29.4, "total_gb": 66.4}
    assert report["reasons"] == [{"metric": "mem_available_gb", "value": 15.0, "threshold": 20.0, "level": "warn"}]
    json.dumps(report)
