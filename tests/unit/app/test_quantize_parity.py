"""B-4 parity verdicts: QNT-001 (fail, not handed over) / QNT-002 (warn, handed over) with the report's own limits."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_pipeline.app import quantize_worker as qw

THRESHOLDS = {"warn_top1": 0.9, "warn_kl": 0.05, "fail_top1": 0.8, "fail_kl": 0.2}


def _report(tmp_path: Path, **parity: object) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"parity": {"samples": 32, "thresholds": THRESHOLDS, **parity}}))
    return path


def test_fail_params_use_the_reports_limits_and_name_the_tripped_one(tmp_path: Path) -> None:
    parity = qw._parity(_report(tmp_path, top1_agreement_mean=0.71, kl_mean=0.31, verdict="fail"))
    assert parity is not None
    expected = {"node": "q", "top1_pct": 71.0, "limit_top1_pct": 80.0, "kl": 0.31, "limit_kl": 0.2}
    assert qw._parity_params("q", parity, "fail") == {**expected, "sample_count": 32, "reason": "top1"}  # both: top1


def test_kl_alone_is_named_kl(tmp_path: Path) -> None:
    parity = qw._parity(_report(tmp_path, top1_agreement_mean=0.95, kl_mean=0.08, verdict="warn"))
    assert parity is not None
    params = qw._parity_params("q", parity, "warn")
    assert params is not None
    assert params["reason"] == "kl"


def test_a_report_the_job_turned_into_a_link_is_ignored(tmp_path: Path) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"parity": {"verdict": "pass"}}))
    link = tmp_path / "report.json"
    link.symlink_to(secret)
    assert qw._parity(link) is None


def test_missing_numbers_give_no_parity_text(tmp_path: Path) -> None:
    """No "nan%" on a card and no NaN in a JSON event: the generic exit error is used instead."""
    parity = qw._parity(_report(tmp_path, verdict="fail"))
    assert parity is not None
    assert qw._parity_params("q", parity, "fail") is None
