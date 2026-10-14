"""Distillation fidelity scoring (ADR-022 #6): student answers vs the teacher's held-out answers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "distill_fidelity.py"
ENGINE = {"vllm": "0.31.0", "kv_cache_gb": 8.0, "container_digest": "sha256:b"}


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("distill_fidelity", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["distill_fidelity"] = module
    spec.loader.exec_module(module)
    return module


def _row(i: int, answer: str, student: str | None, finish: str = "stop") -> dict[str, Any]:
    return {"id": str(i), "question": f"q{i}", "image": f"/x/{i}.png", "answer": answer, "student": student,
            "finish_reason": finish}  # fmt: skip


def test_normalize_drops_case_punctuation_and_articles() -> None:
    assert _mod().normalize("  The TOTAL is: 75,000! ") == "total is 75000"


def test_token_f1_is_partial_credit() -> None:
    mod = _mod()
    assert mod.token_f1("total 75000", "the total is 75000") == pytest.approx(2 * 1.0 * (2 / 3) / (1 + 2 / 3))
    assert mod.token_f1("nothing", "total") == 0.0
    assert mod.token_f1("", "") == 1.0


def test_score_reports_means_ordered_cis_and_truncation() -> None:
    rows = [_row(0, "two cats", "Two cats."), _row(1, "two cats", "three", "length"), _row(2, "two cats", "two dogs")]
    report = _mod().score(rows)
    assert report["questions"] == 3
    assert report["exact_match_pct"] == pytest.approx(33.33, abs=0.01)
    assert report["token_f1_pct"] == pytest.approx(100 * (1 + 0 + 0.5) / 3, abs=0.01)
    assert report["truncated_pct"] == pytest.approx(33.33, abs=0.01)
    lo, hi = report["token_f1_ci95"]
    assert 0 <= lo <= report["token_f1_pct"] <= hi <= 100


def test_a_failed_request_scores_zero_never_drops_out() -> None:
    """Failing on the hard image must not raise a student's score (bao-tester B1)."""
    report = _mod().score([_row(0, "two cats", "two cats"), _row(1, "red", None)])
    assert (report["questions"], report["failed_requests"], report["exact_match_pct"]) == (2, 1, 50.0)


def _run(rows: list[dict[str, Any]], engine: dict[str, Any] = ENGINE, model: str = "m") -> dict[str, Any]:
    return {"model": model, "engine": engine, "rows": rows}


def test_compare_is_paired_and_names_the_better_student_only_outside_the_ci() -> None:
    mod = _mod()
    good = _run([_row(i, "two cats", "two cats") for i in range(40)], model="qa")
    bad = _run([_row(i, "two cats", "one dog") for i in range(40)], model="orig")
    report = mod.compare(good, bad)
    assert report["delta_token_f1_pct"] == 100.0
    assert report["a_better_on_token_f1"]
    same = mod.compare(good, good)
    assert not same["a_better_on_token_f1"]
    assert not same["b_better_on_token_f1"]


def test_compare_refuses_other_questions_or_another_engine() -> None:
    mod = _mod()
    a = _run([_row(0, "x", "x")])
    with pytest.raises(SystemExit, match="same held-out questions"):
        mod.compare(a, _run([_row(1, "x", "x")]))
    with pytest.raises(SystemExit, match="kv_cache_gb"):
        mod.compare(a, _run([_row(0, "x", "x")], engine={**ENGINE, "kv_cache_gb": 4.0}))


def test_held_out_refuses_the_training_part(tmp_path: Path) -> None:
    from vlm_pipeline.common.dataset_ref import DatasetRef, write_dataset_ref

    ref = DatasetRef(path="/workspace/data/x", split="train", records="/workspace/r.jsonl", records_sha256="0" * 64,
                     val_fraction=0.1, seed=0)  # fmt: skip
    path = tmp_path / "dataset.json"
    write_dataset_ref(path, ref)
    with pytest.raises(SystemExit, match="'train' part"):
        _mod().held_out(tmp_path, path)


def test_score_command_writes_the_report(tmp_path: Path) -> None:
    run_file = tmp_path / "s.json"
    run_file.write_text(json.dumps(_run([_row(0, "two cats", "two cats")])))
    out = tmp_path / "r.json"
    assert _mod().main(["score", str(run_file), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["exact_match_pct"] == 100.0
