from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "score_cord_runs.py"
GT = {"menu": {"nm": "Tea", "price": "5"}}
GOOD = "<s_menu><s_nm>Tea</s_nm><s_price>5</s_price></s_menu>"
HALF = "<s_menu><s_nm>Tea</s_nm><s_price>6</s_price></s_menu>"


def _scorer() -> Any:
    spec = importlib.util.spec_from_file_location("score_cord_runs", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(sequences: list[str], **fields: Any) -> dict[int, dict[str, Any]]:
    base = {"gt_parse": GT, "truncated": False, "split": "test"}
    return {i: {"index": i, "sequence": s, **base, **fields} for i, s in enumerate(sequences)}


def test_metrics_and_determinism_check() -> None:
    scorer = _scorer()
    a, b = _records([GOOD, GOOD]), _records([GOOD, HALF])
    assert scorer.metrics(list(a.values()))["field_f1"] == 100.0
    assert scorer.metrics(list(b.values()))["field_f1"] == 75.0
    assert scorer.different_sequences(a, b)["different"] == [1]
    assert scorer.different_sequences(a, a)["different"] == []


def test_paired_bootstrap_reports_observed_difference_inside_its_ci() -> None:
    scorer = _scorer()
    a, b = _records([GOOD] * 10), _records([GOOD] * 5 + [HALF] * 5)
    result = scorer.paired_bootstrap(a, b, 200)
    # micro F1 over 2 fields per image: b = (5*2 + 5*1) / 20 = 75 (a HALF image gets 1 of 2 right), a = 100
    assert result["observed"] == -25.0
    low, high = result["ci95"]
    assert low <= -25.0 <= high
    assert high < 0
    assert result["n_only_a"] == result["n_only_b"] == 0


def test_runs_on_different_images_are_refused() -> None:
    """Same index, different image (test vs validation): pairing must not silently succeed."""
    scorer = _scorer()
    a = _records([GOOD, GOOD])
    b = _records([GOOD, GOOD], split="validation")
    with pytest.raises(SystemExit, match="not the same image"):
        scorer.paired_bootstrap(a, b, 200)
    c = _records([GOOD, GOOD])
    c[1]["gt_parse"] = {"menu": {"nm": "Coffee"}}
    with pytest.raises(SystemExit, match="not the same image"):
        scorer.different_sequences(a, c)


def test_partial_runs_need_an_explicit_flag() -> None:
    scorer = _scorer()
    full, partial = _records([GOOD] * 4), _records([GOOD] * 2)
    with pytest.raises(SystemExit, match="different images"):
        scorer.paired_bootstrap(full, partial, 200)
    result = scorer.different_sequences(full, partial, allow_partial=True)
    assert (result["n_common"], result["n_only_a"], result["n_only_b"]) == (2, 2, 0)


def test_a_run_mixing_decoding_settings_is_refused() -> None:
    scorer = _scorer()
    rows = list(_records([GOOD, GOOD]).values())
    rows[0]["decoding"] = {"repetition_penalty": 1.0}
    rows[1]["decoding"] = {"repetition_penalty": 1.1}
    with pytest.raises(SystemExit, match="mixes 2 values of decoding"):
        scorer.run_settings(rows, "r")


def test_old_records_report_unrecorded_fields() -> None:
    settings = _scorer().run_settings(list(_records([GOOD]).values()), "r")
    assert settings["batch_size"] == "unrecorded"
    assert settings["worker_sha256"] == ["unrecorded"]


def test_duplicate_indices_are_refused(tmp_path: Path) -> None:
    line = json.dumps({"index": 0, "sequence": GOOD, "gt_parse": GT, "truncated": False})
    (tmp_path / "pred.jsonl").write_text(f"{line}\n{line}\n")
    with pytest.raises(SystemExit, match="twice"):
        _scorer().load_run(tmp_path)
