"""OCRBench scoring = the reference rule (Yuliang-Liu/MultimodalOCR OCRBench/example.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.eval.ocrbench import TYPE_TOTALS, score_one, score_samples


@pytest.mark.parametrize(
    ("pred", "answers", "dataset", "expected"),
    [
        ("The text says STOP.", "stop", "IIIT5K", 1),  # case-insensitive substring
        ("stop", "Stop sign", "IIIT5K", 0),  # the ANSWER must be inside the prediction, not the reverse
        ("  answer: 12\n34 ", "12 34", "docVQA", 1),  # newline -> space on both sides
        ("x", ["y", "X"], "STVQA", 1),  # any of several answers
        ("", "a", "STVQA", 0),
        ("\\frac { a } { b }", "\\frac{a}{b}", "HME100k", 1),  # HME100k drops spaces...
        ("\\FRAC{a}{b}", "\\frac{a}{b}", "HME100k", 0),  # ...and keeps case (no lower())
    ],
)
def test_reference_rule(pred: str, answers: str | list[str], dataset: str, expected: int) -> None:
    assert score_one(pred, answers, dataset) == expected


def test_fullwidth_folding_is_an_explicit_option() -> None:
    pred = "\uff33\uff34\uff2f\uff30"  # fullwidth "STOP"
    assert score_one(pred, "stop", "ReCTS") == 0  # reference: no folding
    assert score_one(pred, "stop", "ReCTS", fold_fullwidth=True) == 1  # lmms-eval 0.7 behaviour


def test_totals_add_up_to_1000_and_recognition_is_300() -> None:
    assert sum(TYPE_TOTALS.values()) == 1000
    assert sum(list(TYPE_TOTALS.values())[:6]) == 300


def test_score_samples_aggregates_like_the_reference() -> None:
    rows: list[dict[str, Any]] = [
        {"prediction": "cat", "answer": "cat", "dataset": "IIIT5K", "question_type": "Regular Text Recognition"},
        {"prediction": "dog", "answer": "cat", "dataset": "IIIT5K", "question_type": "Regular Text Recognition"},
        {"prediction": "42", "answer": ["42"], "dataset": "docVQA", "question_type": "Doc-oriented VQA"},
        {"prediction": "a+b", "answer": "a + b", "dataset": "HME100k",
         "question_type": "Handwritten Mathematical Expression Recognition"},
    ]  # fmt: skip
    s = score_samples(rows)
    assert (s.final, s.n_samples, s.recognition) == (3, 4, 1)
    assert s.accuracy == 75.0
    assert s.by_type["Doc-oriented VQA"] == 1


def test_unknown_question_type_is_refused() -> None:
    with pytest.raises(ValueError, match="question_type"):
        score_samples([{"prediction": "a", "answer": "a", "dataset": "x", "question_type": "Riddles"}])


def test_committed_subset_is_500_rows_stratified_like_the_set() -> None:
    import json
    from collections import Counter

    from vlm_pipeline.eval.ocrbench import SUBSET_PATH

    spec = json.loads(SUBSET_PATH.read_text())
    assert spec["revision"] == "92a54bd1384387c178d5a07140a2d85e0a3d12e1"  # the catalog pin
    assert len({s["row"] for s in spec["samples"]}) == 500
    assert Counter({t: v["sampled"] for t, v in spec["strata"].items()}) == {t: n // 2 for t, n in TYPE_TOTALS.items()}


def test_subset_detects_a_reordered_dataset(tmp_path: Path) -> None:
    import hashlib
    import json

    from vlm_pipeline.eval.ocrbench import load_subset

    questions = ["what is written?", "what is the total?", "read the number"]
    sample = [{"row": 1, "q_sha256": hashlib.sha256(questions[1].encode()).hexdigest()[:16]}]
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"samples": sample}))
    assert load_subset(questions, path) == [1]
    with pytest.raises(ValueError, match="differs"):
        load_subset(questions[1:] + questions[:1], path)  # rotated: row 1 is now another question
    with pytest.raises(ValueError, match="beyond"):
        load_subset(questions[:1], path)


def test_paired_accuracy_delta_and_ci() -> None:
    from vlm_pipeline.eval.ocrbench import paired_accuracy

    ref = [1] * 80 + [0] * 20  # 80%
    cand = [1] * 75 + [0] * 25  # 75%: 5 regressions, no gain
    result = paired_accuracy(cand, ref)
    assert result["delta"] == -5.0
    low, high = result["ci95"]
    assert low <= -5.0 <= high
    assert high <= 0  # only regressions: the CI cannot reach a gain
    assert paired_accuracy(ref, ref)["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="same length"):
        paired_accuracy([1], [1, 0])
