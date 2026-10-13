"""OCRBench scoring = the reference rule (Yuliang-Liu/MultimodalOCR OCRBench/example.py)."""

from __future__ import annotations

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
