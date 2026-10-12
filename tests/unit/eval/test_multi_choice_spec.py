"""Spec tests for multiple-choice answer parsing (MMMU / MMBench / AI2D style).

Target: ``vlm_pipeline.eval.scoring.parse_multi_choice`` and ``eval_multi_choice``.
Reference: MMMU official ``mmmu/utils/eval_utils.py::parse_multi_choice_response`` / ``eval_multi_choice``
(https://github.com/MMMU-Benchmark/MMMU @268471d0d488, read 2026-10-12): strip leading/trailing punctuation, look for ``(X)``, then
`` X `` (case-sensitive), then option text when the response has > 5 words; with several candidates
the one occurring LAST wins.
Deliberate deviation: the official parser falls back to ``random.choice`` seeded once at import, so the
result depends on call order. We return ``None`` (scored wrong) and report ``unparseable_rate``; the
expected gap to the official number (+unparseable_rate / n_choices) is accounted for in gate G1.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.eval.scoring MC parsing in P2-6"
)

CHOICES = ["A", "B", "C", "D"]
INDEX2ANS = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("(B)", "B"),
        ("B", "B"),
        ("The answer is C.", "C"),  # trailing punctuation stripped before matching
        ("I think A is wrong, the answer is D", "D"),  # several candidates -> last occurrence wins
        ("(A) looks plausible but (C) is correct", "C"),
        ("After looking carefully the animal shown is clearly a dog", "B"),  # > 5 words, option text
    ],
)
def test_parse_multi_choice(response: str, expected: str) -> None:
    from vlm_pipeline.eval.scoring import parse_multi_choice

    assert parse_multi_choice(response, CHOICES, INDEX2ANS) == expected


def test_parse_multi_choice_unparseable_is_none_and_deterministic() -> None:
    from vlm_pipeline.eval.scoring import parse_multi_choice

    response = "I am not sure."
    assert parse_multi_choice(response, CHOICES, INDEX2ANS) is None
    assert parse_multi_choice(response, CHOICES, INDEX2ANS) is None


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [("B", "B", True), ("B", "C", False), (["B", "C"], "C", True), ("B", None, False)],
)
def test_eval_multi_choice(gold: str | list[str], pred: str | None, expected: bool) -> None:
    from vlm_pipeline.eval.scoring import eval_multi_choice

    assert eval_multi_choice(gold, pred) is expected
