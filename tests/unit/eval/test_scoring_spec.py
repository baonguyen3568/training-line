"""Spec tests for VLM benchmark scoring functions (ANLS, relaxed accuracy, MC parsing).

These encode the reference behaviour of the official / de-facto implementations so that our
scorer reproduces published numbers. They are marked ``xfail(strict=True)`` until the module
exists (P2-6); once implemented, an XPASS fails the run, which is the signal to drop the marker.

Sources:
- ANLS: Biten et al., "Scene Text Visual Question Answering", ICCV 2019, Eq. 1
  (https://arxiv.org/abs/1905.13648); DocVQA uses theta=0.5 (https://arxiv.org/abs/2007.00398).
  Reference code: google-research/pix2struct ``pix2struct/metrics.py::anls_metric`` +
  ``aggregate_metrics`` (lower-cased target and prediction, max over ground truths, x100).
- Relaxed accuracy: Masry et al., "ChartQA", ACL Findings 2022, end of Sec. 5.1
  (https://arxiv.org/abs/2203.10244): numeric answers within 5% relative error are correct,
  other answers require an exact match. Reference code: pix2struct ``relaxed_correctness``
  (``"x%"`` -> x/100; target == 0 falls back to string compare; string compare is case-insensitive).
- MC parsing: MMMU official ``mmmu/utils/eval_utils.py::parse_multi_choice_response`` /
  ``eval_multi_choice`` (https://github.com/MMMU-Benchmark/MMMU).
"""

from __future__ import annotations

import pytest

SPEC = pytest.mark.xfail(strict=True, reason="spec: bao-backend implements vlm_pipeline.eval.scoring in P2-6")

pytestmark = SPEC


# --------------------------------------------------------------------------------------- ANLS
@pytest.mark.parametrize(
    ("target", "prediction", "expected"),
    [
        ("hello", "hello", 1.0),
        ("Hello", "hello", 1.0),  # case-insensitive (pix2struct normalize_fn=lower)
        ("hello", "hallo", 0.8),  # Levenshtein 1 / max_len 5
        ("abcde", "abxye", 0.6),  # 2 / 5 = 0.4 < 0.5 -> 1 - 0.4
        ("abcd", "abxy", 0.0),  # 2 / 4 = 0.5 is NOT < theta -> 0
        ("abc", "", 0.0),  # empty prediction
        ("2019", "2019 ", pytest.approx(0.8)),  # no implicit strip in the reference; trailing space costs 1/5
    ],
)
def test_anls_pair(target: str, prediction: str, expected: float) -> None:
    from vlm_pipeline.eval.scoring import anls

    assert anls(target, prediction) == pytest.approx(expected)


def test_anls_max_over_ground_truths() -> None:
    from vlm_pipeline.eval.scoring import anls_score

    # One question with two accepted answers: best match wins.
    assert anls_score([["1999", "nineteen ninety nine"]], ["1999"]) == pytest.approx(100.0)


def test_anls_score_is_mean_times_100() -> None:
    from vlm_pipeline.eval.scoring import anls_score

    targets = [["hello"], ["abcd"]]
    predictions = ["hallo", "abxy"]
    assert anls_score(targets, predictions) == pytest.approx((0.8 + 0.0) / 2 * 100)


def test_anls_length_mismatch_raises() -> None:
    from vlm_pipeline.eval.scoring import anls_score

    with pytest.raises(ValueError, match="length"):
        anls_score([["a"], ["b"]], ["a"])


# ---------------------------------------------------------------------------- Relaxed accuracy
@pytest.mark.parametrize(
    ("target", "prediction", "expected"),
    [
        ("10", "10.4", True),  # 4% relative error
        ("10", "10.5", True),  # exactly 5% -> inclusive (<=)
        ("10", "10.6", False),  # 6%
        ("-20", "-19.5", True),  # 2.5%, sign handled via abs(target)
        ("50%", "0.5", True),  # percent converted to fraction
        ("0.5", "50%", True),
        ("0", "0.01", False),  # target 0 -> string compare (reference behaviour)
        ("0", "0", True),
        ("Yes", "yes", True),  # non-numeric: case-insensitive exact match
        ("Yes", "no", False),
        ("2017", "2017.0", True),
    ],
)
def test_relaxed_correctness(target: str, prediction: str, expected: bool) -> None:
    from vlm_pipeline.eval.scoring import relaxed_correctness

    assert relaxed_correctness(target, prediction) is expected


# ------------------------------------------------------------------------------- MC parsing
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
        ("After looking carefully the animal shown is clearly a dog", "B"),  # >5 words, option content
    ],
)
def test_parse_multi_choice(response: str, expected: str) -> None:
    from vlm_pipeline.eval.scoring import parse_multi_choice

    assert parse_multi_choice(response, CHOICES, INDEX2ANS) == expected


def test_parse_multi_choice_unparseable_is_deterministic() -> None:
    """Official MMMU falls back to ``random.choice`` with a module-level seed, so the result depends on
    call order. We require a per-input deterministic result (None = unparseable, scored as wrong);
    the scorer must report the unparseable rate so the gap to the official random fallback
    (expected +rate/n_choices) can be accounted for in the G1 reproduction gate."""
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
