"""Spec tests for ANLS (DocVQA / InfoVQA), target module ``vlm_pipeline.eval.scoring``.

Reference behaviour = the harnesses that produce the numbers our G1 gate compares against
(P002), NOT the paper formula verbatim. Both agree with each other:
- VLMEvalKit ``vlmeval/dataset/utils/vqa_eval.py::anls_compute`` + ``hit_calculate`` (read @21e52b8df8ce, 2026-10-12)
- lmms-eval ``lmms_eval/api/metrics.py::anls`` (read @32d527db1e13, 2026-10-12)
Behaviour encoded here:
1. Normalise both strings with ``" ".join(s.strip().lower().split())`` before the edit distance.
2. Divide the distance by ``max(len(raw_target), len(raw_prediction))`` on the RAW strings (harness quirk).
3. Similarity = 1 - normalised distance; best over all ground-truth answers.
4. Threshold: similarity ``< 0.5`` -> 0, so similarity exactly 0.5 is KEPT. This differs from the
   paper (Biten et al. 2019, Eq. 1, https://arxiv.org/abs/1905.13648: ``NL < tau`` else 0) and from
   pix2struct ``anls_metric``; we follow the harnesses because G1 reproduces their numbers.
Scale: 0-1 everywhere in this module; x100 happens only in the report layer.

Marked ``xfail(strict=True, raises=ImportError)``: only a missing module/function is tolerated;
once implemented, a wrong value is a real failure and an unexpected pass forces removing the marker.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.eval.scoring.anls in P2-6"
)


@pytest.mark.parametrize(
    ("answers", "prediction", "expected"),
    [
        (["hello"], "hello", 1.0),
        (["Hello"], "hello", 1.0),  # case-insensitive
        (["2019"], "2019 ", 1.0),  # outer whitespace stripped
        (["new  york"], "New York", 1.0),  # inner whitespace collapsed
        (["hello"], "hallo", 0.8),  # distance 1 / max raw len 5
        (["abcde"], "abxye", 0.6),  # 2 / 5
        (["abcd"], "abxy", 0.5),  # exactly 0.5 is kept (harness), paper would give 0
        (["abcdefg"], "abxyzfg", 4 / 7),  # 3 / 7 -> 0.571
        (["abcdef"], "axyzwf", 0.0),  # 4 / 6 -> similarity 0.333 < 0.5
        (["abc"], "xyz", 0.0),  # similarity 0 < 0.5
        (["abc"], "", 0.0),  # empty prediction
        (["ax"], "  ab ", 0.8),  # raw-length quirk: distance 1 / max(len("ax"), len("  ab ")) = 1/5
        (["1999", "nineteen ninety nine"], "1999", 1.0),  # best over ground truths
        (["", "x"], "", 1.0),  # both empty -> distance 0 -> similarity 1
    ],
)
def test_anls(answers: list[str], prediction: str, expected: float) -> None:
    from vlm_pipeline.eval.scoring import anls

    assert anls(answers, prediction) == pytest.approx(expected)


def test_anls_mean() -> None:
    from vlm_pipeline.eval.scoring import anls_mean

    assert anls_mean([["hello"], ["abc"]], ["hallo", "xyz"]) == pytest.approx((0.8 + 0.0) / 2)


def test_anls_mean_length_mismatch_raises() -> None:
    from vlm_pipeline.eval.scoring import anls_mean

    with pytest.raises(ValueError, match="length"):
        anls_mean([["a"], ["b"]], ["a"])
