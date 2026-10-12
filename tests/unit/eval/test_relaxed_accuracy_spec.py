"""Spec tests for ChartQA relaxed accuracy, target ``vlm_pipeline.eval.scoring.relaxed_correctness``.

Definition: Masry et al., "ChartQA", ACL Findings 2022, end of Sec. 5.1 (https://arxiv.org/abs/2203.10244):
numeric answers within 5% relative error are correct, others need an exact match.
Reference code (identical in both): google-research/pix2struct ``metrics.py::relaxed_correctness`` (@110d6666e881) and
VLMEvalKit ``vqa_eval.py::relaxed_correctness`` (@21e52b8df8ce), both read 2026-10-12:
``"x%"`` -> x/100; numeric branch only when target is
non-zero (target 0 falls back to string compare); string compare is case-insensitive. The harness strips
the prediction before calling it (VLMEvalKit ``process_line``), so we strip too.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="spec: bao-backend implements vlm_pipeline.eval.scoring.relaxed_correctness in P2-6",
)


@pytest.mark.parametrize(
    ("target", "prediction", "expected"),
    [
        ("10", "10.4", True),  # 4% relative error
        # Exactly 5% relies on exact float math: implementation must compute abs(p - t) / abs(t)
        # like the reference (0.5 / 10 == 0.05); a different order can be 1 ulp off.
        ("10", "10.5", True),
        ("10", "10.6", False),  # 6%
        ("-20", "-19.5", True),  # 2.5%, abs(target) in the denominator
        ("50%", "0.5", True),  # percent converted to fraction
        ("0.5", "50%", True),
        ("0", "0.01", False),  # target 0 -> string compare (reference behaviour)
        ("0", "0", True),
        ("Yes", "yes", True),  # non-numeric: case-insensitive exact match
        ("Yes", "no", False),
        ("2017", "2017.0", True),
        ("12", " 12 ", True),  # prediction stripped (harness process_line)
    ],
)
def test_relaxed_correctness(target: str, prediction: str, expected: bool) -> None:
    from vlm_pipeline.eval.scoring import relaxed_correctness

    assert relaxed_correctness(target, prediction) is expected
