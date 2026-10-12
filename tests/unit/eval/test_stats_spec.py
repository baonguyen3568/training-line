"""Spec tests for paired comparison statistics used by the P002 gates (G2/G3).

Marked ``xfail(strict=True)`` until ``vlm_pipeline.eval.stats`` exists (P2-6).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(strict=True, reason="spec: bao-backend implements vlm_pipeline.eval.stats in P2-6")


def test_identical_runs_have_zero_delta() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    a = [1, 0, 1, 1, 0, 1]
    delta, lo, hi = paired_bootstrap_ci(a, a, n_resamples=500, seed=0)
    assert (delta, lo, hi) == (0.0, 0.0, 0.0)


def test_all_improved_gives_degenerate_ci_at_one() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    delta, lo, hi = paired_bootstrap_ci([0] * 20, [1] * 20, n_resamples=500, seed=0)
    assert delta == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_ci_contains_delta_and_is_seed_deterministic() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    a = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 5
    b = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0] * 5
    first = paired_bootstrap_ci(a, b, n_resamples=1000, seed=123)
    second = paired_bootstrap_ci(a, b, n_resamples=1000, seed=123)
    assert first == second
    delta, lo, hi = first
    assert delta == pytest.approx(0.2)
    assert lo <= delta <= hi


def test_unpaired_lengths_raise() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_ci([1, 0], [1], n_resamples=10, seed=0)
