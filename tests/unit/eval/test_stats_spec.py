"""Spec tests for paired comparison statistics used by the P002 gates (G2/G3).

Target: ``vlm_pipeline.eval.stats.paired_bootstrap_ci(*, baseline, candidate, n_resamples, seed, confidence)``
returning ``PairedCI(delta, lo, hi)`` (NamedTuple) with ``delta = mean(candidate - baseline)`` per sample.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.eval.stats in P2-6"
)


def test_identical_runs_have_zero_delta() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    a = [1, 0, 1, 1, 0, 1]
    ci = paired_bootstrap_ci(baseline=a, candidate=a, n_resamples=500, seed=0, confidence=0.95)
    assert (ci.delta, ci.lo, ci.hi) == (0.0, 0.0, 0.0)


def test_direction_is_candidate_minus_baseline() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    ci = paired_bootstrap_ci(baseline=[0] * 20, candidate=[1] * 20, n_resamples=500, seed=0, confidence=0.95)
    assert ci.delta == pytest.approx(1.0)
    assert ci.lo == pytest.approx(1.0)
    assert ci.hi == pytest.approx(1.0)
    worse = paired_bootstrap_ci(baseline=[1] * 20, candidate=[0] * 20, n_resamples=500, seed=0, confidence=0.95)
    assert worse.delta == pytest.approx(-1.0)


def test_ci_contains_delta_and_is_seed_deterministic() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    a = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 5
    b = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0] * 5
    first = paired_bootstrap_ci(baseline=a, candidate=b, n_resamples=1000, seed=123, confidence=0.95)
    second = paired_bootstrap_ci(baseline=a, candidate=b, n_resamples=1000, seed=123, confidence=0.95)
    assert first == second
    assert first.delta == pytest.approx(0.2)
    assert first.lo <= first.delta <= first.hi


def test_unpaired_lengths_raise() -> None:
    from vlm_pipeline.eval.stats import paired_bootstrap_ci

    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_ci(baseline=[1, 0], candidate=[1], n_resamples=10, seed=0, confidence=0.95)
