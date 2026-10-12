"""Spec tests for ``compute_cache_key(spec, params, input_hashes, image_digest) -> str`` (P009 §5).

ASSUMED: ``vlm_pipeline.graph.cache.compute_cache_key`` and ``make_test_registry().get(type_id)`` (registry described in
test_validate_spec.py). UI-only fields (canvas position) are tested at executor level, not here.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.graph.cache (P009) in P2"
)

IMG = "sha256:" + "0" * 64


def _key(params: dict, inputs: dict, image: str = IMG, type_id: str = "test.train") -> str:
    from vlm_pipeline.graph.cache import compute_cache_key
    from vlm_pipeline.graph.testing import make_test_registry

    return compute_cache_key(make_test_registry().get(type_id), params, inputs, image)


def test_deterministic_and_order_independent() -> None:
    a = _key({"max_steps": 10, "lr": 1e-4}, {"data": "h1", "base_model": "h2"})
    b = _key({"lr": 1e-4, "max_steps": 10}, {"base_model": "h2", "data": "h1"})
    assert a == b


@pytest.mark.parametrize(
    "change",
    [
        {"params": {"max_steps": 11, "lr": 1e-4}},
        {"inputs": {"data": "h1-changed", "base_model": "h2"}},
        {"image": "sha256:" + "1" * 64},
    ],
)
def test_any_relevant_change_changes_key(change: dict) -> None:
    base = {"params": {"max_steps": 10, "lr": 1e-4}, "inputs": {"data": "h1", "base_model": "h2"}, "image": IMG}
    assert _key(**base) != _key(**{**base, **change})


def test_omitted_default_equals_explicit_default() -> None:
    """Key is computed on validated params with defaults filled in (test.train: max_steps=100, lr=1e-4)."""
    assert _key({}, {"data": "h1"}) == _key({"max_steps": 100, "lr": 1e-4}, {"data": "h1"})


def test_coerced_values_give_same_key() -> None:
    """``lr`` is a float param: 1 and 1.0 validate to the same value, so the key must not differ."""
    assert _key({"lr": 1}, {"data": "h1"}) == _key({"lr": 1.0}, {"data": "h1"})
