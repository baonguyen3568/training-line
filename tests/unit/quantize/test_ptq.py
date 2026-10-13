"""Host-side parts of the quantize node (B-4): scope checks, memory estimate, parity verdict, container argv."""

from __future__ import annotations

import pytest

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.quantize.config import QuantizeParams
from vlm_pipeline.quantize.ptq import FAIL_KL, FAIL_TOP1_AGREEMENT, MAX_KL, MIN_TOP1_AGREEMENT, parity_stats
from vlm_pipeline.quantize.scope import QuantizeNotSupportedError, check_supported, est_peak_gb, ptq_argv


def _entry(model_id: str):  # type: ignore[no-untyped-def]
    entry = load_catalog().get(model_id)
    assert entry is not None
    return entry


def test_native_official_fp8_model_is_supported() -> None:
    check_supported(_entry("qwen3.5-2b"), QuantizeParams(), "n3")


@pytest.mark.parametrize(
    ("model_id", "params", "reason"),
    [
        ("nemotron-parse-v1.1", QuantizeParams(), "remote_code_model"),
        ("gemma-4-e2b-it", QuantizeParams(), "fp8_not_official"),
        ("qwen3.5-2b", QuantizeParams(quantize_vision=True), "vision_quant"),
        ("qwen3.5-2b", QuantizeParams(kv_cache_quant="fp8"), "kv_cache_quant"),
    ],
)
def test_out_of_scope_requests_are_refused_with_a_reason(model_id: str, params: QuantizeParams, reason: str) -> None:
    with pytest.raises(QuantizeNotSupportedError) as err:
        check_supported(_entry(model_id), params, "n3")
    assert err.value.params["reason"] == reason


def test_memory_estimate_never_below_the_measured_probe() -> None:
    assert est_peak_gb(_entry("qwen3.5-0.8b")) == 20.0  # first node run 2026-10-13: 19.3 GB (run.json, cold caches)
    big = _entry("qwen3.5-9b")
    assert est_peak_gb(big) == round(big.disk_gb * 2 + 16, 1)


def test_parity_verdict() -> None:
    probe = parity_stats([0.99, 0.9245, 0.968], [0.0087, 0.013, 0.004])  # B-4 probe: mean 0.961, min 0.9245
    assert (probe["verdict"], probe["samples"]) == ("pass", 3)
    node_run = parity_stats([0.9346], [0.02144])  # first node run 2026-10-13 (calib 128): below the probe, not broken
    assert node_run["verdict"] == "warn"
    assert parity_stats([MIN_TOP1_AGREEMENT - 0.01], [0.001])["verdict"] == "warn"
    assert parity_stats([0.99], [MAX_KL + 0.001])["verdict"] == "warn"
    assert parity_stats([FAIL_TOP1_AGREEMENT - 0.01], [0.001])["verdict"] == "fail"
    assert parity_stats([0.99], [FAIL_KL + 0.01])["verdict"] == "fail"
    assert parity_stats([], [])["verdict"] == "none"  # no validation split: the benchmark gate decides


def test_container_argv() -> None:
    argv = ptq_argv(model="/workspace/m", data="/workspace/d", out="/workspace/o", fmt="fp8", calib_size=128)
    assert argv[:3] == ("python", "-m", "vlm_pipeline.quantize.ptq")
    assert argv[argv.index("--calib-size") + 1] == "128"
    assert argv[argv.index("--format") + 1] == "fp8"


def test_check_support_covers_the_calibration_wire() -> None:
    from vlm_pipeline.quantize.scope import check_support

    qwen = _entry("qwen3.5-2b")
    check_support("n3", QuantizeParams(), qwen, {"calib_data": "dataset:hub"})
    check_support("n3", QuantizeParams(), None, {"calib_data": "dataset:hub"})  # model unknown yet: wire still checked
    for inputs, reason in (({}, "calib_data_missing"), ({"calib_data": "dataset:local"}, "local_calib")):
        with pytest.raises(QuantizeNotSupportedError) as err:
            check_support("n3", QuantizeParams(), qwen, inputs)
        assert err.value.params["reason"] == reason
    with pytest.raises(QuantizeNotSupportedError) as err:
        check_support("n3", QuantizeParams(), _entry("nemotron-parse-v1.1"), {"calib_data": "dataset:hub"})
    assert err.value.params["reason"] == "remote_code_model"
