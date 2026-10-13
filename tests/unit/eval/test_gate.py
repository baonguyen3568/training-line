"""Verdict thresholds of eval.benchmarks (configs/eval/gates.yaml, agreed with bao-research 2026-10-13)."""

from __future__ import annotations

import pytest

from vlm_pipeline.eval.gate import BenchmarkResult, Verdict, finetune_gate, load_gates, quant_gate


def _cord(**kw: object) -> BenchmarkResult:
    base: dict[str, object] = {
        "benchmark": "cord_test",
        "n": 100,
        "structured_output": True,
        "truncated_pct": 2.0,
        "parse_fail_pct": 0.0,
        "delta": 30.0,
        "delta_ci95": (20.0, 40.0),
    }
    return BenchmarkResult(**{**base, **kw})  # type: ignore[arg-type]


def test_shipped_config_loads() -> None:
    assert load_gates()["schema_version"] == 1


def test_clear_gain_passes() -> None:
    result = finetune_gate([_cord()])
    assert (result.status, result.reasons) == (Verdict.PASS, [])


@pytest.mark.parametrize(
    ("kw", "status", "code"),
    [
        ({"delta_ci95": (-1.0, 5.0)}, Verdict.WARN, "no_clear_gain"),
        ({"delta_ci95": (-9.0, -1.0)}, Verdict.FAIL, "worse_than_base"),
        ({"truncated_pct": 10.0}, Verdict.PASS, None),  # boundary: ≤ 10 passes
        ({"truncated_pct": 15.0}, Verdict.WARN, "truncated_pct_high"),
        ({"truncated_pct": 86.0}, Verdict.FAIL, "truncated_pct_high"),  # P-3 s299 on test
        ({"parse_fail_pct": 7.0}, Verdict.WARN, "parse_fail_pct_high"),
        ({"parse_fail_pct": 21.0}, Verdict.FAIL, "parse_fail_pct_high"),
        ({"n": 20}, Verdict.WARN, "quick_measurement"),
    ],
)
def test_each_threshold(kw: dict[str, object], status: Verdict, code: str | None) -> None:
    result = finetune_gate([_cord(**kw)])
    assert result.status == status
    assert [r.code for r in result.reasons] == ([code] if code else [])


def test_worst_reason_wins_and_comes_first() -> None:
    result = finetune_gate([_cord(n=20, truncated_pct=40.0)])
    assert result.status == Verdict.FAIL
    assert [r.level for r in result.reasons] == [Verdict.FAIL, Verdict.WARN]
    assert result.to_json()["status"] == "FAIL"


def test_without_base_only_absolute_limits_apply() -> None:
    assert finetune_gate([_cord(delta=None, delta_ci95=None)]).status == Verdict.PASS


def test_truncation_ignored_for_unstructured_benchmarks() -> None:
    mmmu = BenchmarkResult("mmmu", n=900, truncated_pct=50.0, delta=1.0, delta_ci95=(0.5, 1.5))
    assert finetune_gate([mmmu]).status == Verdict.PASS


def test_general_forgetting_warns_from_two_benchmarks_never_fails() -> None:
    def drop(name: str) -> BenchmarkResult:
        return BenchmarkResult(name, n=500, delta=-3.0, delta_ci95=(-5.0, -1.0), general=True)

    assert finetune_gate([_cord(), drop("mmmu")]).status == Verdict.PASS
    result = finetune_gate([_cord(), drop("mmmu"), drop("mmbench")])
    assert result.status == Verdict.WARN
    assert [r.code for r in result.reasons] == ["general_forgetting"]


@pytest.mark.parametrize(
    ("deltas", "status"),
    [
        ([-0.5, -1.0], Verdict.PASS),
        ([-1.5, -1.5], Verdict.WARN),
        ([-2.5, -2.5], Verdict.FAIL),
        ([0.0, -6.0], Verdict.FAIL),
    ],
)
def test_fp8(deltas: list[float], status: Verdict) -> None:
    results = [BenchmarkResult(f"b{i}", n=500, delta=d) for i, d in enumerate(deltas)]
    assert quant_gate(results, "fp8").status == status


def test_nvfp4_ocr_is_stricter() -> None:
    ok = [BenchmarkResult("mmmu", n=900, delta=-2.0), BenchmarkResult("ocrbench", n=1000, delta=-2.0)]
    assert quant_gate(ok, "nvfp4").status == Verdict.PASS
    bad = [BenchmarkResult("mmmu", n=900, delta=-2.0), BenchmarkResult("ocrbench", n=1000, delta=-3.5)]
    assert [r.code for r in quant_gate(bad, "nvfp4").reasons] == ["quant_drop_ocr"]


def test_quant_on_few_samples_is_only_indicative() -> None:
    result = quant_gate([BenchmarkResult("cord_test", n=30, delta=0.0)], "fp8")
    assert (result.status, [r.code for r in result.reasons]) == (Verdict.WARN, ["quick_measurement"])


def test_quant_ci_reaching_the_fail_level_asks_for_a_full_run() -> None:
    # P002 A6: the point estimate passes (-0.5), but the paired CI reaches -2 (FP8 FAIL level): WARN, run the full set.
    wide = BenchmarkResult("ocrbench", n=500, delta=-0.5, delta_ci95=(-2.6, 1.6))
    result = quant_gate([wide], "fp8")
    assert result.status == Verdict.WARN
    assert [r.code for r in result.reasons] == ["uncertain_run_full"]
    narrow = BenchmarkResult("ocrbench", n=1000, delta=-0.5, delta_ci95=(-1.4, 0.4))
    assert quant_gate([narrow], "fp8").status == Verdict.PASS


def test_quant_point_estimate_still_decides_fail() -> None:
    bad = BenchmarkResult("ocrbench", n=1000, delta=-2.5, delta_ci95=(-3.5, -1.5))
    result = quant_gate([bad], "fp8")
    assert result.status == Verdict.FAIL
    assert "uncertain_run_full" not in [r.code for r in result.reasons]  # already a FAIL, no extra hedge
