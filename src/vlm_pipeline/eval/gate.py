"""User-facing verdict of an evaluation (node ``eval.benchmarks``): PASS / WARN / FAIL with machine-readable reasons.

Thresholds live in ``configs/eval/gates.yaml`` (agreed with bao-research, P002 B3/A4). Pure Python, host-safe.
Each reason carries a ``code`` the UI translates (``eval_gate.<code>``) plus the numbers behind it, so the verdict is
explainable without reading the code. The overall status is the worst reason; no reason at all means PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml

GATES_PATH = Path(__file__).resolve().parents[3] / "configs" / "eval" / "gates.yaml"


class Verdict(IntEnum):
    PASS = 0
    WARN = 1
    FAIL = 2


@dataclass(frozen=True)
class Reason:
    code: str
    level: Verdict
    metric: str
    value: float | list[float] | None
    threshold: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level.name,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark of the evaluated model, primary metric on 0-100, oriented so that higher is better."""

    benchmark: str
    n: int
    structured_output: bool = False
    truncated_pct: float | None = None
    parse_fail_pct: float | None = None
    delta: float | None = None  # vs base / parent on the same samples
    delta_ci95: tuple[float, float] | None = None
    general: bool = False  # a general-knowledge benchmark (forgetting check), not the target task


@dataclass(frozen=True)
class GateResult:
    status: Verdict
    reasons: list[Reason] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status.name, "reasons": [r.to_json() for r in self.reasons]}


def load_gates(path: Path = GATES_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version {config.get('schema_version')!r}")
    return dict(config)


def _rate(metric: str, value: float | None, limits: dict[str, float]) -> list[Reason]:
    if value is None:
        return []
    if value > limits["fail_above"]:
        return [Reason(f"{metric}_high", Verdict.FAIL, metric, value, limits["fail_above"])]
    if value > limits["warn_above"]:
        return [Reason(f"{metric}_high", Verdict.WARN, metric, value, limits["warn_above"])]
    return []


def _structured(result: BenchmarkResult, gates: dict[str, Any]) -> list[Reason]:
    if not result.structured_output:
        return []
    limits = gates["structured_output"]
    return _rate("truncated_pct", result.truncated_pct, limits["truncated_pct"]) + _rate(
        "parse_fail_pct", result.parse_fail_pct, limits["parse_fail_pct"]
    )


def finetune_gate(results: list[BenchmarkResult], gates: dict[str, Any] | None = None) -> GateResult:
    """Fine-tune vs base (``compare_with=base``) or alone (no delta): see configs/eval/gates.yaml."""
    gates = gates or load_gates()
    rules = gates["finetune_vs_base"]
    reasons: list[Reason] = []
    forgetting = []
    for r in results:
        reasons += _structured(r, gates)
        if r.delta_ci95 is not None:
            low, high = r.delta_ci95
            if r.general:
                if high < 0:
                    forgetting.append(r.benchmark)
                continue
            if high < 0:
                reasons.append(Reason("worse_than_base", Verdict.FAIL, r.benchmark, [low, high], 0.0))
            elif low <= 0:
                reasons.append(Reason("no_clear_gain", Verdict.WARN, r.benchmark, [low, high], 0.0))
        if not r.general and r.n < rules["min_samples_for_pass"]:
            reasons.append(Reason("quick_measurement", Verdict.WARN, "n", r.n, rules["min_samples_for_pass"]))
    if len(forgetting) >= rules["general_forgetting_warn_min_benchmarks"]:
        reasons.append(Reason("general_forgetting", Verdict.WARN, ",".join(sorted(forgetting)), len(forgetting)))
    return _result(reasons)


def quant_gate(results: list[BenchmarkResult], precision: str, gates: dict[str, Any] | None = None) -> GateResult:
    """Quantized model vs its bf16 parent (P002 B3); ``precision`` is a key of ``quant_vs_parent``.

    PASS / WARN / FAIL follow the point estimates (P002 §A6). The paired CI95 of each benchmark's delta is still
    read: a CI that reaches the FAIL level while the point estimate does not adds WARN ``uncertain_run_full``
    (with 0/1 scores the CI half-width is about 1.96 * sqrt(discordant share / n): 500 questions cannot separate
    -1 from 0).
    TODO(multi-benchmark suites): -2 is the level of the MEAN; with several benchmarks the uncertain label should
    use the CI of the mean against -2 and each benchmark's CI against fail_below_any (-5). Correct today because
    the quantize template's suite is OCRBench alone.
    TODO: benchmark ids ("ocrbench", "docvqa") and structured_output move to the benchmark registry.
    """
    gates = gates or load_gates()
    rules = gates["quant_vs_parent"][precision]
    reasons: list[Reason] = []
    deltas = [r.delta for r in results if r.delta is not None]
    min_n = gates["finetune_vs_base"]["min_samples_for_pass"]
    for r in results:
        reasons += _structured(r, gates)
        if r.n < min_n:
            reasons.append(Reason("quick_measurement", Verdict.WARN, "n", r.n, min_n))
        if r.delta is None:
            continue
        if r.delta < rules["fail_below_any"]:
            reasons.append(Reason("quant_drop_any", Verdict.FAIL, r.benchmark, r.delta, rules["fail_below_any"]))
        elif (
            r.delta_ci95 is not None and r.delta >= rules["warn_min_mean"] and r.delta_ci95[0] < rules["warn_min_mean"]
        ):
            reasons.append(
                Reason("uncertain_run_full", Verdict.WARN, r.benchmark, list(r.delta_ci95), rules["warn_min_mean"])
            )
        ocr_limit = rules.get("ocr_docvqa_fail_below")
        if ocr_limit is not None and r.benchmark in {"ocrbench", "docvqa"} and r.delta < ocr_limit:
            reasons.append(Reason("quant_drop_ocr", Verdict.FAIL, r.benchmark, r.delta, ocr_limit))
    if deltas:
        mean = sum(deltas) / len(deltas)
        if mean < rules["warn_min_mean"]:
            reasons.append(
                Reason("quant_drop_mean", Verdict.FAIL, "mean_delta", round(mean, 2), rules["warn_min_mean"])
            )
        elif mean < rules["pass_min_mean"]:
            reasons.append(
                Reason("quant_drop_mean", Verdict.WARN, "mean_delta", round(mean, 2), rules["pass_min_mean"])
            )
    return _result(reasons)


def _result(reasons: list[Reason]) -> GateResult:
    status = max((r.level for r in reasons), default=Verdict.PASS)
    return GateResult(status, sorted(reasons, key=lambda r: -r.level))
