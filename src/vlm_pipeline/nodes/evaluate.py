"""Evaluation nodes: accuracy (``eval.benchmarks``) and speed/memory (``bench.perf``)."""

from __future__ import annotations

from vlm_pipeline.eval.config import EvalParams
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec
from vlm_pipeline.serve.config import BenchParams

EVAL_BENCHMARKS = NodeSpec(
    type_id="eval.benchmarks",
    version="1.0.0",
    category="eval",
    icon="eval",
    inputs={"model": PortSpec(type="Model"), "data": PortSpec(type="Dataset", optional=True)},
    outputs={"report": PortSpec(type="EvalReport")},
    params_model=EvalParams,
    runner="container",
)

BENCH_PERF = NodeSpec(
    type_id="bench.perf",
    version="1.0.0",
    category="eval",
    icon="speed",
    inputs={"model": PortSpec(type="Model")},
    outputs={"report": PortSpec(type="BenchReport")},
    params_model=BenchParams,
    # Timings depend on host load at run time; a cached result would be misleading.
    cacheable=False,
    runner="container",
)
