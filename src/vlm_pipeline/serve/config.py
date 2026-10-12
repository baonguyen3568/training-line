"""Parameters of the serving engine: trial inference (``serve.vllm``) and speed benchmark (``bench.perf``).

The server always binds 127.0.0.1 (not a parameter, ADR-003). ``gpu_memory_utilization`` above 0.3 is allowed only
when preflight agrees (ADR-001). Defaults: node-params-v1.md §5 and §7.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vlm_pipeline.common.params import Params, ui

ANSWER_TOKENS = {"short": 256, "medium": 1024, "long": 4096}


class InferenceParams(Params):
    answer_length: Literal["short", "medium", "long"] = Field(
        "medium", json_schema_extra=ui(order=0, widget="segmented", option_keys="answer_length")
    )
    engine: Literal["vllm", "hf"] = Field("vllm", json_schema_extra=ui(advanced=True, option_keys="serve_engine"))
    temperature: float = Field(0.0, ge=0.0, le=1.5, json_schema_extra=ui(advanced=True))
    top_p: float = Field(1.0, ge=0.1, le=1.0, json_schema_extra=ui(advanced=True))
    gpu_memory_utilization: float = Field(0.3, ge=0.2, le=0.5, json_schema_extra=ui(advanced=True))
    max_model_len: int = Field(8192, ge=2048, le=32768, json_schema_extra=ui(advanced=True))

    @property
    def max_tokens(self) -> int:
        return ANSWER_TOKENS[self.answer_length]


class BenchParams(Params):
    scenario: Literal["single_user", "multi_user"] = Field(
        "single_user", json_schema_extra=ui(order=0, widget="segmented", summary=True, option_keys="bench_scenario")
    )
    compare_with: Literal["none", "base", "unquantized"] = Field(
        "base", json_schema_extra=ui(order=1, option_keys="compare_with")
    )
    num_requests: int = Field(50, ge=10, le=500, json_schema_extra=ui(advanced=True))
    output_tokens: int = Field(256, ge=64, le=2048, json_schema_extra=ui(advanced=True))
    gpu_memory_utilization: float = Field(0.3, ge=0.2, le=0.5, json_schema_extra=ui(advanced=True))

    @property
    def concurrency(self) -> int:
        return 1 if self.scenario == "single_user" else 4
