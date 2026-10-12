"""Parameters of the accuracy-evaluation engine (node ``eval.benchmarks``). Host-safe.

Defaults: .team/knowledge/node-params-v1.md §4, ADR-002 (suites, levels, paired comparison).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vlm_pipeline.common.params import Params, ui

# "auto" = what the template / model catalog suggests (Parse -> CORD test, VLM -> documents).
EvalSuite = Literal["auto", "cord_test", "documents", "charts", "knowledge", "perception", "math", "video", "my_val"]
EvalLevel = Literal["quick", "standard", "full"]
CompareWith = Literal["none", "base", "previous"]


class EvalParams(Params):
    suite: EvalSuite = Field("auto", json_schema_extra=ui(order=0, summary=True, option_keys="eval_suite"))
    level: EvalLevel = Field("quick", json_schema_extra=ui(order=1, widget="segmented", option_keys="eval_level"))
    compare_with: CompareWith = Field("base", json_schema_extra=ui(order=2, option_keys="compare_with"))
    max_new_tokens: int = Field(1024, ge=64, le=16384, json_schema_extra=ui(advanced=True))
    thinking: bool = Field(False, json_schema_extra=ui(advanced=True))
