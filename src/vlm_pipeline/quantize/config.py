"""Parameters of the quantization engine (ModelOpt PTQ, node ``quantize.modelopt_ptq``). Host-safe.

v1 offers FP8 only (the format with official ModelOpt support for the v1 VLMs; NVFP4 stays hidden until measured).
The export format is fixed (HF unified checkpoint) and not a parameter. Defaults: node-params-v1.md §6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vlm_pipeline.common.params import Params, ui

QuantFormat = Literal["fp8"]
CalibSize = Literal[128, 512]


class QuantizeParams(Params):
    format: QuantFormat = Field("fp8", json_schema_extra=ui(order=0, summary=True, option_keys="quantize_preset"))
    # 128 by default (A/B 2026-10-13, Qwen3.5-0.8B FP8: calib 32/128/512 gave the same parity, top-1 0.928/0.935/0.932;
    # 512 costs ~7 more minutes and +1.7 GB peak). 512 = ModelOpt hf_ptq default, kept as the option.
    calib_size: CalibSize = Field(128, json_schema_extra=ui(order=1, widget="segmented", option_keys="calib_size"))
    # Worker v1 (quantize/scope.py) refuses these: shown disabled; STD-001 still blocks old workflows that set them.
    quantize_vision: bool = Field(
        False, json_schema_extra=ui(advanced=True, disabled_options={"true": "param_unsupported.worker_v1"})
    )
    kv_cache_quant: Literal["none", "fp8"] = Field(
        "none",
        json_schema_extra=ui(
            advanced=True, option_keys="kv_cache_quant", disabled_options={"fp8": "param_unsupported.worker_v1"}
        ),
    )
