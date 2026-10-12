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
    calib_size: CalibSize = Field(512, json_schema_extra=ui(order=1, widget="segmented", option_keys="calib_size"))
    quantize_vision: bool = Field(False, json_schema_extra=ui(advanced=True))
    kv_cache_quant: Literal["none", "fp8"] = Field(
        "none", json_schema_extra=ui(advanced=True, option_keys="kv_cache_quant")
    )
