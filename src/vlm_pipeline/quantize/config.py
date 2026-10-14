"""Parameters of the quantization engine (ModelOpt PTQ, node ``quantize.modelopt_ptq``). Host-safe.

v1 runs FP8 only (official ModelOpt support for the v1 VLMs). NVFP4 is offered disabled until research measures it on
GB10 (sm_121) and a score gate exists; INT4-AWQ is out of 1.0.
The export format is fixed (HF unified checkpoint) and not a parameter. Defaults: node-params-v1.md §6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vlm_pipeline.common.params import Params, ui

QuantFormat = Literal["fp8", "nvfp4"]
CalibSize = Literal[128, 512]


# The format choice of PTQ and QAT (one definition): nvfp4 is offered disabled until research shows GB10 can SERVE it.
_FORMAT_UI = ui(
    order=0, summary=True, option_keys="quantize_preset", disabled_options={"nvfp4": "param_unsupported.worker_v1"}
)


class QuantizeParams(Params):
    format: QuantFormat = Field("fp8", json_schema_extra=_FORMAT_UI)  # scope.check_supported refuses nvfp4
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


QatAdapter = Literal["lora", "full"]
QatLrPreset = Literal["low", "lower"]
QAT_LR: dict[QatLrPreset, float] = {"low": 1e-5, "lower": 5e-6}  # ModelOpt examples/llm_qat: a short run at a low LR


class QatParams(Params):
    """quantize.qat (ADR-022 #10, P023 §3): calibrate like PTQ, then a short fine-tune with fake quantization.

    Worker: vlm_pipeline.quantize.qat (bao-fixcode, ModelOpt 0.44 QATTrainer in the train image). ``full`` is refused by
    the support check above the full fine-tune size limit (as train.sft_vlm). QAD and INT4 formats are not in 1.0.
    """

    format: QuantFormat = Field("fp8", json_schema_extra=_FORMAT_UI)
    adapter: QatAdapter = Field("lora", json_schema_extra=ui(order=1, widget="segmented", option_keys="qat_adapter"))
    epochs: Literal[1, 2, 3] = Field(1, json_schema_extra=ui(order=2, widget="segmented", option_keys="epoch_count"))
    lr_preset: QatLrPreset = Field("low", json_schema_extra=ui(order=3, widget="segmented", option_keys="qat_lr"))
    max_steps: int = Field(0, ge=0, le=100_000, json_schema_extra=ui(advanced=True))  # 0 = by epochs (~10% of SFT)
    calib_size: CalibSize = Field(
        128, json_schema_extra=ui(advanced=True, widget="segmented", option_keys="calib_size")
    )

    @property
    def learning_rate(self) -> float:
        # QAT_LR is the full-weight rate; a LoRA adapter barely moves at it in a short run: x10 for LoRA (1e-4 / 5e-5),
        # to confirm in the Q2 smoke (fixcode 2026-10-14).
        return QAT_LR[self.lr_preset] * (10 if self.adapter == "lora" else 1)
