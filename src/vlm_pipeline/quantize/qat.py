"""Quantization-aware training with ModelOpt (node ``quantize.qat``, ADR-022 #10, P023 §3). Host-safe at import.

Recipe (ModelOpt 0.44 ``examples/llm_qat``, probed in vlm-automodel:26.08-7): calibrate like PTQ (same scope: vision,
projector and lm_head stay BF16) -> a short fine-tune of the fake-quantized model at a low LR (LoRA by default) ->
export the same unified HF checkpoint as PTQ, so Eval / Benchmark / Export after it work unchanged.

The worker is not built yet: ``check_support`` refuses every QAT node (``worker_pending``) until it is, so the canvas
never offers a run the container would refuse (support invariant).
"""

from __future__ import annotations

from collections.abc import Mapping

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.quantize.config import QatParams
from vlm_pipeline.quantize.scope import QuantizeNotSupportedError

NODE_TYPE = "quantize.qat"


def _refuse(node_id: str, reason: str, model: CatalogEntry | None = None) -> None:
    params = {"node": node_id, "node_type": NODE_TYPE, "reason": reason}
    if model is not None:
        params["model"] = model.display_name
    raise QuantizeNotSupportedError(f"qat not supported: {reason}", params=params)


def check_support(node_id: str, params: QatParams, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Validate-time twin of what the QAT worker refuses: raise ``QuantizeNotSupportedError`` (CFG-011)."""
    del inputs  # train_data / calib_data kinds: decided with the worker
    if params.format != "fp8":
        _refuse(node_id, "format", model)
    if model is not None:
        if model.auto_class != "AutoModelForImageTextToText":
            _refuse(node_id, "remote_code_model", model)
        if model.quant_fp8 != "official":
            _refuse(node_id, "fp8_not_official", model)
    _refuse(node_id, "worker_pending", model)


__all__ = ["NODE_TYPE", "check_support"]
