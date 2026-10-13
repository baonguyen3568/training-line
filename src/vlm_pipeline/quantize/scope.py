"""Scope and sizing of the quantize node (B-4), pure policy used by the app adapter (app/quantize_worker.py).

v1 (refused with a clear reason otherwise): format fp8, native models (``auto_class: AutoModelForImageTextToText``,
catalog ``quant_fp8`` official), hub datasets with an image column, vision tower and KV cache not quantized.
Layer rule (ADR-004 A3): imports ``common`` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.quantize.config import QuantizeParams

PTQ_EXIT_PARITY_FAILED = 3  # vlm_pipeline.quantize.ptq: quantized, but the BF16-parity sanity check failed


class QuantizeNotSupportedError(VlmPipelineError):
    """The model / data / options are outside what the quantize worker supports yet (refused before the GPU)."""

    code: ClassVar[str] = "CFG-011"


def est_peak_gb(entry: CatalogEntry, calib_size: int = 128) -> float:
    """BF16 weights twice (model + calibration activations / export copy) + runtime and cold JIT compile overhead.

    Measured on Qwen3.5-0.8B (1.77 GB snapshot), run.json peaks with cold Triton + nvcc caches (a read-only job
    compiles in its own run dir every time): calib 32 16.8 GB, 128 19.3 GB, 512 21.0 GB (A/B 2026-10-13).
    Overhead fitted on that one model: 16 GB up to calib 128, 18 GB above (catalog disk_gb 2.0 -> 20.0 / 22.0 GB).
    """
    return round(entry.disk_gb * 2 + (16 if calib_size <= 128 else 18), 1)


def check_supported(entry: CatalogEntry, params: QuantizeParams, node_id: str) -> None:
    def refuse(reason: str) -> None:
        raise QuantizeNotSupportedError(
            f"quantize not supported: {reason}",
            params={
                "node": node_id,
                "node_type": "quantize.modelopt_ptq",
                "model": entry.display_name,
                "reason": reason,
            },
        )

    if params.format != "fp8":
        refuse("format")
    if entry.auto_class != "AutoModelForImageTextToText":
        refuse("remote_code_model")  # Nemotron Parse: P-5 spike not done
    if entry.quant_fp8 != "official":
        refuse("fp8_not_official")
    if params.quantize_vision:
        refuse("vision_quant")
    if params.kv_cache_quant != "none":
        refuse("kv_cache_quant")


def ptq_argv(*, model: str, data: str, out: str, fmt: str, calib_size: int) -> tuple[str, ...]:
    return (
        "python", "-m", "vlm_pipeline.quantize.ptq", "--model", model, "--data", data, "--out", out,
        "--format", fmt, "--calib-size", str(calib_size),
    )  # fmt: skip


def check_support(node_id: str, params: QuantizeParams, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Validate-time twin of what the worker refuses (one source of truth): raise ``QuantizeNotSupportedError``.

    ``model`` is the catalog model upstream (None when the graph cannot tell yet: nothing to check here);
    ``inputs`` maps wired ports to their kind, e.g. ``{"calib_data": "dataset:hub"}`` (unwired ports are absent).
    """
    if model is not None:
        check_supported(model, params, node_id)
    calib = inputs.get("calib_data")
    if calib is None:
        _refuse(node_id, "calib_data_missing")
    elif calib != "dataset:hub":
        _refuse(node_id, "local_calib")


def _refuse(node_id: str, reason: str) -> None:
    raise QuantizeNotSupportedError(
        f"quantize not supported: {reason}",
        params={"node": node_id, "node_type": "quantize.modelopt_ptq", "reason": reason},
    )
