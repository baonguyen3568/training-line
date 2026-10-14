"""Nodes that transform or run a model: merge LoRA, quantize, trial inference, export."""

from __future__ import annotations

from pydantic import Field

from vlm_pipeline.common.params import LocalPath, Params, ui
from vlm_pipeline.nodes.spec import EmptyParams, NodeSpec, PortSpec
from vlm_pipeline.quantize.config import QatParams, QuantizeParams
from vlm_pipeline.serve.config import InferenceParams


class ExportParams(Params):
    """Export is always safetensors with the licence notice attached (ADR-003); neither is a parameter."""

    # TODO(M1): move to an export engine config when the export worker exists.
    dest: LocalPath = Field("outputs/exports/my-model", json_schema_extra=ui(order=0, summary=True))


TRAIN_MERGE_LORA = NodeSpec(
    type_id="train.merge_lora",
    version="1.0.0",
    category="train",
    icon="merge",
    inputs={"base_model": PortSpec(type="Model"), "adapter": PortSpec(type="Adapter")},
    outputs={"model": PortSpec(type="Checkpoint")},
    params_model=EmptyParams,
    runner="container",
)

QUANTIZE_MODELOPT_PTQ = NodeSpec(
    type_id="quantize.modelopt_ptq",
    version="1.0.0",
    category="quantize",
    icon="quantize",
    inputs={"model": PortSpec(type="Model"), "calib_data": PortSpec(type="Dataset", optional=True)},
    outputs={"quantized": PortSpec(type="QuantizedModel")},
    params_model=QuantizeParams,
    runner="container",
)

SERVE_VLLM = NodeSpec(
    type_id="serve.vllm",
    version="1.0.0",
    category="test",
    icon="play",
    inputs={"model": PortSpec(type="Model")},
    outputs={"endpoint": PortSpec(type="Endpoint")},
    params_model=InferenceParams,
    cacheable=False,  # a live endpoint, not a stored artifact
    runner="container",
)

EXPORT_HF_DIR = NodeSpec(
    type_id="export.hf_dir",
    version="1.0.0",
    category="export",
    icon="export",
    inputs={"model": PortSpec(type="Model")},
    outputs={"artifact": PortSpec(type="ExportArtifact")},
    params_model=ExportParams,
    cacheable=False,  # writes to a user-chosen folder
    runner="container",
)

# ADR-022 #10: same output as PTQ (QuantizedModel), so Eval / Benchmark / Export after it work unchanged.
QUANTIZE_QAT = NodeSpec(
    type_id="quantize.qat",
    version="1.0.0",
    category="quantize",
    icon="quantize",
    inputs={
        "model": PortSpec(type="Model"),
        "train_data": PortSpec(type="Dataset"),
        "calib_data": PortSpec(type="Dataset", optional=True),
    },
    outputs={"quantized": PortSpec(type="QuantizedModel")},
    params_model=QatParams,
    runner="container",
)
