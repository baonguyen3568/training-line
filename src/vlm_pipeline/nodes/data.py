"""Data preparation nodes (category ``data``)."""

from __future__ import annotations

from vlm_pipeline.data.config import SplitParams
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec

DATA_SPLIT = NodeSpec(
    type_id="data.split",
    version="1.0.0",
    category="data",
    icon="split",
    inputs={"dataset": PortSpec(type="Dataset")},
    outputs={"train": PortSpec(type="Dataset"), "val": PortSpec(type="Dataset")},
    params_model=SplitParams,
    runner="host",
)
