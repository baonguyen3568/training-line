"""Data preparation nodes (category ``data``)."""

from __future__ import annotations

from vlm_pipeline.data.config import DedupeParams, FilterParams, FormatConvertParams, MixParams, SplitParams
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

# Curation nodes (Release 1.0, stream B): Dataset in, Dataset out, on the host. Workers: bao-backend / bao-fixcode.
DATA_FILTER = NodeSpec(
    type_id="data.filter",
    version="1.0.0",
    category="data",
    icon="filter",
    inputs={"dataset": PortSpec(type="Dataset")},
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=FilterParams,
    runner="host",
)

DATA_DEDUPE = NodeSpec(
    type_id="data.dedupe",
    version="1.0.0",
    category="data",
    icon="copy",
    # heldout: the val/test data whose images must never appear in training (leak check, P4-2 eval_leak)
    inputs={"dataset": PortSpec(type="Dataset"), "heldout": PortSpec(type="Dataset", optional=True, many=True)},
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=DedupeParams,
    runner="host",
)

DATA_FORMAT_CONVERT = NodeSpec(
    type_id="data.format_convert",
    version="1.0.0",
    category="data",
    icon="file-text",
    inputs={"dataset": PortSpec(type="Dataset")},
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=FormatConvertParams,
    runner="host",
)

DATA_MIX = NodeSpec(
    type_id="data.mix",
    version="1.0.0",
    category="data",
    icon="layers",
    inputs={
        "dataset_1": PortSpec(type="Dataset"),
        "dataset_2": PortSpec(type="Dataset"),
        "dataset_3": PortSpec(type="Dataset", optional=True),
        "dataset_4": PortSpec(type="Dataset", optional=True),
    },
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=MixParams,
    runner="host",
)

# Data nodes that keep one dataset's identity through their ``dataset`` input (standards follow them upstream).
PASS_THROUGH = frozenset({DATA_FILTER.type_id, DATA_DEDUPE.type_id, DATA_FORMAT_CONVERT.type_id})
