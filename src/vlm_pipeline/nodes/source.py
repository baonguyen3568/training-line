"""Source nodes: pick a base model, load a dataset (category ``source``, host-side, P009 §2.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vlm_pipeline.common.params import Params, ui
from vlm_pipeline.data.config import HubDatasetParams, LocalFolderParams
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec

# Models vetted for v1 (ADR-001 catalog, licenses checked by security). TODO(M1, bao-research): read from the
# model catalog once configs/catalog/models.yaml exists, so RAM/licence labels and per-model defaults come with it.
BaseModelRepo = Literal[
    "nvidia/NVIDIA-Nemotron-Parse-v1.1",
    "nvidia/Cosmos-Reason2-2B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
]


class ModelPickParams(Params):
    repo: BaseModelRepo = Field("nvidia/NVIDIA-Nemotron-Parse-v1.1", json_schema_extra=ui(order=0, summary=True))


MODEL_HF_PICK = NodeSpec(
    type_id="model.hf_pick",
    version="1.0.0",
    category="source",
    icon="model",
    outputs={"model": PortSpec(type="Model")},
    params_model=ModelPickParams,
    runner="host",
)

DATA_HF_LOAD = NodeSpec(
    type_id="data.hf_load",
    version="1.0.0",
    category="source",
    icon="dataset",
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=HubDatasetParams,
    runner="host",
)

DATA_LOCAL_FOLDER = NodeSpec(
    type_id="data.local_folder",
    version="1.0.0",
    category="source",
    icon="folder",
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=LocalFolderParams,
    # A folder's content can change without its path changing; cached only once a content manifest hash exists.
    cacheable=False,
    runner="host",
)
