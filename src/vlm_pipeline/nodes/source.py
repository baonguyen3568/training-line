"""Source nodes: pick a base model, load a dataset (category ``source``, host-side, P009 §2.3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, create_model

from vlm_pipeline.common.catalog import Catalog, load_catalog
from vlm_pipeline.common.params import Params, ui
from vlm_pipeline.data.config import HubDatasetParams, LocalFolderParams
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec


def _model_pick_params(catalog: Catalog) -> type[Params]:
    """Params of ``model.hf_pick``: one choice among the vetted catalog models (configs/catalog/models.yaml).

    Option labels are data, not i18n: ``display_name`` + tier/RAM/licence/gated badges travel in ``x-ui.option_meta``;
    the descriptive tagline is the i18n key ``model_catalog.<id>.tagline``.
    """
    entries = catalog.selectable()
    option_meta = {
        m.id: {
            "label": m.display_name,
            "tier": m.tier,
            "ram_gb": m.ram_gb_train,
            "ram_source": m.ram_source,
            "gated": m.gated,
            "license": m.license,
            "vendor": m.vendor,
            # The model always runs one fixed task (Nemotron-Parse): "Chạy thử" hides its question box (B-7).
            "task_prompt_fixed": m.task_prompt_fixed,
        }
        for m in entries
    }
    model_id: Any = Literal.__getitem__(tuple(m.id for m in entries))
    field = Field(catalog.default.id, json_schema_extra=ui(order=0, summary=True, option_meta=option_meta))
    params: type[Params] = create_model("ModelPickParams", __base__=Params, model=(model_id, field))
    return params


ModelPickParams = _model_pick_params(load_catalog())


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
