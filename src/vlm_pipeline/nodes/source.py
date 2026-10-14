"""Source nodes: pick a base model, load a dataset (category ``source``, host-side, P009 §2.3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, create_model, model_validator

from vlm_pipeline.common.catalog import Catalog, CatalogEntry, load_catalog
from vlm_pipeline.common.obligations import ObligationsTable, load_obligations, model_obligations
from vlm_pipeline.common.params import Params, ui
from vlm_pipeline.data.config import HubDatasetParams, LocalFolderParams
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec


def _option_meta(m: CatalogEntry, catalog: Catalog, obligations: ObligationsTable) -> dict[str, Any]:
    license_class, duties = model_obligations(m, obligations)
    return {
        "label": m.display_name,
        "tier": m.tier,
        "ram_gb": m.ram_gb_train,
        "ram_source": m.ram_source,
        "gated": m.gated,
        "license": m.license,
        "vendor": m.vendor,
        "vendor_name": catalog.vendor_name(m),  # proper noun for the download/terms dialogs (G-5), not i18n
        # G-5 consent dialog: licence duties from bao-security's table (configs/licenses/obligations.yaml)
        "license_class": license_class,
        "obligations": duties,
        # The model always runs one fixed task (Nemotron-Parse): "Chạy thử" hides its question box (B-7).
        "task_prompt_fixed": m.task_prompt_fixed,
    }


# Path-safe (no "..", no "/", lowercase): the id names a file of the user asset store (cleancode review).
USER_MODEL_PATTERN = r"^(user--[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*)?$"
USER_REPO_PATTERN = r"^([A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*)?$"


@model_validator(mode="after")
def _user_model_matches(self: Any) -> Any:
    from vlm_pipeline.download.resolver import user_model_id

    ids = (self.user_model, self.user_repo, self.user_revision)
    if not any(ids):
        return self
    if not all(ids) or user_model_id(self.user_repo, self.user_revision) != self.user_model:
        raise ValueError("user_model must be the id of user_repo at user_revision")
    return self


def _model_pick_params(catalog: Catalog) -> type[Params]:
    """Params of ``model.hf_pick``: one choice among the vetted catalog models (configs/catalog/models.yaml).

    Option labels are data, not i18n: ``display_name`` + tier/RAM/licence/gated badges travel in ``x-ui.option_meta``;
    the descriptive tagline is the i18n key ``model_catalog.<id>.tagline``.
    """
    entries = catalog.selectable()
    obligations = load_obligations()
    option_meta = {m.id: _option_meta(m, catalog, obligations) for m in entries}
    model_id: Any = Literal.__getitem__(tuple(m.id for m in entries))
    field = Field(catalog.default.id, json_schema_extra=ui(order=0, summary=True, option_meta=option_meta))
    # ADR-021: a model the user added (list: GET /api/user-assets/models); when set it wins over ``model``. The UI
    # also writes its repo + 40-hex revision, so a workflow opened on another machine can offer "add this model"
    # prefilled with exactly the author's revision (PM 2026-10-14); they must be the ones the id was made from.
    user = Field("", max_length=206, pattern=USER_MODEL_PATTERN, json_schema_extra=ui(order=1, widget="user_model"))
    repo = Field("", max_length=193, pattern=USER_REPO_PATTERN, json_schema_extra=ui(widget="hidden"))
    revision = Field("", pattern=r"^([0-9a-f]{40})?$", json_schema_extra=ui(widget="hidden"))
    params: type[Params] = create_model(
        "ModelPickParams", __base__=Params,
        # create_model: validators passed by name (the decorator proxy is not a plain Callable for mypy)
        __validators__={"_user_model_matches": _user_model_matches},  # type: ignore[dict-item]
        model=(model_id, field), user_model=(str, user), user_repo=(str, repo), user_revision=(str, revision),
    )  # fmt: skip
    return params


def picked_model_id(params: object) -> str:
    """The model a ``model.hf_pick`` picks: the user-added one when set, else the catalog choice."""
    return str(getattr(params, "user_model", "") or getattr(params, "model", "") or "")


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

# ADR-021 H5: a Hugging Face dataset the user added (converted to images + labels.jsonl by the app; config, split and
# column mapping live in its record, not here, so the id is stable for the cache). It wins over ``repo`` when set.
USER_DATASET_PATTERN = r"^(user--ds--[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*)?$"


class HubLoadParams(HubDatasetParams):
    user_dataset: str = Field(
        "", max_length=300, pattern=USER_DATASET_PATTERN, json_schema_extra=ui(order=1, widget="user_dataset")
    )
    user_repo: str = Field("", max_length=193, pattern=USER_REPO_PATTERN, json_schema_extra=ui(widget="hidden"))
    user_revision: str = Field("", pattern=r"^([0-9a-f]{40})?$", json_schema_extra=ui(widget="hidden"))

    @model_validator(mode="after")
    def _user_dataset_matches(self) -> HubLoadParams:
        """The id must come from the hidden repo + revision (another machine offers "add this dataset" from them)."""
        from vlm_pipeline.download.resolver import user_dataset_id_prefix

        ids = (self.user_dataset, self.user_repo, self.user_revision)
        if not any(ids):
            return self
        if not all(ids) or not self.user_dataset.startswith(user_dataset_id_prefix(self.user_repo, self.user_revision)):
            raise ValueError("user_dataset must be an id of user_repo at user_revision")
        return self


def is_user_dataset(params: object) -> bool:
    """Whether a ``data.hf_load`` reads a user-added dataset (local after conversion), not the catalog repo."""
    return bool(getattr(params, "user_dataset", ""))


DATA_HF_LOAD = NodeSpec(
    type_id="data.hf_load",
    version="1.0.0",
    category="source",
    icon="dataset",
    outputs={"dataset": PortSpec(type="Dataset")},
    params_model=HubLoadParams,
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
