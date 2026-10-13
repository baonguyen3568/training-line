"""Cross-node rules of the first-party nodes (run by ``graph.validate`` through ``Registry.rules``)."""

from __future__ import annotations

from collections.abc import Iterator

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.nodes.model_ops import EXPORT_HF_DIR
from vlm_pipeline.nodes.source import MODEL_HF_PICK
from vlm_pipeline.nodes.spec import GraphView, Issue, is_compatible
from vlm_pipeline.nodes.train import TRAIN_LORA_VLM, TRAIN_SFT_VLM

_TRAIN_TYPES = frozenset({TRAIN_LORA_VLM.type_id, TRAIN_SFT_VLM.type_id})


def _model_input(view: GraphView, node_id: str) -> str | None:
    spec = view.specs[node_id]
    return next((name for name, port in spec.inputs.items() if is_compatible(port.type, "Model")), None)


def origin_model_id(view: GraphView, node_id: str, port: str) -> str | None:
    """Catalog id of the model picked upstream of ``port``, following model wires through train/merge/quantize."""
    seen: set[str] = set()
    current = view.source(node_id, port)
    while current is not None and current not in seen:
        seen.add(current)
        if view.specs[current].type_id == MODEL_HF_PICK.type_id:
            picked = view.params.get(current)
            return getattr(picked, "model", None)
        upstream = _model_input(view, current)
        current = view.source(current, upstream) if upstream else None
    return None


def train_node_matches_model(view: GraphView) -> Iterator[Issue]:
    """CFG-010: a fine-tune node the catalog has not vetted for the picked model (e.g. LoRA on Nemotron-Parse)."""
    catalog = load_catalog()
    for node_id, spec in view.specs.items():
        if spec.type_id not in _TRAIN_TYPES:
            continue
        entry = catalog.get(origin_model_id(view, node_id, "base_model") or "")
        if entry is None or entry.supports_train_node(spec.type_id):
            continue
        params = {"node": node_id, "model": entry.display_name, "supported": entry.train_node}
        yield Issue("CFG-010", node_id=node_id, port="base_model", params=params)


def export_needs_portable_model(view: GraphView) -> Iterator[Issue]:
    """EXP-002: Export of a model that runs its own code (catalog remote_code) is not portable yet (ADR-012 B3).

    The export worker re-checks the folder's real content (``export.portable``); this rule reports it on the canvas.
    """
    catalog = load_catalog()
    for node_id, spec in view.specs.items():
        if spec.type_id != EXPORT_HF_DIR.type_id:
            continue
        entry = catalog.get(origin_model_id(view, node_id, "model") or "")
        if entry is not None and entry.remote_code != "none":
            yield Issue("EXP-002", node_id=node_id, port="model", params={"node": node_id, "model": entry.display_name})


RULES = (train_node_matches_model, export_needs_portable_model)
