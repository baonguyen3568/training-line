"""Cross-node rules of the first-party nodes (run by ``graph.validate`` through ``Registry.rules``)."""

from __future__ import annotations

from collections.abc import Iterator

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.common.heldout_data import is_heldout_folder
from vlm_pipeline.nodes.data import DATA_SPLIT, PASS_THROUGH
from vlm_pipeline.nodes.model_ops import EXPORT_HF_DIR, QUANTIZE_MODELOPT_PTQ
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


# Nodes whose output is data of their input's origin (a split / filtered / deduped local folder is still local).
# One source with the standards (nodes.data.PASS_THROUGH): a new curation node is seen by both.
_DATA_PASS_THROUGH = frozenset({DATA_SPLIT.type_id, *PASS_THROUGH})
_SOURCE_KINDS = {"data.local_folder": "dataset:local", "data.hf_load": "dataset:hub"}


def input_kinds(view: GraphView, node_id: str) -> dict[str, str]:
    """Wired input ports -> ``dataset:hub`` | ``dataset:local`` | ``model`` | ``other`` (SupportQuery.inputs).

    Data is traced through pass-through nodes (split and the curation nodes) to its source: a split of a
    local folder is local; a mix is local when every input is.
    """
    kinds: dict[str, str] = {}
    for port, pspec in view.specs[node_id].inputs.items():
        src = view.source(node_id, port)
        if src is None:
            continue
        if is_compatible(pspec.type, "Model"):
            kinds[port] = "model"
            continue
        kinds[port] = _data_kind(view, src, set())
    return kinds


def _data_kind(view: GraphView, src: str | None, seen: set[str]) -> str:
    """Origin kind of the data ``src`` outputs, through pass-through nodes; a mix is local only if all inputs are."""
    while src is not None and src not in seen and view.specs[src].type_id in _DATA_PASS_THROUGH:
        seen.add(src)
        src = view.source(src, next(iter(view.specs[src].inputs)))
    if src is None or src in seen:
        return "other"
    if view.specs[src].type_id == "data.mix":
        seen.add(src)
        parts = {_data_kind(view, view.source(src, p), seen) for p in view.specs[src].inputs if view.source(src, p)}
        return "dataset:local" if parts == {"dataset:local"} else "other"
    return _SOURCE_KINDS.get(view.specs[src].type_id, "other")


def model_input(view: GraphView, node_id: str) -> str | None:
    """The node's model input port (first input that accepts a Model), if any."""
    return _model_input(view, node_id)


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


# Inputs whose data shapes the model (trained or calibrated on); standards treat hub eval_only data the same way.
_SHAPING_PORTS = {
    TRAIN_LORA_VLM.type_id: ("train_data", "val_data"),
    TRAIN_SFT_VLM.type_id: ("train_data", "val_data"),
    QUANTIZE_MODELOPT_PTQ.type_id: ("calib_data",),
}
_REFERENCE_ONLY_PORTS = frozenset({"heldout"})  # data.dedupe compares against it; it never reaches the output


def _data_origins(view: GraphView, node_id: str, port: str) -> set[str]:
    """Every node upstream of ``port`` whose data can reach it: all Dataset inputs, except reference-only ones."""
    found: set[str] = set()
    stack = list(view.sources.get((node_id, port), ()))
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        for name, pspec in view.specs[current].inputs.items():
            if name not in _REFERENCE_ONLY_PORTS and is_compatible(pspec.type, "Dataset"):
                stack.extend(view.sources.get((current, name), ()))
    return found


def heldout_data_never_trains(view: GraphView) -> Iterator[Issue]:
    """DATA-007: benchmark test data (a local folder under data/eval/) reaches train/val data or quantize calibration.

    Traced through every data step (split, filter, dedupe, mix, caption); symlinks are followed when the workspace
    is known (validate in the app always passes it; a local path is never absolute). Reported once per folder node.
    """
    reported: set[str] = set()
    for node_id, spec in view.specs.items():
        ports = _SHAPING_PORTS.get(spec.type_id, ())
        origins = set().union(set(), *(_data_origins(view, node_id, port) for port in ports))
        for origin in sorted(origins - reported):
            if view.specs[origin].type_id != "data.local_folder":
                continue
            path = getattr(view.params.get(origin), "path", None)
            if isinstance(path, str) and is_heldout_folder(view.workspace, path):
                reported.add(origin)
                yield Issue("DATA-007", node_id=origin, param="path", params={"node": origin, "target_node": node_id})


RULES = (train_node_matches_model, export_needs_portable_model, heldout_data_never_trains)
__all__ = ["RULES", "input_kinds", "model_input", "origin_model_id"]
