"""Cross-node rules of the first-party nodes (run by ``graph.validate`` through ``Registry.rules``)."""

from __future__ import annotations

from collections.abc import Iterator

from vlm_pipeline.common.catalog import load_dataset_catalog, model_entry
from vlm_pipeline.common.heldout_data import is_heldout_folder
from vlm_pipeline.common.local_cache import dataset_missing, model_missing
from vlm_pipeline.data.local_folder import preview_folder
from vlm_pipeline.nodes.data import DATA_SPLIT, PASS_THROUGH
from vlm_pipeline.nodes.model_ops import EXPORT_HF_DIR, QUANTIZE_MODELOPT_PTQ
from vlm_pipeline.nodes.source import MODEL_HF_PICK, picked_model_id
from vlm_pipeline.nodes.spec import GraphView, Issue, is_compatible, model_kind
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
            return picked_model_id(view.params.get(current)) or None
        upstream = _model_input(view, current)
        current = view.source(current, upstream) if upstream else None
    return None


# Nodes whose output is data of their input's origin (a split / filtered / deduped local folder is still local).
# One source with the standards (nodes.data.PASS_THROUGH): a new curation node is seen by both.
_DATA_PASS_THROUGH = frozenset({DATA_SPLIT.type_id, *PASS_THROUGH})
_SOURCE_KINDS = {"data.local_folder": "dataset:local", "data.hf_load": "dataset:hub"}


def input_kinds(view: GraphView, node_id: str) -> dict[str, str]:
    """Wired input ports -> ``dataset:hub`` | ``dataset:local`` | ``model:<type>`` | ``other`` (SupportQuery.inputs).

    Data is traced through pass-through nodes (split and the curation nodes) to its source: a split of a
    local folder is local; a mix is local when every input is.
    """
    kinds: dict[str, str] = {}
    for port, pspec in view.specs[node_id].inputs.items():
        src = view.source(node_id, port)
        if src is None:
            continue
        if is_compatible(pspec.type, "Model"):
            kinds[port] = _model_kind_of(view, src)
            continue
        kinds[port] = _data_kind(view, src, set())
    return kinds


def _model_kind_of(view: GraphView, src: str) -> str:
    """``model:<type>`` of the model ``src`` outputs (its one Model-typed output port), ``model`` if ambiguous."""
    types = {p.type for p in view.specs[src].outputs.values() if is_compatible(p.type, "Model")}
    return model_kind(types.pop()) if len(types) == 1 else "model"


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
    for node_id, spec in view.specs.items():
        if spec.type_id not in _TRAIN_TYPES:
            continue
        entry = model_entry(origin_model_id(view, node_id, "base_model"))
        if entry is None or entry.supports_train_node(spec.type_id):
            continue
        params = {"node": node_id, "model": entry.display_name, "supported": entry.train_node}
        yield Issue("CFG-010", node_id=node_id, port="base_model", params=params)


def export_needs_portable_model(view: GraphView) -> Iterator[Issue]:
    """EXP-002: Export of a model that runs its own code (catalog remote_code) is not portable yet (ADR-012 B3).

    The export worker re-checks the folder's real content (``export.portable``); this rule reports it on the canvas.
    """
    for node_id, spec in view.specs.items():
        if spec.type_id != EXPORT_HF_DIR.type_id:
            continue
        entry = model_entry(origin_model_id(view, node_id, "model"))
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


def local_folder_preview(view: GraphView) -> Iterator[Issue]:
    """DATA-006 (warn) on the canvas: a local folder's labels / image headers the run's scan would refuse (DATA-V).

    Header-only, at most ``PREVIEW_BUDGET_S`` per folder, cached until the folder changes; a folder that takes
    longer gets no verdict here (the run decides). Needs the workspace (validate in the app always passes it).
    """
    if view.workspace is None:
        return
    root = view.workspace.resolve()
    for node_id, spec in view.specs.items():
        if spec.type_id != "data.local_folder":
            continue
        params = view.params.get(node_id)
        path = getattr(params, "path", None)
        if not isinstance(path, str):
            continue
        folder = (root / path).resolve()
        if folder != root and root not in folder.parents:  # SEC-001 already says it
            continue
        side = int(getattr(params, "max_image_side_px", 4096))
        scan = preview_folder(folder, max_image_side_px=side) if folder.is_dir() else None
        if folder.is_dir() and scan is None:
            continue  # over the time budget: the run's full scan decides
        if not folder.is_dir():
            problem = {"line": 0, "reason": "labels_missing"}
            count = 1
        elif scan is not None and scan.problem_count:
            problem, count = scan.problems[0], scan.problem_count
        else:
            continue
        params_out = {"node": node_id, "problem_count": count, "first_line": problem["line"],
                      "first_reason": problem["reason"]}  # fmt: skip
        yield Issue("DATA-006", severity="warn", node_id=node_id, param="path", params=params_out)


def catalog_assets_on_this_machine(view: GraphView) -> Iterator[Issue]:
    """NET-002 (warn, P018 §3.1): a picked model / hub dataset the run needs is not downloaded (or not prepared) yet.

    Same check and params as the run's NotCachedError (``common.local_cache``), so the UI can offer "download all
    missing" before Run. A warning, not an error: the run itself says it again if nothing was downloaded.
    """
    if view.workspace is None:
        return
    root = view.workspace
    for node_id, spec in view.specs.items():
        params = view.params.get(node_id)
        missing: dict[str, object] | None = None
        if spec.type_id == MODEL_HF_PICK.type_id:
            entry = model_entry(picked_model_id(params))
            user_id = str(getattr(params, "user_model", "") or "")
            if user_id and entry is None:  # opened on a machine where that model was never added (PM: offer to add it)
                yield Issue("MODEL-006", node_id=node_id, param="user_model", params={
                    "node": node_id, "repo": getattr(params, "user_repo", ""),
                    "revision": getattr(params, "user_revision", "")})  # fmt: skip
                continue
            missing = model_missing(root, entry) if entry is not None else None
            param = "model"
        elif spec.type_id == "data.hf_load":
            repo = getattr(params, "repo", None)
            dataset = next((d for d in load_dataset_catalog().datasets if d.repo == repo), None)
            missing = dataset_missing(root, dataset) if dataset is not None else None
            param = "repo"
        if missing is not None:
            yield Issue("NET-002", severity="warn", node_id=node_id, param=param, params=missing)


RULES = (
    train_node_matches_model,
    export_needs_portable_model,
    heldout_data_never_trains,
    local_folder_preview,
    catalog_assets_on_this_machine,
)
__all__ = ["RULES", "input_kinds", "model_input", "origin_model_id"]
