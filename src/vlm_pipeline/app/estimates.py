"""Time estimate of a workflow before it runs (PM 2026-10-13): train nodes from measured runs, nothing invented.

For each train node: the base model (upstream ``model.hf_pick``), the training data size (upstream ``data.hf_load``
catalog ``num_train`` or its quick-run limit, through ``data.split`` which keeps the hub train split whole) and the
node's own epochs / batch shape -> steps; the model's training template gives the checkpoint/validation cadence;
``train.estimate`` turns them into minutes from measured runs (same model, image, task, batch, dataset).
A train node whose size or speed is unknown makes the whole duration unknown (``None``): a partial sum would read as
a full answer. Other heavy nodes (eval, quantize, bench) have no measurement yet and are listed in ``unmeasured``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlm_pipeline.common.catalog import CatalogEntry, load_catalog, load_dataset_catalog
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.train.estimate import estimate_train, measured_timings, train_steps

TRAIN_TASKS = {"train.sft_vlm": "sft", "train.lora_vlm": "lora"}
HEAVY_NODES = frozenset({"eval.benchmarks", "quantize.modelopt_ptq", "bench.perf", "serve.vllm", "train.merge_lora"})


def _source(workflow: Mapping[str, Any], node_id: str, port: str) -> dict[str, Any] | None:
    nodes = {n.get("id"): n for n in workflow.get("nodes", []) if isinstance(n, dict)}
    for edge in workflow.get("edges", []):
        try:
            (src, _out), (dst, inp) = edge["from"], edge["to"]
        except (KeyError, TypeError, ValueError):
            continue
        if dst == node_id and inp == port:
            return nodes.get(src)
    return None


def _train_data(workflow: Mapping[str, Any], train_id: str) -> tuple[int, str] | None:
    """(samples, ``repo@revision``) of the hub dataset a train node learns from; None for local or unknown data."""
    node = _source(workflow, train_id, "train_data")
    if node is not None and node.get("type") == "data.split":
        node = _source(workflow, str(node.get("id")), "dataset")
    if node is None or node.get("type") != "data.hf_load":  # local folders: size known only once read
        return None
    params = _params(node)
    if params is None:
        return None
    entry = next((d for d in load_dataset_catalog().datasets if d.repo == getattr(params, "repo", None)), None)
    if entry is None or entry.num_train <= 0:
        return None
    limit = int(getattr(params, "max_samples", 0) or 0)
    return (min(limit, entry.num_train) if limit else entry.num_train), f"{entry.repo}@{entry.revision}"


def _params(node: Mapping[str, Any]) -> Any:
    """The node's params through its own model (untrusted workflow: junk -> None, validate reports it)."""
    spec = REGISTRY.get(str(node.get("type")))
    raw = node.get("params") or {}
    if spec is None or not isinstance(raw, dict):
        return None
    try:
        return spec.validate_params(dict(raw))
    except ValueError:
        return None


def _base_model(workflow: Mapping[str, Any], train_id: str) -> CatalogEntry | None:
    node = _source(workflow, train_id, "base_model")
    if node is None or node.get("type") != "model.hf_pick":
        return None
    params = _params(node)
    return load_catalog().get(str(getattr(params, "model", ""))) if params is not None else None


def _cadence(root: Path, entry: CatalogEntry) -> int | None:
    """Checkpoint (= validation) cadence the recipe will use: the model's training template ``ckpt_every_steps``."""
    from vlm_pipeline.train.recipe import RecipeError, load_template, template_path

    try:
        sched = load_template(template_path(root, entry)).get("step_scheduler") or {}
        return int(sched["ckpt_every_steps"])
    except (RecipeError, OSError, KeyError, TypeError, ValueError):
        return None


def train_node_estimate(root: Path, workflow: Mapping[str, Any], node: Mapping[str, Any], image: str) -> dict[str, Any]:
    """``{steps, minutes: {lo, hi} | None, based_on}`` for one train node (steps None when the data size is unknown)."""
    node_id, type_id = str(node.get("id")), str(node.get("type"))
    data, entry, params = _train_data(workflow, node_id), _base_model(workflow, node_id), _params(node)
    every = _cadence(root, entry) if entry is not None else None
    if data is None or entry is None or params is None or every is None:
        return {"steps": None, "minutes": None, "based_on": []}
    samples, dataset = data
    gbs, local, epochs = (int(getattr(params, k)) for k in ("global_batch_size", "local_batch_size", "epochs"))
    steps = train_steps(samples, gbs, epochs)
    timings = measured_timings(
        root, base_model=entry.repo, image=image, task=TRAIN_TASKS[type_id], batch=(gbs, local), dataset=dataset
    )
    est = estimate_train(steps, every, timings)
    if est is None:
        return {"steps": steps, "minutes": None, "based_on": []}
    return {"steps": steps, "minutes": {"lo": est.minutes_lo, "hi": est.minutes_hi}, "based_on": list(est.based_on)}


def workflow_duration(root: Path, workflow: Mapping[str, Any], image: str) -> dict[str, Any]:
    """``duration_min`` ({lo, hi} or None), ``per_node`` train estimates and the heavy nodes without a measurement."""
    per_node: dict[str, Any] = {}
    unmeasured: list[str] = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("type") in TRAIN_TASKS:
            per_node[str(node.get("id"))] = train_node_estimate(root, workflow, node, image)
        elif node.get("type") in HEAVY_NODES:
            unmeasured.append(str(node.get("id")))
    minutes = [e["minutes"] for e in per_node.values()]
    total = (
        {"lo": round(sum(m["lo"] for m in minutes), 1), "hi": round(sum(m["hi"] for m in minutes), 1)}
        if minutes and all(m is not None for m in minutes)
        else None
    )
    return {"duration_min": total, "per_node": per_node, "unmeasured": unmeasured}


__all__ = ["train_node_estimate", "workflow_duration"]
