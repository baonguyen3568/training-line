"""Graph executor (P009 §5): validate -> topological order -> per node: cache hit or run on a Worker.

v1 scope: nodes run one at a time in topological order (one GPU, GPU lock held by the job runner); a failing node
only skips its downstream nodes, independent branches keep running. Cache index = ``artifacts.index.ArtifactIndex``
(SQLite, shared with retention). Interrupted nodes resume from their checkpoint on the next run because a node's
run dir is derived from its cache key.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vlm_pipeline.artifacts.index import ArtifactIndex, CacheEntry, OutputRef
from vlm_pipeline.graph.cache import compute_cache_key
from vlm_pipeline.graph.errors import UserError, classify
from vlm_pipeline.graph.validation import Issue, validate
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, Worker, read_checkpoint_step
from vlm_pipeline.nodes.spec import NodeSpec, Registry

logger = logging.getLogger(__name__)

Event = dict[str, Any]
DONE = frozenset({"completed", "cached"})
# Output port types that hold trained weights: their cache entries are never auto-evicted (bao-security / P005).
TRAINED_PORT_TYPES = frozenset({"Checkpoint", "Adapter"})


def produces_trained_model(spec: NodeSpec) -> bool:
    return any(port.type in TRAINED_PORT_TYPES for port in spec.outputs.values())


@dataclass
class GraphRun:
    graph_run_id: str
    status: dict[str, str]  # pending|queued|running|completed|failed|interrupted|cached|skipped
    errors: dict[str, UserError] = field(default_factory=dict)
    warnings: dict[str, list[UserError]] = field(default_factory=dict)  # e.g. CACHE-001
    outputs: dict[str, dict[str, ArtifactRef]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)  # validation errors (run did not start)

    @property
    def state(self) -> str:
        """Overall state for the run list: invalid | failed | interrupted | completed."""
        if self.issues:
            return "invalid"
        values = set(self.status.values())
        if "failed" in values:
            return "failed"
        if "interrupted" in values or "skipped" in values or "pending" in values:
            return "interrupted"
        return "completed"


@dataclass(frozen=True)
class _Node:
    node_id: str
    spec: NodeSpec
    params: BaseModel


def _topo_order(node_ids: list[str], edges: list[tuple[str, str, str, str]]) -> list[str]:
    """Kahn's algorithm; ties keep the workflow file order (stable, predictable for users and tests)."""
    indeg = dict.fromkeys(node_ids, 0)
    children: dict[str, list[str]] = {n: [] for n in node_ids}
    for src, _, dst, _ in edges:
        indeg[dst] += 1
        children[src].append(dst)
    ready = deque(n for n in node_ids if indeg[n] == 0)
    order: list[str] = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return order


class Executor:
    def __init__(
        self,
        registry: Registry,
        worker: Worker,
        state_dir: Path,
        *,
        index: ArtifactIndex | None = None,
        on_event: Callable[[Event], None] | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.registry = registry
        self.worker = worker
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.index = index or ArtifactIndex(self.state_dir / "app.db")
        self.on_event = on_event
        self.workspace = workspace
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------- public

    def cancel(self, graph_run_id: str) -> bool:
        """Request cancellation of a running graph (thread-safe). False if no such run is active."""
        with self._lock:
            event = self._cancels.get(graph_run_id)
        if event is None:
            return False
        event.set()
        return True

    def run(self, workflow: Mapping[str, Any], *, graph_run_id: str | None = None) -> GraphRun:
        gid = graph_run_id or f"g-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        issues = [i for i in validate(workflow, self.registry, self.workspace) if i.severity == "error"]
        if issues:
            run = GraphRun(gid, {}, issues=issues)
            self._emit(gid, {"type": "run_status", "status": "invalid", "issues": [_issue_dict(i) for i in issues]})
            return run

        nodes, edges = self._parse(workflow)
        order = _topo_order(list(nodes), edges)
        run = GraphRun(gid, dict.fromkeys(order, "pending"))
        cancel = threading.Event()
        with self._lock:
            self._cancels[gid] = cancel
        self._emit(gid, {"type": "run_status", "status": "running", "nodes": order})
        keys: dict[str, str] = {}
        try:
            for node_id in order:
                self._step(run, nodes[node_id], edges, keys, cancel)
        finally:
            self.index.release_run(gid)  # artifacts used by this run become evictable again
            with self._lock:
                self._cancels.pop(gid, None)
        self._emit(gid, {"type": "run_status", "status": run.state})
        return run

    # ----------------------------------------------------------------------------------------- internals

    def _parse(self, workflow: Mapping[str, Any]) -> tuple[dict[str, _Node], list[tuple[str, str, str, str]]]:
        nodes: dict[str, _Node] = {}
        for raw in workflow["nodes"]:
            spec = self.registry.get(raw["type"])
            if spec is None:  # pragma: no cover - validate() already rejected it
                raise ValueError(raw["type"])
            nodes[raw["id"]] = _Node(raw["id"], spec, spec.validate_params(raw.get("params", {})))
        edges = [(e["from"][0], e["from"][1], e["to"][0], e["to"][1]) for e in workflow.get("edges", [])]
        return nodes, edges

    def _set(self, run: GraphRun, node_id: str, status: str, **extra: Any) -> None:
        run.status[node_id] = status
        self._emit(run.graph_run_id, {"type": "node_status", "node_id": node_id, "status": status, **extra})

    def _step(
        self,
        run: GraphRun,
        node: _Node,
        edges: list[tuple[str, str, str, str]],
        keys: dict[str, str],
        cancel: threading.Event,
    ) -> None:
        nid = node.node_id
        upstream = [(src, out, inp) for src, out, dst, inp in edges if dst == nid]
        if cancel.is_set() or any(run.status.get(src) not in DONE for src, _, _ in upstream):
            self._set(run, nid, "skipped")
            return
        inputs = {inp: run.outputs[src][out] for src, out, inp in upstream}
        key = compute_cache_key(node.spec, node.params, {p: r.hash for p, r in inputs.items()}, node.spec.image_digest)
        keys[nid] = key
        if node.spec.cacheable and self._try_cache(run, nid, key):
            return

        run_dir = self.state_dir / "runs" / f"{nid}-{key.removeprefix('ck1-')[:16]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        job = NodeJob(nid, node.spec, node.params, inputs, run_dir, cancel, read_checkpoint_step(run_dir))
        self._set(run, nid, "running", cache_key=key, resume_from_step=job.resume_from_step)
        try:
            result = self.worker.run(job)
            missing = set(node.spec.outputs) - set(result.outputs)
            if missing:
                raise RuntimeError(f"worker returned no output for ports {sorted(missing)}")
        except Exception as exc:  # noqa: BLE001 - every worker failure becomes a catalog error, never a crash
            outcome = classify(exc, cancelled=cancel.is_set(), checkpoint_step=read_checkpoint_step(run_dir))
            if outcome.error is not None:
                run.errors[nid] = outcome.error
            logger.info("node %s %s: %s", nid, outcome.status, outcome.error.code if outcome.error else "cancelled")
            self._set(run, nid, outcome.status, **({"error": _error_dict(outcome.error)} if outcome.error else {}))
            return
        run.outputs[nid] = dict(result.outputs)
        if node.spec.cacheable:
            self.index.record(
                key,
                run_id=run.graph_run_id,
                node_type=node.spec.type_id,
                outputs={p: OutputRef(r.type, r.path, r.hash) for p, r in result.outputs.items()},
                parents=[keys[src] for src, _, _ in upstream],
                trained_model=produces_trained_model(node.spec),
            )
            self.index.lease([key], run.graph_run_id)  # downstream nodes of this run still read it
        self._set(run, nid, "completed", cache_key=key)

    def _try_cache(self, run: GraphRun, nid: str, key: str) -> bool:
        # Lease FIRST: retention never begins evicting a leased key, and a key whose eviction already began is no
        # longer `completed`, so lookup() then reports a miss. Only after that are the files worth verifying.
        self.index.lease([key], run.graph_run_id)
        entry: CacheEntry | None = self.index.lookup(key)
        if entry is None:
            return False
        if not ArtifactIndex.verify(entry):
            warning = UserError("CACHE-001", {"node": nid, "saved_at": entry.completed_at, "est_minutes": None})
            run.warnings.setdefault(nid, []).append(warning)
            self.index.invalidate(key)
            self._emit(run.graph_run_id, {"type": "warning", "node_id": nid, "error": _error_dict(warning)})
            return False
        self.index.mark_hit(key)
        run.outputs[nid] = {p: ArtifactRef(o.type, o.path, o.hash) for p, o in entry.outputs.items()}
        cached_from = {"run_id": entry.run_id, "completed_at": entry.completed_at}
        self._set(run, nid, "cached", cache_key=key, cached_from=cached_from)
        return True

    def _emit(self, gid: str, event: Event) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"graph_run_id": gid, "ts": time.time(), **event})
        except Exception:
            logger.exception("on_event callback failed")  # a broken UI stream must never stop a run


def _issue_dict(issue: Issue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "severity": issue.severity,
        "node_id": issue.node_id,
        "port": issue.port,
        "param": issue.param,
        "params": dict(issue.params),
    }


def _error_dict(err: UserError) -> dict[str, Any]:
    return {
        "code": err.code,
        "params": dict(err.params),
        "resumable": err.resumable,
        "checkpoint_step": err.checkpoint_step,
        "details": err.details,
    }


__all__ = ["DONE", "Executor", "GraphRun"]
