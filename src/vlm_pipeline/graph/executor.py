"""Graph executor (P009 §5): validate -> topological order -> per node: cache hit or run on a Worker.

v1 scope: nodes run one at a time in topological order (one GPU, GPU lock held by the job runner); a failing node
only skips its downstream nodes, independent branches keep running. Cache index = ``artifacts.index.ArtifactIndex``
(SQLite, shared with retention). Interrupted nodes resume from their checkpoint on the next run because a node's
run dir is derived from its cache key.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from vlm_pipeline.artifacts.index import ArtifactIndex, CacheEntry, OutputRef
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.run_claims import (
    GenerationError,
    drop_resume_lease,
    node_dir_used,
    node_generation,
    write_resume_lease,
)
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, Stage, read_manifest, write_manifest
from vlm_pipeline.graph.actions import issue_actions
from vlm_pipeline.graph.cache import compute_cache_key
from vlm_pipeline.graph.errors import UserError, classify
from vlm_pipeline.graph.groups import expand
from vlm_pipeline.graph.validation import Issue, validate
from vlm_pipeline.graph.worker import (
    CHECKPOINT_MARKER,
    MAX_ATTEMPTS,
    NOT_RESUMABLE_MARKER,
    ArtifactRef,
    NodeJob,
    Worker,
    attempt_dirs,
    checkpoint_dirs,
    checkpoint_tree_problem,
    read_checkpoint_step,
    worker_identity,
    worker_support_problem,
    worker_supports,
)
from vlm_pipeline.nodes.spec import NodeSpec, Registry

logger = logging.getLogger(__name__)

Event = dict[str, Any]
DONE = frozenset({"completed", "cached"})
# Output port types that hold trained weights: their cache entries are never auto-evicted (bao-security / P005).
TRAINED_PORT_TYPES = frozenset({"Checkpoint", "Adapter"})


_STAGE_BY_CATEGORY: dict[str, Stage] = {
    "source": "data",
    "data": "data",
    "train": "train",
    "test": "eval",
    "quantize": "quantize",
    "eval": "eval",
    "export": "export",
    "util": "data",
}


class NonHostOutputError(RuntimeError):
    """A worker returned an output path that is not an existing host path inside the node's run dir."""


def check_output_paths(outputs: Mapping[str, ArtifactRef], run_dir: Path) -> None:
    """Outputs must be existing, absolute HOST paths under ``run_dir``.

    A container path like ``/workspace/...`` would make the cache index point at nothing and retention unable to
    free it (bao-cleanup).
    """
    root = run_dir.resolve()
    for port, ref in outputs.items():
        path = Path(ref.path)
        if not path.is_absolute() or not path.exists():
            raise NonHostOutputError(f"output {port!r}: not an existing absolute host path")
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise NonHostOutputError(f"output {port!r}: outside the node run dir")


class AttemptsExhaustedError(VlmPipelineError):
    """Every attempt dir of a node is used (P017, PM: max 10): a setup problem, not bad luck."""

    code: ClassVar[str] = "SYS-005"


def _manifest_digest(run_dir: Path) -> str | None:
    try:
        return read_manifest(run_dir).container_digest
    except (OSError, ValueError):
        return None


def _write_node_manifest(
    run_dir: Path,
    *,
    node_id: str,
    spec: NodeSpec,
    gid: str,
    status: RunStatus,
    error_code: str | None,
    started: float,
) -> None:
    """Write ``run.json`` of a node run dir (dashboard + retention).

    A manifest written by the worker itself (container jobs via jobrun) keeps its own status/details; the executor
    only links it to the graph run.
    """
    try:
        manifest = read_manifest(run_dir)
        manifest = manifest.model_copy(
            update={"parent_run_id": manifest.parent_run_id or gid, "tags": sorted({*manifest.tags, f"graph:{gid}"})}
        )
    except (OSError, ValueError):
        manifest = RunManifest(
            run_id=run_dir.name,
            name=f"{node_id} ({spec.type_id})",
            stage=_STAGE_BY_CATEGORY.get(spec.category, "data"),
            task=spec.type_id,
            status=status,
            parent_run_id=gid,
            error_code=error_code,
            started_at=started,
            ended_at=time.time() if status.is_terminal else None,
            tags=[f"graph:{gid}", f"node:{node_id}"],
        )
    write_manifest(run_dir, manifest)


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


def gather_inputs(spec: NodeSpec, wired: list[tuple[str, str, str, ArtifactRef]]) -> dict[str, ArtifactRef]:
    """Input port -> artifact; a ``many`` port's wires become ``port``, ``port.1``, ``port.2``... (``port_inputs``).

    Wires of one many-port are ordered by their artifact's content hash (node ids only break ties), so neither the
    wiring order nor renaming a node changes the cache key; a plain dict by port kept only the last wire (data.dedupe
    lost all but one heldout set).
    """
    inputs: dict[str, ArtifactRef] = {}
    for _src, _out, inp, ref in sorted(wired, key=lambda w: (w[2], w[3].hash, w[0], w[1])):
        port = spec.inputs.get(inp)
        if port is not None and port.many and inp in inputs:
            n = sum(1 for k in inputs if k.startswith(inp + "."))
            inputs[f"{inp}.{n + 1}"] = ref
        else:
            inputs[inp] = ref
    return inputs


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
        self._sinks: dict[str, Callable[[Event], None]] = {}  # per-run event sink (one shared Executor per app)
        # One computation per cache key at a time: concurrent graph runs sharing a node would otherwise run it twice
        # in the same run dir (and the same container name); the second one waits and then gets a cache hit.
        self._key_locks: dict[str, threading.Lock] = {}
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

    def run(
        self,
        workflow: Mapping[str, Any],
        *,
        graph_run_id: str | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> GraphRun:
        """Run a workflow (blocking; call it from a worker thread). ``on_event`` overrides the executor-wide sink."""
        gid = graph_run_id or f"g-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        with self._lock:
            if gid in self._cancels or gid in self._sinks:
                raise ValueError(f"graph run {gid!r} is already running")
            if on_event is not None:
                self._sinks[gid] = on_event
        try:
            return self._run(workflow, gid)
        finally:
            with self._lock:
                self._sinks.pop(gid, None)

    def _run(self, workflow: Mapping[str, Any], gid: str) -> GraphRun:
        can_run = functools.partial(worker_supports, self.worker)
        support = functools.partial(worker_support_problem, self.worker)
        issues = [
            i
            for i in validate(workflow, self.registry, self.workspace, can_run=can_run, support=support)
            if i.severity == "error"
        ]
        if issues:
            run = GraphRun(gid, {}, issues=issues)
            self._emit(gid, {"type": "run_status", "status": "invalid", "issues": [_issue_dict(i) for i in issues]})
            return run

        # P016: validate saw the file as written; the executor runs the flat graph (group instances inlined, ids
        # ``<group>__<inner>``). Over-limit groups gave a CFG-012 error above, so this never runs on an empty graph.
        nodes, edges = self._parse(expand(workflow, self.registry).workflow)
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
        key = extra.get("cache_key")
        if isinstance(key, str) and "-" in key:  # G-4: the "<nid>-<key16>" a "Reset this step" acts on
            extra.setdefault("node_base", f"{node_id}-{key.split('-', 1)[1][:16]}")
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
        inputs = gather_inputs(node.spec, [(src, out, inp, run.outputs[src][out]) for src, out, inp in upstream])
        key = compute_cache_key(
            node.spec,
            node.params,
            {p: r.hash for p, r in inputs.items()},
            node.spec.image_digest,
            worker=worker_identity(self.worker, node.spec),
        )
        keys[nid] = key
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            self._compute(run, node, key=key, inputs=inputs, upstream=upstream, keys=keys, cancel=cancel)

    def _used(self, run_dir: Path) -> bool:
        """A container job ran here or under it (run_claims), or a dir from before claims existed (it has job logs)."""
        if self.workspace is None:  # tests / demo without a project: nothing is ever claimed
            return False
        return node_dir_used(self.workspace, run_dir.name) or os.path.lexists(run_dir / "logs")

    def _attempt(self, node: _Node, base: Path) -> tuple[Path, Path | None, int | None]:
        """(run dir, checkpoint to resume from, its step) for this try of ``node`` (P017).

        A dir a container job already wrote is never handed out read-write again: the first unused attempt dir is
        taken. A resume reads the newest complete checkpoint of the newest earlier attempt that saved one (an attempt
        that failed before its first save is skipped), chosen by the host (C2:
        dir names, never LATEST), only when that attempt ran the same image (C1: the image is part of the cache key,
        re-checked against the host-written run.json) and its tree holds plain files only (C3).
        """
        attempts = attempt_dirs(base)
        free = next((d for d in attempts if not self._used(d)), None)
        if free is None:
            raise AttemptsExhaustedError(
                f"node {node.node_id}: {MAX_ATTEMPTS} attempts used",
                params={"node": node.node_id, "attempt_count": MAX_ATTEMPTS},
            )
        identity = worker_identity(self.worker, node.spec)
        for prev in reversed(attempts[: attempts.index(free)]):  # newest earlier attempt with a usable checkpoint
            if (prev / NOT_RESUMABLE_MARKER).exists():
                return free, None, None  # that attempt said its checkpoints cannot be resumed: start over
            ckpts = checkpoint_dirs(prev)
            if not ckpts:
                continue  # e.g. it failed before its first save: fall back to the attempt before it
            step, ckpt = ckpts[-1]
            problem = checkpoint_tree_problem(ckpt)
            if problem is not None:
                logger.warning("not resuming %s from %s: %s is not a plain file or dir", node.node_id, ckpt, problem)
                return free, None, None
            # identity = "container/<image>@<id>[+recipe:<sha12>]": compare the image id only (C1)
            if "@" in identity and _manifest_digest(prev) != identity.rsplit("@", 1)[1].split("+", 1)[0]:
                logger.warning("not resuming %s: the earlier attempt's image cannot be confirmed", node.node_id)
                return free, None, None
            return free, ckpt, step
        return free, None, None

    def _pick_dir(self, run: GraphRun, node: _Node, key: str) -> tuple[Path, Path | None, int | None] | None:
        """This try's run dir (generation + attempt, P017) and resume source, its C6 lease taken; None = node failed."""
        nid = node.node_id
        base_name = f"{nid}-{key.split('-', 1)[1][:16]}"
        try:
            gen = node_generation(self.workspace, base_name) if self.workspace is not None else 0
            base = self.state_dir / "runs" / (f"{base_name}-g{gen}" if gen else base_name)
            run_dir, resume_from, resume_step = self._attempt(node, base)
        except (AttemptsExhaustedError, GenerationError) as exc:
            error = classify(exc, cancelled=False, checkpoint_step=None).error
            assert error is not None  # noqa: S101 - a catalog error, never "cancelled"
            run.errors[nid] = error
            self._set(run, nid, "failed", cache_key=key, error=_error_dict(error))
            return None
        if resume_from is not None and self.workspace is not None:  # C6: keep the source from now on (host side)
            write_resume_lease(self.workspace, run_dir.name, run=resume_from.parent.parent.name,
                               checkpoint=resume_from.name, node=nid)  # fmt: skip
        return run_dir, resume_from, resume_step

    def _compute(
        self,
        run: GraphRun,
        node: _Node,
        *,
        key: str,
        inputs: dict[str, ArtifactRef],
        upstream: list[tuple[str, str, str]],
        keys: dict[str, str],
        cancel: threading.Event,
    ) -> None:
        nid = node.node_id
        if node.spec.cacheable and self._try_cache(run, nid, key):
            return

        picked = self._pick_dir(run, node, key)
        if picked is None:
            return
        run_dir, resume_from, resume_step = picked
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / NOT_RESUMABLE_MARKER).exists() and os.path.lexists(run_dir / CHECKPOINT_MARKER):
            # The worker will start over: tell the user instead of silently redoing the steps (CACHE-001 text:
            # "the saved result cannot be read, this step runs again").
            saved_at = os.lstat(run_dir / CHECKPOINT_MARKER).st_mtime  # epoch seconds, like the cache index
            warning = UserError("CACHE-001", {"node": nid, "saved_at": saved_at, "est_minutes": None})
            run.warnings.setdefault(nid, []).append(warning)
            self._emit(run.graph_run_id, {"type": "warning", "node_id": nid, "error": _error_dict(warning)})

        def waiting(info: dict[str, float], nid: str = nid) -> None:  # queued for memory, then for the GPU lock
            extra: dict[str, Any] = {k: info[k] for k in ("need_gb", "avail_gb", "since") if k in info}
            state = "waiting_gpu" if info.get("gpu_busy") else "waiting_memory"
            self._emit(run.graph_run_id, {"type": "node_state", "node_id": nid, "state": state, **extra})

        step = resume_step if resume_from is not None else read_checkpoint_step(run_dir)
        job = NodeJob(nid, node.spec, node.params, inputs, run_dir, cancel, step, waiting, resume_from=resume_from)
        self._set(run, nid, "running", cache_key=key, resume_from_step=job.resume_from_step)
        started = time.time()
        try:
            result = self.worker.run(job)
            missing = set(node.spec.outputs) - set(result.outputs)
            if missing:
                raise RuntimeError(f"worker returned no output for ports {sorted(missing)}")
            check_output_paths(result.outputs, run_dir)
        except Exception as exc:  # noqa: BLE001 - every worker failure becomes a catalog error, never a crash
            # A try that failed before saving keeps the earlier attempt's checkpoint as its resume point.
            saved = read_checkpoint_step(run_dir) if resume_from is None else (read_checkpoint_step(run_dir) or step)
            outcome = classify(exc, cancelled=cancel.is_set(), checkpoint_step=saved)
            if outcome.error is not None:
                run.errors[nid] = outcome.error
            logger.info("node %s %s: %s", nid, outcome.status, outcome.error.code if outcome.error else "cancelled")
            status = RunStatus.FAILED if outcome.status == "failed" else RunStatus.INTERRUPTED
            code = outcome.error.code if outcome.error else None
            _write_node_manifest(
                run_dir,
                node_id=nid,
                spec=node.spec,
                gid=run.graph_run_id,
                status=status,
                error_code=code,
                started=started,
            )
            failure = {"error": _error_dict(outcome.error, node)} if outcome.error else {}
            self._set(run, nid, outcome.status, cache_key=key, **failure)
            self._drop_lease(run_dir)  # an interrupted attempt's run.json says resumed_from: retention takes over
            return
        run.outputs[nid] = dict(result.outputs)
        if node.spec.cacheable:
            self.index.record(
                key,
                run_id=run_dir.name,  # the node run dir (outputs/graph/runs/<run_id>), not the graph run
                run_dir=run_dir,  # absolute host path: retention evicts bytes inside it
                node_type=node.spec.type_id,
                outputs={p: OutputRef(r.type, r.path, r.hash) for p, r in result.outputs.items()},
                parents=[keys[src] for src, _, _ in upstream],
                trained_model=produces_trained_model(node.spec),
            )
            self.index.lease([key], run.graph_run_id)  # downstream nodes of this run still read it
        _write_node_manifest(
            run_dir, node_id=nid, spec=node.spec, gid=run.graph_run_id, status=RunStatus.COMPLETED, error_code=None,
            started=started,
        )  # fmt: skip
        self._set(run, nid, "completed", cache_key=key)
        self._drop_lease(run_dir)

    def _drop_lease(self, run_dir: Path) -> None:
        """End of the node: its resume lease (if any) goes; a crash leaves it to stale_runs (never to retention)."""
        if self.workspace is not None:
            drop_resume_lease(self.workspace, run_dir.name)

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
        with self._lock:
            sink = self._sinks.get(gid, self.on_event)
        if sink is None:
            return
        try:
            sink({"graph_run_id": gid, "ts": time.time(), **event})
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
        "inner_path": list(issue.inner_path),  # P016: where inside a group the problem is
    }


def _error_dict(err: UserError, node: _Node | None = None) -> dict[str, Any]:
    """The error of an event, with its G-3 quick fixes computed from the failing node's own params."""
    params = node.params.model_dump(mode="json") if node is not None else None
    model = type(node.params) if node is not None else None
    return {
        "code": err.code,
        "params": dict(err.params),
        "resumable": err.resumable,
        "checkpoint_step": err.checkpoint_step,
        "details": err.details,
        **issue_actions(err, node_id=node.node_id if node else None, node_params=params, params_model=model),
    }


__all__ = ["DONE", "Executor", "GraphRun", "NonHostOutputError", "check_output_paths"]
