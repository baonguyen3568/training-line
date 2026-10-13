"""Contract between the executor and whatever runs one node (FakeWorker in tests, the job runner for containers).

A worker runs ONE node and either returns its outputs or raises. It never classifies errors itself: the executor
maps any exception to a catalog code (``graph.errors.classify``). Checkpoints are detected from disk, never trusted
from the exception: ``<run_dir>/checkpoints/LATEST`` (see ``read_checkpoint_step``).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.nodes.spec import NodeSpec

CHECKPOINT_MARKER = Path("checkpoints") / "LATEST"
# Written by a worker whose checkpoints cannot be resumed (e.g. a model whose saved state misses keys): its checkpoints
# never count as resumable, and the worker starts such a run dir over. Permanent for that run dir: fixing the model
# later does not make checkpoints saved before the fix resumable.
NOT_RESUMABLE_MARKER = ".not-resumable"


@dataclass(frozen=True)
class ArtifactRef:
    """One output of a node. ``hash`` identifies the content (manifest hash), it feeds downstream cache keys."""

    type: str
    path: str
    hash: str


@dataclass(frozen=True)
class NodeJob:
    node_id: str
    spec: NodeSpec
    params: BaseModel  # validated, defaults filled in
    inputs: Mapping[str, ArtifactRef]  # input port -> upstream artifact
    run_dir: Path  # created by the executor, empty unless resuming
    cancel: threading.Event = field(default_factory=threading.Event)  # set => stop ASAP (SIGTERM, save ckpt)
    resume_from_step: int | None = None  # set when resuming an interrupted node from its checkpoint
    # Called while a job waits for free memory (``jobrun.run_job(on_wait=...)``): {need_gb, avail_gb, waited_s, since}.
    on_wait: Callable[[dict[str, float]], None] | None = None


@dataclass(frozen=True)
class NodeResult:
    outputs: Mapping[str, ArtifactRef]  # one entry per output port of the spec


class Worker(Protocol):
    """Runs one node.

    Long-running workers (container jobs) MUST watch ``job.cancel``: when it is set, send SIGTERM to the job (grace
    period so it can save a checkpoint), then raise ``ContainerExit(..., cancelled_by_executor=True)``.
    """

    def run(self, job: NodeJob) -> NodeResult: ...


@dataclass(frozen=True)
class SupportQuery:
    """What decides whether a worker can run a node, known both on the canvas and at run time (no file access).

    ``model``: the catalog model at the origin of the node's model input (None: no model input, or not traceable).
    ``inputs``: wired ports only, ``port -> "dataset:hub" | "dataset:local" | "model" | "other"``.
    """

    node_id: str
    spec: NodeSpec
    params: BaseModel
    model: CatalogEntry | None = None
    inputs: Mapping[str, str] = field(default_factory=dict)


def worker_support_problem(worker: object, query: SupportQuery) -> VlmPipelineError | None:
    """The catalog error a worker would raise for ``query`` before running it (optional ``check_support``), or None.

    ONE function decides, for the canvas (graph validate) and for the run: "validate is clean" implies "the worker
    does not refuse for a support reason".
    """
    check = getattr(worker, "check_support", None)
    if not callable(check):
        return None
    try:
        check(query)
    except VlmPipelineError as exc:
        return exc
    return None


def worker_supports(worker: object, spec: NodeSpec) -> bool:
    """Whether ``worker`` can run ``spec`` (optional ``supports(spec) -> bool``; workers without it run every node).

    Validation uses it so a node the real worker cannot run yet is refused before the run (CFG-011).
    """
    supports = getattr(worker, "supports", None)
    return bool(supports(spec)) if callable(supports) else True


def worker_identity(worker: object, spec: NodeSpec) -> str:
    """What runs ``spec`` on ``worker``; part of the cache key.

    Workers may define ``identity(spec) -> str`` (e.g. ``"container/vlm-automodel@sha256:..."`` with the image they
    really use for that node type); otherwise the worker class name is used.
    """
    ident = getattr(worker, "identity", None)
    if callable(ident):
        return str(ident(spec))
    return f"{type(worker).__module__}.{type(worker).__qualname__}"


_AUTOMODEL_CKPT = re.compile(r"^epoch_(\d+)_step_(\d+)$")


def read_checkpoint_step(run_dir: Path) -> int | None:
    """Step of the last COMPLETE checkpoint in ``run_dir`` (None if none). Drives ``resumable`` on errors.

    ``checkpoints/LATEST`` is either a text file holding the step (FakeWorker / simple workers) or, as NeMo AutoModel
    writes it, a symlink to ``epoch_<E>_step_<S>``; a checkpoint dir that still has ``.incomplete`` does not count.
    """
    if (run_dir / NOT_RESUMABLE_MARKER).exists():
        return None
    marker = run_dir / CHECKPOINT_MARKER
    try:
        if marker.is_dir():  # follows the AutoModel symlink
            target = marker.resolve()
            match = _AUTOMODEL_CKPT.match(target.name)
            if match is None or (target / ".incomplete").exists():
                return None
            return int(match.group(2))
        return int(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


__all__ = [
    "CHECKPOINT_MARKER",
    "NOT_RESUMABLE_MARKER",
    "ArtifactRef",
    "NodeJob",
    "NodeResult",
    "SupportQuery",
    "Worker",
    "read_checkpoint_step",
    "worker_identity",
    "worker_support_problem",
    "worker_supports",
]
