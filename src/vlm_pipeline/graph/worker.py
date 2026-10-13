"""Contract between the executor and whatever runs one node (FakeWorker in tests, the job runner for containers).

A worker runs ONE node and either returns its outputs or raises. It never classifies errors itself: the executor
maps any exception to a catalog code (``graph.errors.classify``). Checkpoints are detected from disk, never trusted
from the exception: ``<run_dir>/checkpoints/LATEST`` (see ``read_checkpoint_step``).
"""

from __future__ import annotations

import os
import re
import stat
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
    # Host path of the checkpoint step dir of an EARLIER attempt to resume from (P017: a retry runs in a new attempt
    # dir; the old checkpoint is a read-only input). None: start fresh, or (fake/host workers) resume in ``run_dir``.
    resume_from: Path | None = None


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


MAX_ATTEMPTS = 10  # PM 2026-10-13: an 11th try of the same node is a configuration problem, not bad luck


def attempt_dirs(base: Path) -> list[Path]:
    """``<nid>-<key16>``, then ``-a2`` .. ``-a10``: one run dir per job attempt of a node (P017)."""
    return [base, *(base.with_name(f"{base.name}-a{n}") for n in range(2, MAX_ATTEMPTS + 1))]


def checkpoint_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Complete AutoModel checkpoint step dirs of ``run_dir`` as (step, path), newest last (P017 C2).

    Chosen by the HOST: real directories whose name matches ``epoch_<E>_step_<S>`` (lstat, never a symlink), without
    ``.incomplete``; ``LATEST`` and any other pointer the job wrote are never followed. The step is the int in the name.
    """
    root = run_dir / "checkpoints"
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        match = _AUTOMODEL_CKPT.match(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir() or os.path.lexists(entry / ".incomplete"):
            continue
        found.append((int(match.group(2)), entry))
    return sorted(found)


def checkpoint_tree_problem(ckpt: Path) -> str | None:
    """The first entry of a checkpoint tree that is not a plain dir/file (symlink, fifo...), or None (P017 C3).

    A container follows a symlink INSIDE a read-only mount: one pointing at /run/secrets or /workspace must never be
    handed to the next job.
    """
    unreadable: list[str] = []

    def failed(err: OSError) -> None:  # a dir that cannot be listed cannot be verified: fail closed
        unreadable.append(str(err.filename))

    try:
        for dirpath, dirnames, filenames in os.walk(ckpt, followlinks=False, onerror=failed):
            for name in (*dirnames, *filenames):
                entry = Path(dirpath) / name
                mode = entry.lstat().st_mode
                if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    return entry.relative_to(ckpt).as_posix()
    except OSError as exc:  # an entry vanished or cannot be read while walking: not a checkpoint to trust
        return f"unreadable: {exc.strerror}"
    return f"unreadable: {unreadable[0]}" if unreadable else None


__all__ = [
    "CHECKPOINT_MARKER",
    "MAX_ATTEMPTS",
    "NOT_RESUMABLE_MARKER",
    "ArtifactRef",
    "NodeJob",
    "NodeResult",
    "SupportQuery",
    "Worker",
    "attempt_dirs",
    "checkpoint_dirs",
    "checkpoint_tree_problem",
    "read_checkpoint_step",
    "worker_identity",
    "worker_support_problem",
    "worker_supports",
]
