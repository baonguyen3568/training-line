"""Contract between the executor and whatever runs one node (FakeWorker in tests, the job runner for containers).

A worker runs ONE node and either returns its outputs or raises. It never classifies errors itself: the executor
maps any exception to a catalog code (``graph.errors.classify``). Checkpoints are detected from disk, never trusted
from the exception: ``<run_dir>/checkpoints/LATEST`` (see ``read_checkpoint_step``).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from vlm_pipeline.nodes.spec import NodeSpec

CHECKPOINT_MARKER = Path("checkpoints") / "LATEST"


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


@dataclass(frozen=True)
class NodeResult:
    outputs: Mapping[str, ArtifactRef]  # one entry per output port of the spec


class Worker(Protocol):
    """Runs one node.

    Long-running workers (container jobs) MUST watch ``job.cancel``: when it is set, send SIGTERM to the job (grace
    period so it can save a checkpoint), then raise ``ContainerExit(..., cancelled_by_executor=True)``.
    """

    def run(self, job: NodeJob) -> NodeResult: ...


_AUTOMODEL_CKPT = re.compile(r"^epoch_(\d+)_step_(\d+)$")


def read_checkpoint_step(run_dir: Path) -> int | None:
    """Step of the last COMPLETE checkpoint in ``run_dir`` (None if none). Drives ``resumable`` on errors.

    ``checkpoints/LATEST`` is either a text file holding the step (FakeWorker / simple workers) or, as NeMo AutoModel
    writes it, a symlink to ``epoch_<E>_step_<S>``; a checkpoint dir that still has ``.incomplete`` does not count.
    """
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


__all__ = ["CHECKPOINT_MARKER", "ArtifactRef", "NodeJob", "NodeResult", "Worker", "read_checkpoint_step"]
