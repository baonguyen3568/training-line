"""What the HTTP layer needs from the rest of the system, as one injectable object (tests pass fakes).

``default_services`` wires the real modules lazily: ``graph`` (validate/executor, bao-fixcode), ``artifacts.runs_index``
(runs + metrics pages, bao-cleanup), ``nodes`` (registry), ``common`` (memory/disk/lock). A module that has not landed
yet makes only its endpoints answer 501, never the whole app.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.nodes.spec import Registry

EventSink = Callable[[dict[str, Any]], None]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class NotAvailableError(RuntimeError):
    """A backing module is not implemented yet (HTTP 501)."""


class BadRequestError(VlmPipelineError):
    """Invalid user input at the API edge (HTTP 400). Params are safe to show (no paths, no internals)."""

    code: ClassVar[str] = "CFG-005"


class Services(Protocol):
    registry: Registry

    def validate(self, workflow: Mapping[str, Any]) -> list[dict[str, Any]]: ...
    def run_graph(self, workflow: Mapping[str, Any], graph_run_id: str, on_event: EventSink) -> None: ...
    def cancel(self, graph_run_id: str) -> bool: ...
    def list_runs(self) -> list[dict[str, Any]]: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def metrics_page(self, run_id: str, after: str | None, limit: int, downsample_to: int | None) -> dict[str, Any]: ...
    def system(self) -> dict[str, Any]: ...


def _issue_to_dict(issue: object) -> dict[str, Any]:
    dump = getattr(issue, "model_dump", None)
    if callable(dump):
        result: dict[str, Any] = dump(mode="json")
        return result
    if isinstance(issue, Mapping):
        return dict(issue)
    if dataclasses.is_dataclass(issue) and not isinstance(issue, type):
        return dataclasses.asdict(issue)
    raise TypeError(f"unsupported issue type {type(issue).__name__}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _PacedWorker:
    """Delays a (fake) worker so node states are observable in the UI; honours cancellation while waiting."""

    def __init__(self, inner: Any, delay_s: float) -> None:
        self._inner = inner
        self._delay = delay_s

    def run(self, job: Any) -> Any:
        cancel = getattr(job, "cancel", None)
        if cancel is not None:
            cancel.wait(self._delay)
        else:
            time.sleep(self._delay)
        return self._inner.run(job)


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and module.startswith(exc.name):
            raise NotAvailableError(f"{module} is not implemented yet") from exc
        raise


@dataclass
class DefaultServices:
    registry: Registry
    root: Path
    worker: str = "fake"  # "fake" (FakeWorker, demo/E2E) | "container" (real jobs)
    fake_delay_s: float = 0.8  # per node, so the canvas visibly animates in demo/E2E

    @property
    def _state_dir(self) -> Path:
        """Executor state + run dirs; demo (fake worker) runs are kept apart from real ones."""
        base = self.root / "outputs" / "graph"
        return base / "demo" if self.worker == "fake" else base

    @functools.cached_property
    def index(self) -> Any:
        """The single cache index shared by executor, retention and the storage panel."""
        artifacts_index = _optional("vlm_pipeline.artifacts.index")
        path = artifacts_index.default_index_path(self.root)
        if self.worker == "fake":  # demo artifacts must never satisfy a real run's cache lookup
            path = self._state_dir / "app.db"
        idx = artifacts_index.ArtifactIndex(path)
        idx.release_all_except(self._active_external_runs())  # leases left by a crashed app process
        return idx

    def _active_external_runs(self) -> list[str]:
        """run_ids still queued/running with a live pid (graphs started outside this app must keep their leases)."""
        from vlm_pipeline.artifacts.runs_index import list_runs

        return [
            m.run_id for m in list_runs(self.root) if m.status in {"queued", "running"} and m.pid and _pid_alive(m.pid)
        ]

    @functools.cached_property
    def executor(self) -> Any:
        """ONE executor per app: cancel() only sees runs started by the same instance."""
        graph_executor = _optional("vlm_pipeline.graph.executor")
        worker: Any
        if self.worker == "fake":
            worker = _PacedWorker(_optional("vlm_pipeline.graph.testing").FakeWorker(), self.fake_delay_s)
        else:
            from vlm_pipeline.app.workers import RealWorker

            worker = RealWorker(self.root)
        return graph_executor.Executor(self.registry, worker, self._state_dir, index=self.index, workspace=self.root)

    def validate(self, workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
        graph = _optional("vlm_pipeline.graph")
        validate = getattr(graph, "validate", None)
        if validate is None:
            raise NotAvailableError("vlm_pipeline.graph.validate is not implemented yet")
        from vlm_pipeline.graph.worker import worker_supports

        worker = self.executor.worker
        can_run = functools.partial(worker_supports, worker)
        return [
            _issue_to_dict(i) for i in validate(dict(workflow), self.registry, workspace=self.root, can_run=can_run)
        ]

    def run_graph(self, workflow: Mapping[str, Any], graph_run_id: str, on_event: EventSink) -> None:
        self.executor.run(dict(workflow), graph_run_id=graph_run_id, on_event=on_event)

    def cancel(self, graph_run_id: str) -> bool:
        return bool(self.executor.cancel(graph_run_id))

    def list_runs(self) -> list[dict[str, Any]]:
        index = _optional("vlm_pipeline.artifacts.runs_index")
        return [m.model_dump(mode="json") for m in index.list_runs(self.root)]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return next((r for r in self.list_runs() if r.get("run_id") == run_id), None)

    def metrics_page(self, run_id: str, after: str | None, limit: int, downsample_to: int | None) -> dict[str, Any]:
        index = _optional("vlm_pipeline.artifacts.runs_index")
        if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
            raise BadRequestError(params={"param": "run_id", "rule": "format"})
        run_dir = self.root / "outputs" / "runs" / run_id
        try:
            page = index.read_metrics_page(run_dir, after=after, limit=limit, downsample_to=downsample_to)
        except ValueError as exc:  # runs_index only raises ValueError for bad cursor/limit/downsample_to
            raise BadRequestError(params={"param": "after|limit|downsample_to", "rule": "range"}) from exc
        result: dict[str, Any] = page.model_dump(mode="json")
        return result

    def system(self) -> dict[str, Any]:
        from vlm_pipeline.common import gpu_lock
        from vlm_pipeline.common.diskguard import read_disk
        from vlm_pipeline.common.memguard import read_meminfo

        mem = read_meminfo()
        disk = read_disk(self.root)
        lock: dict[str, Any] = {"held": False}
        try:
            info = gpu_lock.read_lock(gpu_lock.default_lock_path())
            if info is not None:
                lock = {"held": True, "who": info.who, "job": info.job, "since": info.started_at,
                        "stale": gpu_lock.is_stale(info)}  # fmt: skip
        except gpu_lock.GpuLockError:
            lock = {"held": True, "who": "?", "job": "?", "since": "?", "stale": False, "corrupt": True}
        return {
            "mem_total_gb": round(mem.mem_total_gb, 1),
            "mem_available_gb": round(mem.mem_available_gb, 1),
            "disk_free_gb": round(disk.free_gb, 1),
            "gpu_lock": lock,
            "worker": self.worker,  # "fake" -> the UI shows the demo-mode banner
        }
