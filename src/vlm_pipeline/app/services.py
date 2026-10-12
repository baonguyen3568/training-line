"""What the HTTP layer needs from the rest of the system, as one injectable object (tests pass fakes).

``default_services`` wires the real modules lazily: ``graph`` (validate/executor, bao-fixcode), ``artifacts.runs_index``
(runs + metrics pages, bao-cleanup), ``nodes`` (registry), ``common`` (memory/disk/lock). A module that has not landed
yet makes only its endpoints answer 501, never the whole app.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vlm_pipeline.nodes.spec import Registry

EventSink = Callable[[dict[str, Any]], None]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class NotAvailableError(RuntimeError):
    """A backing module is not implemented yet (HTTP 501)."""


class Services(Protocol):
    registry: Registry

    def validate(self, workflow: Mapping[str, Any]) -> list[dict[str, Any]]: ...
    def run_graph(self, workflow: Mapping[str, Any], graph_run_id: str, on_event: EventSink) -> None: ...
    def cancel(self, graph_run_id: str) -> None: ...
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

    def _executor(self, on_event: EventSink) -> Any:
        graph_executor = _optional("vlm_pipeline.graph.executor")
        if self.worker == "fake":
            worker = _optional("vlm_pipeline.graph.testing").FakeWorker()
        else:
            raise NotAvailableError("container worker is not wired yet")
        state_dir = self.root / "outputs" / "graph"
        state_dir.mkdir(parents=True, exist_ok=True)
        return graph_executor.Executor(self.registry, worker, state_dir, on_event=on_event)

    def validate(self, workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
        graph = _optional("vlm_pipeline.graph")
        validate = getattr(graph, "validate", None)
        if validate is None:
            raise NotAvailableError("vlm_pipeline.graph.validate is not implemented yet")
        return [_issue_to_dict(i) for i in validate(dict(workflow), self.registry, workspace=self.root)]

    def run_graph(self, workflow: Mapping[str, Any], graph_run_id: str, on_event: EventSink) -> None:
        self._executor(on_event).run(dict(workflow), graph_run_id=graph_run_id)

    def cancel(self, graph_run_id: str) -> None:
        raise NotAvailableError("cancel is wired with the executor instance (todo)")

    def list_runs(self) -> list[dict[str, Any]]:
        index = _optional("vlm_pipeline.artifacts.runs_index")
        return [m.model_dump(mode="json") for m in index.list_runs(self.root)]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return next((r for r in self.list_runs() if r.get("run_id") == run_id), None)

    def metrics_page(self, run_id: str, after: str | None, limit: int, downsample_to: int | None) -> dict[str, Any]:
        index = _optional("vlm_pipeline.artifacts.runs_index")
        if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
            raise ValueError("invalid run id")
        run_dir = self.root / "outputs" / "runs" / run_id
        page = index.read_metrics_page(run_dir, after=after, limit=limit, downsample_to=downsample_to)
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
        }
