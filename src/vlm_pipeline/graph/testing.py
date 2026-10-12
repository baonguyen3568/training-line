"""Test doubles for the graph layer (like ``django.test``): a CPU-only registry of dummy nodes and a fake worker.

Used by tests/unit/graph/*_spec.py and the M1 E2E (FakeWorker behind the real API). Built only through the public
``nodes.spec`` API so the registry also exercises it. Nothing here touches a GPU, a container or the network.

Dummy nodes (every param a test uses is declared, so no test can pass on "unknown param" by accident):
  test.dataset  out data: Dataset              params path (local path, "data"), note (str, "")
  test.model    out model: Model               params revision (40-hex, "0"*40)
  test.quant    in model: Model  -> out model: QuantizedModel
  test.train    in data: Dataset, base_model: Model -> out adapter: Adapter
                                               params max_steps (1..100000, 100), lr (> 0, 1e-4)
  test.eval     in model: Model  -> out report: EvalReport
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping

from pydantic import Field

from vlm_pipeline.common.params import Params, ui
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import CHECKPOINT_MARKER, ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.nodes.spec import LocalPath, NodeSpec, PortSpec, Registry


class DatasetParams(Params):
    path: LocalPath = Field("data", json_schema_extra=ui(order=0))  # shared type: format "local-path"
    note: str = ""


class ModelParams(Params):
    revision: str = Field("0" * 40, pattern=r"^[0-9a-f]{40}$")


class TrainParams(Params):
    max_steps: int = Field(100, ge=1, le=100_000)
    lr: float = Field(1e-4, gt=0)


def _spec(type_id: str, category: str, **kwargs: object) -> NodeSpec:
    return NodeSpec.model_validate(
        {"type_id": type_id, "version": "1.0.0", "category": category, "icon": "box", "runner": "host", **kwargs}
    )


def make_test_registry() -> Registry:
    """Fresh registry of the five dummy node types (see module docstring)."""
    return Registry(
        [
            _spec("test.dataset", "source", outputs={"data": PortSpec(type="Dataset")}, params_model=DatasetParams),
            _spec("test.model", "source", outputs={"model": PortSpec(type="Model")}, params_model=ModelParams),
            _spec(
                "test.quant",
                "quantize",
                inputs={"model": PortSpec(type="Model")},
                outputs={"model": PortSpec(type="QuantizedModel")},
            ),
            _spec(
                "test.train",
                "train",
                inputs={"data": PortSpec(type="Dataset"), "base_model": PortSpec(type="Model")},
                outputs={"adapter": PortSpec(type="Adapter")},
                params_model=TrainParams,
            ),
            _spec(
                "test.eval",
                "eval",
                inputs={"model": PortSpec(type="Model")},
                outputs={"report": PortSpec(type="EvalReport")},
            ),
        ]
    )


def _output_hash(job: NodeJob, port: str) -> str:
    """Deterministic content hash: same node, params and inputs => same hash (drives downstream cache keys)."""
    payload = json.dumps(
        {
            "type_id": job.spec.type_id,
            "version": job.spec.version,
            "params": job.params.model_dump(mode="json"),
            "inputs": {name: ref.hash for name, ref in sorted(job.inputs.items())},
            "port": port,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class FakeWorker:
    """Records ``calls`` (node ids in execution order) and returns deterministic artifacts.

    ``fail`` maps node ids to the exception to raise; for ids in ``checkpoints`` the worker first writes the
    checkpoint marker (step 1), exactly like a real worker that saved a checkpoint before failing.
    """

    def __init__(self, fail: Mapping[str, BaseException] | None = None, checkpoints: Iterable[str] = ()) -> None:
        self.fail = dict(fail or {})
        self.checkpoints = set(checkpoints)
        self.calls: list[str] = []

    def run(self, job: NodeJob) -> NodeResult:
        self.calls.append(job.node_id)
        if job.node_id in self.fail:
            if job.node_id in self.checkpoints:
                marker = job.run_dir / CHECKPOINT_MARKER
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("1", encoding="utf-8")
            raise copy.copy(self.fail[job.node_id])  # fresh instance: no traceback piling up across runs
        # Like a real worker, the artifact exists on disk: the executor verifies cache hits against it.
        job.run_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for port, spec in job.spec.outputs.items():
            digest = _output_hash(job, port)
            path = job.run_dir / port
            path.write_text(digest, encoding="utf-8")
            outputs[port] = ArtifactRef(type=spec.type, path=str(path), hash=digest)
        return NodeResult(outputs=outputs)


__all__ = ["ContainerExit", "DatasetParams", "FakeWorker", "ModelParams", "TrainParams", "make_test_registry"]
