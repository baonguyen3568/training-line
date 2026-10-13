"""Invariant (PM 2026-10-13): if graph validate is clean, the real worker never refuses a node for a support reason.

The real ``RealWorker.run`` -> ``support_query`` -> ``check_support`` path runs; only the heavy handler bodies are
stubbed (they write the same artifact shapes: run.json lineage, DatasetRef descriptors). Every selectable catalog
model goes through every pipeline shape and the parameter variants that change support (quantize format/vision/kv,
eval suites, bench scenario/quantized, trial image), including models that reach a node only after training.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

import pytest

from vlm_pipeline.app.workers import RealWorker
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.common.catalog import CatalogEntry, load_catalog, load_dataset_catalog
from vlm_pipeline.common.dataset_ref import DatasetRef, write_dataset_ref
from vlm_pipeline.common.runinfo import RunManifest, write_manifest
from vlm_pipeline.eval.config import EvalSuite
from vlm_pipeline.graph import validate
from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult, worker_support_problem, worker_supports
from vlm_pipeline.nodes import REGISTRY

SUPPORT_CODES = frozenset({"CFG-010", "CFG-011", "EXP-002", "DATA-005"})  # "this worker does not run that"
CORD = next(d for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")
UPLOAD = "0" * 32


class StubbedWorker(RealWorker):
    """RealWorker with its support path intact and cheap handler bodies (no container, no model files)."""

    @functools.cached_property
    def _handlers(self) -> dict[str, Callable[[NodeJob], NodeResult]]:
        return {spec.type_id: functools.partial(self._stub, spec.type_id) for spec in REGISTRY.all()}

    def _stub(self, type_id: str, job: NodeJob) -> NodeResult:
        run_dir = job.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        params = job.params
        if type_id == "model.hf_pick":
            entry = self.catalog.get(str(getattr(params, "model")))  # noqa: B009
            assert entry is not None
            return NodeResult({"model": ArtifactRef("Model", str(run_dir), f"{entry.repo}@{entry.revision}")})
        if type_id in ("data.hf_load", "data.local_folder"):
            ref = (
                DatasetRef(repo=CORD.repo, revision=CORD.revision, path="/workspace/d")
                if type_id == "data.hf_load"
                else DatasetRef(path="/workspace/mine", split="all", records="/workspace/r", records_sha256="a" * 64)
            )
            return NodeResult({"dataset": self._write_ref(run_dir / "d.json", ref)})
        if type_id == "data.split":
            src = self.dataset_input(job.inputs["dataset"])
            return NodeResult({
                "train": self._write_ref(run_dir / "t.json", src.model_copy(update={"split": "train"})),
                "val": self._write_ref(run_dir / "v.json", src.model_copy(update={"split": "validation"})),
            })  # fmt: skip
        out = run_dir / "out"
        out.mkdir(exist_ok=True)
        model_port = next((p for p in ("base_model", "model") if p in job.inputs), None)
        if model_port is not None:  # trained/derived artifacts carry their base in run.json (lineage)
            base = self.base_of(job.inputs[model_port])
            write_manifest(run_dir, RunManifest(run_id=run_dir.name, name=type_id, stage="train", task=type_id,
                                                base_model=base.repo, base_model_revision=base.revision))  # fmt: skip
        port, port_spec = next(iter(job.spec.outputs.items()))
        return NodeResult({port: ArtifactRef(port_spec.type, str(out), f"{run_dir.name}:h")})

    def _write_ref(self, path: Path, ref: DatasetRef) -> ArtifactRef:
        write_dataset_ref(path, ref)
        return ArtifactRef("Dataset", str(path), ref.hash)


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def _wf(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format": "vlm-workflow", "schema_version": 1, "name": "inv", "nodes": nodes, "edges": edges}


def _trained(entry: CatalogEntry, train: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, str]]:
    """pick -> train (-> merge for LoRA): nodes, edges and the (node, port) that carries the trained model."""
    nodes = [_n("m", "model.hf_pick", model=entry.id), _n("d", "data.hf_load"), _n("t", train)]
    edges = [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data")]
    if train == "train.lora_vlm":
        nodes.append(_n("mg", "train.merge_lora"))
        edges += [_e("m", "model", "mg", "base_model"), _e("t", "adapter", "mg", "adapter")]
        return nodes, edges, ("mg", "model")
    return nodes, edges, ("t", "model")


def _cases() -> Iterator[tuple[str, dict[str, Any]]]:
    for entry in load_catalog().selectable():
        mid = entry.id
        for train in ("train.lora_vlm", "train.sft_vlm"):
            nodes, edges, (src, out) = _trained(entry, train)
            for suite in get_args(EvalSuite):
                yield (
                    f"{mid}/{train}/eval-{suite}",
                    _wf([*nodes, _n("ev", "eval.benchmarks", suite=suite)], [*edges, _e(src, out, "ev", "model")]),
                )
            yield f"{mid}/{train}/export", _wf([*nodes, _n("x", "export.hf_dir")], [*edges, _e(src, out, "x", "model")])
        pick = [_n("m", "model.hf_pick", model=mid)]
        for data in ("data.hf_load", "data.local_folder"):
            for vision in (False, True):
                for kv in ("none", "fp8"):
                    q = _n("q", "quantize.modelopt_ptq", format="fp8", quantize_vision=vision, kv_cache_quant=kv)
                    for scenario in ("single_user", "multi_user"):
                        yield f"{mid}/quant-{data}-v{vision}-kv{kv}/bench-{scenario}", _wf(
                            [*pick, _n("c", data), q, _n("b", "bench.perf", scenario=scenario)],
                            [_e("m", "model", "q", "model"), _e("c", "dataset", "q", "calib_data"),
                             _e("q", "quantized", "b", "model")],
                        )  # fmt: skip
        yield f"{mid}/quant-no-calib", _wf([*pick, _n("q", "quantize.modelopt_ptq")], [_e("m", "model", "q", "model")])
        yield f"{mid}/bench-base", _wf([*pick, _n("b", "bench.perf")], [_e("m", "model", "b", "model")])
        for image in ("", UPLOAD):
            yield (
                f"{mid}/trial-img{bool(image)}",
                _wf([*pick, _n("s", "serve.vllm", image=image)], [_e("m", "model", "s", "model")]),
            )


CASES = list(_cases())


@pytest.mark.parametrize(("name", "workflow"), CASES, ids=[c[0] for c in CASES])
def test_clean_validate_means_no_support_refusal_at_run(tmp_path: Path, name: str, workflow: dict[str, Any]) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    worker = StubbedWorker(root)
    issues = validate(
        workflow, REGISTRY, workspace=root,
        can_run=functools.partial(worker_supports, worker), support=functools.partial(worker_support_problem, worker),
    )  # fmt: skip
    if [i for i in issues if i.severity == "error"]:
        return  # refused on the canvas: the invariant only constrains clean graphs
    run = Executor(REGISTRY, worker, root / "outputs" / "graph", index=ArtifactIndex(tmp_path / "i.db"),
                   workspace=root).run(workflow)  # fmt: skip
    refused = {nid: err.code for nid, err in run.errors.items() if err.code in SUPPORT_CODES}
    assert refused == {}, f"{name}: validate was clean but the worker refused {refused}"


@pytest.mark.parametrize(
    ("case", "code", "node"),
    [
        ("nemotron-parse-v1.1/train.lora_vlm/export", "CFG-010", "t"),  # rule-level (before this change too)
        ("qwen3.5-0.8b/train.lora_vlm/eval-auto", "CFG-011", "ev"),  # eval engine: Parse only today
        ("qwen3.5-0.8b/quant-data.hf_load-vFalse-kvnone/bench-single_user", "CFG-011", "b"),  # bench: Parse only
        ("nemotron-parse-v1.1/trial-imgFalse", "DATA-005", "s"),  # "Chạy thử" without an image
    ],
)
def test_the_canvas_now_catches_what_used_to_fail_at_run(case: str, code: str, node: str) -> None:
    """Teeth: without ``support`` these graphs validated clean and failed only at run time (after the queue)."""
    workflow = dict(CASES)[case]
    worker = StubbedWorker(Path("/nonexistent"))
    with_support = validate(workflow, REGISTRY, support=functools.partial(worker_support_problem, worker))
    assert (code, node) in {(i.code, i.node_id) for i in with_support}
    if code != "CFG-010":
        assert (code, node) not in {(i.code, i.node_id) for i in validate(workflow, REGISTRY)}


def test_run_asks_the_same_check_before_any_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run() calls check_support with the run-time query first; a refusal means the handler never starts."""
    worker = StubbedWorker(tmp_path)
    calls: list[str] = []

    def refuse(query: Any) -> None:
        calls.append(query.spec.type_id)
        raise RuntimeError("refused")

    monkeypatch.setattr(worker, "check_support", refuse)
    monkeypatch.setattr(worker, "_stub", lambda *a, **k: pytest.fail("handler ran before the support check"))
    spec = REGISTRY.get("bench.perf")
    assert spec is not None
    job = NodeJob("b", spec, spec.validate_params({}), {}, tmp_path / "r")
    with pytest.raises(RuntimeError, match="refused"):
        worker.run(job)
    assert calls == ["bench.perf"]


def test_every_node_with_a_model_input_has_a_support_check() -> None:
    """A renamed or removed owner check must fail loudly, never turn the invariant off silently (tester review)."""
    from vlm_pipeline.app import workers

    needs = {
        spec.type_id for spec in REGISTRY.all() if any(p.type in ("Model", "Adapter") for p in spec.inputs.values())
    }
    assert needs <= set(workers._SUPPORT_CHECKS), needs - set(workers._SUPPORT_CHECKS)
    with pytest.raises(ImportError, match="check_support is missing"):
        workers._owner_check("vlm_pipeline.common.catalog")
