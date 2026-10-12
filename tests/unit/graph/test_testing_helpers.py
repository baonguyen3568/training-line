"""Tests for the graph test doubles themselves (``vlm_pipeline.graph.testing``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.graph.errors import ContainerExit as RealContainerExit
from vlm_pipeline.graph.testing import ContainerExit, FakeWorker, make_test_registry
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, read_checkpoint_step


def _job(tmp_path: Path, type_id: str, node_id: str = "n", **params: object) -> NodeJob:
    spec = make_test_registry().get(type_id)
    assert spec is not None
    inputs = {name: ArtifactRef(type=port.type, path="x", hash="h-" + name) for name, port in spec.inputs.items()}
    return NodeJob(node_id=node_id, spec=spec, params=spec.validate_params(params), inputs=inputs, run_dir=tmp_path)


def test_registry_has_the_five_dummy_nodes() -> None:
    ids = {spec.type_id for spec in make_test_registry().all()}
    assert ids == {"test.dataset", "test.model", "test.quant", "test.train", "test.eval"}


def test_every_param_used_by_the_specs_is_declared() -> None:
    reg = make_test_registry()
    assert reg.get("test.dataset").validate_params({"path": "data", "note": "x"})  # type: ignore[union-attr]
    assert reg.get("test.model").validate_params({"revision": "f" * 40})  # type: ignore[union-attr]
    train = reg.get("test.train").validate_params({})  # type: ignore[union-attr]
    assert (train.max_steps, train.lr) == (100, 1e-4)  # type: ignore[attr-defined]
    schema = reg.get("test.dataset").params_schema()  # type: ignore[union-attr]
    assert schema["properties"]["path"]["format"] == "local-path"


def test_outputs_are_deterministic_and_input_sensitive(tmp_path: Path) -> None:
    worker = FakeWorker()
    a = worker.run(_job(tmp_path, "test.train", max_steps=10)).outputs["adapter"]
    b = worker.run(_job(tmp_path, "test.train", max_steps=10)).outputs["adapter"]
    c = worker.run(_job(tmp_path, "test.train", max_steps=11)).outputs["adapter"]
    assert a == b
    assert a.hash != c.hash
    assert a.type == "Adapter"
    assert worker.calls == ["n", "n", "n"]


def test_fail_writes_checkpoint_only_when_asked(tmp_path: Path) -> None:
    with_ckpt, without = tmp_path / "a", tmp_path / "b"
    with pytest.raises(RuntimeError):
        FakeWorker(fail={"n": RuntimeError("x")}, checkpoints={"n"}).run(_job(with_ckpt, "test.model"))
    with pytest.raises(RuntimeError):
        FakeWorker(fail={"n": RuntimeError("x")}).run(_job(without, "test.model"))
    assert read_checkpoint_step(with_ckpt) == 1
    assert read_checkpoint_step(without) is None


def test_container_exit_is_the_real_class() -> None:
    assert ContainerExit is RealContainerExit


def test_outputs_exist_on_disk(tmp_path: Path) -> None:
    """The executor verifies cache hits against the artifact path, so the fake must create it."""
    out = FakeWorker().run(_job(tmp_path / "run", "test.train")).outputs["adapter"]
    assert Path(out.path).is_file()
    assert Path(out.path).read_text() == out.hash


def test_identity_is_fixed_across_subclasses() -> None:
    class Sub(FakeWorker):
        pass

    spec = make_test_registry().get("test.train")
    assert spec is not None
    assert FakeWorker().identity(spec) == Sub().identity(spec) == "fake/1"
