"""A ``many`` input port receives every wire (data.dedupe heldout), in an order the cache key does not depend on."""

from __future__ import annotations

from vlm_pipeline.graph.executor import gather_inputs
from vlm_pipeline.graph.worker import ArtifactRef, port_inputs, port_of
from vlm_pipeline.nodes import REGISTRY


def _ref(name: str) -> ArtifactRef:
    return ArtifactRef("Dataset", f"/x/{name}.json", f"h-{name}")


def test_every_wire_of_a_many_port_reaches_the_worker_in_a_stable_order() -> None:
    spec = REGISTRY.get("data.dedupe")
    assert spec is not None
    assert spec.inputs["heldout"].many
    wired = [("e2", "dataset", "heldout", _ref("e2")), ("d", "dataset", "dataset", _ref("d")),
             ("e1", "dataset", "heldout", _ref("e1"))]  # fmt: skip
    inputs = gather_inputs(spec, wired)
    assert set(inputs) == {"dataset", "heldout", "heldout.1"}
    assert [r.hash for r in port_inputs(inputs, "heldout")] == ["h-e1", "h-e2"]
    assert gather_inputs(spec, list(reversed(wired))) == inputs  # wiring order never changes the cache key
    assert port_inputs(inputs, "dataset") == [_ref("d")]
    assert port_of("heldout.1") == "heldout"


def test_renaming_the_source_nodes_keeps_the_same_inputs() -> None:
    spec = REGISTRY.get("data.dedupe")
    assert spec is not None
    a = [("e1", "dataset", "heldout", _ref("x")), ("e2", "dataset", "heldout", _ref("y"))]
    b = [("e2", "dataset", "heldout", _ref("x")), ("e1", "dataset", "heldout", _ref("y"))]  # ids swapped
    assert gather_inputs(spec, a) == gather_inputs(spec, b)
