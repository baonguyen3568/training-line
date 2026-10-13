"""Group (subgraph) expansion before validation (P016 §2): flat ids, wiring, limits, re-anchored issues."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_pipeline.graph import validate
from vlm_pipeline.graph.groups import GROUP_CODE, MAX_NODES, expand
from vlm_pipeline.graph.testing import make_test_registry
from vlm_pipeline.graph.workflow import GroupDefinition

REGISTRY = make_test_registry()


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def _group(group_id: str, nodes: list[dict[str, Any]], **mapping: Any) -> dict[str, Any]:
    return {"id": group_id, "version": "1.0.0", "title": "My steps", "nodes": nodes, **mapping}


QUANT_EVAL = _group(  # quantize, then score the quantized model
    "user.quant_eval",
    [_n("q", "test.quant"), _n("e", "test.eval")],
    edges=[_e("q", "model", "e", "model")],
    inputs={"model": [["q", "model"]]},
    outputs={"quantized": ["q", "model"]},
)
FINETUNE = _group(
    "user.finetune",
    [_n("t", "test.train", max_steps=10), _n("e", "test.eval")],
    inputs={"data": [["t", "data"]], "model": [["t", "base_model"], ["e", "model"]]},  # one input, two inner ports
    outputs={"adapter": ["t", "adapter"]},
    params={"steps": ["t", "max_steps"]},
)


def _wf(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *groups: dict[str, Any]) -> dict[str, Any]:
    wf = {"format": "vlm-workflow", "schema_version": 1, "name": "g", "nodes": nodes, "edges": edges}
    return {**wf, "groups": list(groups)} if groups else wf


def _finetune(**params: Any) -> dict[str, Any]:
    nodes = [_n("d", "test.dataset"), _n("m", "test.model"), _n("ft", "user.finetune", **params)]
    edges = [_e("d", "data", "ft", "data"), _e("m", "model", "ft", "model")]
    return _wf(nodes, edges, FINETUNE)


def _codes(wf: dict[str, Any]) -> list[tuple[str, str | None, tuple[str, ...]]]:
    return [(i.code, i.node_id, i.inner_path) for i in validate(wf, REGISTRY, None)]


# ------------------------------------------------------------------------------------------- expansion


def test_group_expands_to_flat_ids_and_wires() -> None:
    wf = _finetune(steps=20)
    expansion = expand(wf, REGISTRY, first_party={})
    flat = expansion.workflow
    assert "groups" not in flat
    assert [n["id"] for n in flat["nodes"]] == ["d", "m", "ft__t", "ft__e"]
    assert flat["nodes"][2]["params"] == {"max_steps": 20}  # exposed param overrides the inner default
    assert sorted((e["from"][0], e["to"][0], e["to"][1]) for e in flat["edges"]) == [
        ("d", "ft__t", "data"),
        ("m", "ft__e", "model"),  # the exposed input fans out to both inner ports
        ("m", "ft__t", "base_model"),
    ]
    assert expansion.origins["ft__t"] == ("ft", ("t",))
    assert validate(wf, REGISTRY) == []


def test_group_output_feeds_the_outer_graph() -> None:
    nodes = [_n("m", "test.model"), _n("qe", "user.quant_eval"), _n("x", "test.eval")]
    edges = [_e("m", "model", "qe", "model"), _e("qe", "quantized", "x", "model")]
    flat = expand(_wf(nodes, edges, QUANT_EVAL), REGISTRY, first_party={}).workflow
    assert {"from": ["qe__q", "model"], "to": ["x", "model"]} in flat["edges"]
    assert validate(_wf(nodes, edges, QUANT_EVAL), REGISTRY) == []


def test_nested_groups_get_one_id_segment_per_level() -> None:
    outer = _group(
        "user.outer",
        [_n("inner", "user.quant_eval")],
        inputs={"model": [["inner", "model"]]},
        outputs={"quantized": ["inner", "quantized"]},
    )
    nodes = [_n("m", "test.model"), _n("o", "user.outer")]
    wf = _wf(nodes, [_e("m", "model", "o", "model")], outer, QUANT_EVAL)
    expansion = expand(wf, REGISTRY, first_party={})
    assert [n["id"] for n in expansion.workflow["nodes"]] == ["m", "o__inner__q", "o__inner__e"]
    assert expansion.origins["o__inner__q"] == ("o", ("inner", "q"))
    assert validate(wf, REGISTRY) == []


def test_first_party_groups_are_used_and_cannot_be_redefined() -> None:
    first = GroupDefinition.model_validate(
        {**QUANT_EVAL, "id": "group.quant_eval", "title": None, "title_key": "groups.qe"}
    )
    nodes = [_n("m", "test.model"), _n("qe", "group.quant_eval")]
    wf = _wf(nodes, [_e("m", "model", "qe", "model")])
    assert [n["id"] for n in expand(wf, REGISTRY, first_party={first.id: first}).workflow["nodes"]][1:] == [
        "qe__q",
        "qe__e",
    ]
    redefined = _wf(nodes, [], {**QUANT_EVAL, "id": "group.quant_eval"})
    [issue] = expand(redefined, REGISTRY, first_party={first.id: first}).issues
    assert (issue.code, issue.params["reason"]) == (GROUP_CODE, "namespace")  # a file only defines user.* groups


# --------------------------------------------------------------------------------------- re-anchored issues


def test_inner_param_errors_point_at_the_group_node() -> None:
    [issue] = validate(_finetune(steps=0), REGISTRY)
    assert (issue.code, issue.node_id, issue.inner_path, issue.param) == ("CFG-005", "ft", ("t",), "max_steps")
    [unknown] = validate(_finetune(stepz=5), REGISTRY)
    assert (unknown.code, unknown.node_id, unknown.param, unknown.params["rule"]) == (
        "CFG-005",
        "ft",
        "stepz",
        "extra_forbidden",
    )


def test_missing_input_inside_a_group_keeps_the_inner_port() -> None:
    wf = _finetune()
    wf["edges"] = wf["edges"][:1]  # model input of the group unwired
    assert sorted(_codes(wf)) == [("CFG-003", "ft", ("e",)), ("CFG-003", "ft", ("t",))]


def test_wire_to_an_unexposed_port_is_dangling() -> None:
    wf = _finetune()
    wf["edges"].append(_e("d", "data", "ft", "nope"))
    assert ("CFG-006", "ft", ()) in _codes(wf)


# -------------------------------------------------------------------------------------------- limits / trust


def _reasons(wf: dict[str, Any]) -> list[str]:
    return [str(i.params["reason"]) for i in validate(wf, REGISTRY) if i.code == GROUP_CODE]


def test_recursion_and_depth_are_refused() -> None:
    loop = _group("user.loop", [_n("again", "user.loop")])
    assert _reasons(_wf([_n("l", "user.loop")], [], loop)) == ["recursion"]
    levels = [_group(f"user.l{i}", [_n("x", f"user.l{i + 1}")]) for i in range(3)]
    levels.append(_group("user.l3", [_n("x", "test.model")]))
    assert _reasons(_wf([_n("top", "user.l0")], [], *levels)) == ["too_deep"]
    assert _reasons(_wf([_n("top", "user.l1")], [], *levels)) == []  # three levels are fine


def test_node_count_after_expansion_is_capped() -> None:
    wide = _group("user.wide", [_n(f"m{i}", "test.model") for i in range(50)])
    nodes = [_n(f"g{i}", "user.wide") for i in range(MAX_NODES // 50 + 1)]
    assert _reasons(_wf(nodes, [], wide)) == ["too_many_nodes"]


def test_bad_definitions_are_reported_without_echoing_them() -> None:
    bad = {**QUANT_EVAL, "id": "user.<script>"}
    issues = [i for i in validate(_wf([_n("m", "test.model")], [], bad), REGISTRY) if i.code == GROUP_CODE]
    assert [(i.params["group"], i.params["reason"]) for i in issues] == [("", "invalid_definition")]
    mapped = {**QUANT_EVAL, "outputs": {"quantized": ["q", "nope"]}}
    nodes = [_n("m", "test.model"), _n("qe", "user.quant_eval")]
    assert _reasons(_wf(nodes, [_e("m", "model", "qe", "model")], mapped)) == ["bad_mapping"]
    twice = _wf([_n("m", "test.model")], [], QUANT_EVAL, QUANT_EVAL)
    assert _reasons(twice) == ["name_clash"]


def test_separator_is_reserved_and_generated_ids_stay_short() -> None:
    assert ("CFG-007", None, ()) in _codes(_wf([_n("a__b", "test.model")], [], QUANT_EVAL))
    inner = {**QUANT_EVAL, "nodes": [_n("q__x", "test.quant")], "edges": [], "outputs": {}, "inputs": {}}
    assert _reasons(_wf([_n("m", "test.model")], [], inner)) == ["invalid_definition"]
    long_id = "g" * 62
    assert _reasons(_wf([_n(long_id, "user.quant_eval")], [], QUANT_EVAL)) == ["id_too_long", "id_too_long"]


def test_open_node_is_offered_only_for_issues_on_a_group_node() -> None:
    from vlm_pipeline.common.error_catalog import available_actions

    for issue in validate(_wf([_n("l", "user.loop")], [], _group("user.loop", [_n("again", "user.loop")])), REGISTRY):
        assert (issue.node_id, available_actions(issue)) == ("l", ["open_node"])
    for issue in validate(_wf([_n("m", "test.model")], [], QUANT_EVAL, QUANT_EVAL), REGISTRY):  # file-level
        assert (issue.node_id, available_actions(issue)) == (None, [])


def test_nested_fan_out_is_cut_by_one_budget() -> None:
    """Security G1: 500 instances per level, 5 levels (a ~150 KB file) must stop at once, not after N^3 expansions."""
    import time

    levels = [_group(f"user.l{i}", [_n(f"x{j}", f"user.l{i + 1}") for j in range(500)]) for i in range(4)]
    levels.append(_group("user.l4", [_n("m", "test.model")]))
    start = time.perf_counter()
    issues = validate(_wf([_n("top", "user.l0")], [], *levels), REGISTRY)
    assert time.perf_counter() - start < 1.0
    assert [(i.code, i.params.get("reason")) for i in issues] == [(GROUP_CODE, "too_many_nodes")]


def test_wire_fan_out_is_capped() -> None:
    fan = _group(
        "user.fan",
        [_n(f"e{i}", "test.eval") for i in range(400)],
        inputs={"model": [[f"e{i}", "model"] for i in range(400)]},
    )
    nodes = [_n("m", "test.model"), _n("f0", "user.fan")]
    assert _reasons(_wf(nodes, [_e("m", "model", "f0", "model")], fan)) == []  # 400 wires
    edges = [_e("m", "model", "f0", "model")] * 6  # 6 x 400 generated wires > MAX_EDGES
    assert _reasons(_wf(nodes, edges, fan)) == ["too_many_edges"]


def test_the_executor_runs_the_expanded_graph(tmp_path: Path) -> None:
    """backend (a): validate checks the file as written; the executor runs the inlined nodes (ids <group>__<inner>)."""
    from vlm_pipeline.graph.executor import Executor
    from vlm_pipeline.graph.testing import FakeWorker

    run = Executor(REGISTRY, FakeWorker(), tmp_path).run(_finetune(steps=20))
    assert set(run.status) == {"d", "m", "ft__t", "ft__e"}
    assert set(run.status.values()) == {"completed"}
