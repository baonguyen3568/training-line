"""Registry and node-spec contract tests (M1-1). Every first-party node is checked the same way (P004 §8.4)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.spec import PORT_TYPES, NodeSpec, PortSpec, Registry, is_compatible

ALL_SPECS = REGISTRY.all()
IDS = [s.type_id for s in ALL_SPECS]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=IDS)
def test_defaults_are_valid(spec: NodeSpec) -> None:
    spec.validate_params({})


@pytest.mark.parametrize("spec", ALL_SPECS, ids=IDS)
def test_unknown_param_is_rejected(spec: NodeSpec) -> None:
    with pytest.raises(ValidationError):
        spec.validate_params({"definitely_not_a_param": 1})


@pytest.mark.parametrize("spec", ALL_SPECS, ids=IDS)
def test_api_item_matches_frontend_contract(spec: NodeSpec) -> None:
    item = spec.to_api()
    assert set(item) == {
        "type_id", "version", "category", "icon", "inputs", "outputs", "params_schema",
        "cacheable", "runner", "origin", "permissions", "policy_block",
    }  # fmt: skip
    json.dumps(item)  # serialisable as-is
    schema = item["params_schema"]
    assert schema["type"] == "object"
    for name, prop in schema.get("properties", {}).items():
        # The editor renders flat JSON Schema properties only (frontend ParamSchema): no anyOf/oneOf/$ref.
        assert "type" in prop, f"{spec.type_id}.{name}: Optional/union params are not renderable"
        hints = prop.get("x-ui")
        assert hints is not None, f"{spec.type_id}.{name} has no x-ui hints"
        assert set(hints) <= {"widget", "order", "preset", "advanced", "option_keys", "summary"}


def test_registry_api_lists_port_types() -> None:
    body = REGISTRY.to_api()
    assert {p["id"] for p in body["port_types"]} == {p.id for p in PORT_TYPES}
    assert len(body["node_types"]) == len(ALL_SPECS)


@pytest.mark.parametrize(
    ("src", "dst", "ok"),
    [
        ("Model", "Model", True),
        ("Checkpoint", "Model", True),
        ("QuantizedModel", "Model", True),
        ("Model", "Checkpoint", False),
        ("Adapter", "Model", False),
        ("Dataset", "Model", False),
    ],
)
def test_port_compatibility(src: str, dst: str, ok: bool) -> None:
    assert is_compatible(src, dst) is ok


class _Loose(BaseModel):
    x: int = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _spec(**kw: object) -> NodeSpec:
    base: dict[str, object] = {"type_id": "test.node", "version": "1.0.0", "category": "util", "icon": "x"}
    return NodeSpec.model_validate({"runner": "host", **base, **kw})


def test_registry_rejects_duplicates_unknown_ports_loose_params_and_unpinned_images() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Registry([_spec(), _spec()])
    with pytest.raises(ValueError, match="unknown type"):
        Registry([_spec(outputs={"x": PortSpec(type="Nope")})])
    with pytest.raises(ValueError, match="forbid"):
        Registry([_spec(params_model=_Loose)])
    with pytest.raises(ValueError, match="pinned"):
        Registry([_spec(params_model=_Strict, runner="container")])


def test_image_must_be_digest_pinned() -> None:
    digest = "sha256:" + "a" * 64
    assert _spec(runner="container", image=f"vlm-automodel@{digest}").image_digest == digest
    with pytest.raises(ValidationError):
        _spec(runner="container", image="vlm-automodel:latest")


@pytest.mark.parametrize("path", ["/etc", "../outside", "data/../../x", "a\\b"])
def test_local_folder_path_must_stay_inside_project(path: str) -> None:
    with pytest.raises(ValidationError):
        REGISTRY.get("data.local_folder").validate_params({"path": path})  # type: ignore[union-attr]


def test_lora_batch_sizes_must_divide() -> None:
    spec = REGISTRY.get("train.lora_vlm")
    assert spec is not None
    with pytest.raises(ValidationError, match="multiple"):
        spec.validate_params({"global_batch_size": 6, "local_batch_size": 4})
