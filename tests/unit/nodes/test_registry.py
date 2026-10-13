"""Registry and node-spec contract tests (M1-1). Every first-party node is checked the same way (P004 §8.4)."""

from __future__ import annotations

import json
from typing import Any

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
        assert set(hints) <= {
            "widget", "order", "preset", "advanced", "option_keys", "summary", "option_meta", "disabled_options",
            "recommended",
        }  # fmt: skip


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


@pytest.mark.parametrize("path", ["", "/etc", "../outside", "data/../../x", "a\\b"])
def test_local_folder_path_must_stay_inside_project(path: str) -> None:
    with pytest.raises(ValidationError):
        REGISTRY.get("data.local_folder").validate_params({"path": path})  # type: ignore[union-attr]


def test_lora_batch_sizes_must_divide() -> None:
    spec = REGISTRY.get("train.lora_vlm")
    assert spec is not None
    with pytest.raises(ValidationError, match="multiple"):
        spec.validate_params({"global_batch_size": 6, "local_batch_size": 4})


def test_local_path_params_are_marked_for_sec_001() -> None:
    spec = REGISTRY.get("data.local_folder")
    assert spec is not None
    assert spec.params_schema()["properties"]["path"]["format"] == "local-path"


def test_model_pick_offers_the_catalog_with_the_default_first_choice() -> None:
    from vlm_pipeline.common.catalog import load_catalog

    spec = REGISTRY.get("model.hf_pick")
    assert spec is not None
    prop = spec.params_schema()["properties"]["model"]
    catalog = load_catalog()
    assert prop["default"] == catalog.default.id
    assert set(prop["enum"]) == {m.id for m in catalog.selectable()}
    assert set(prop["x-ui"]["option_meta"]) == set(prop["enum"])
    with pytest.raises(ValidationError):
        spec.validate_params({"model": "not-in-catalog"})


def _values_as_strings(prop: dict[str, Any]) -> set[str]:
    """Every value a param can take, spelled like a ``disabled_options`` key."""
    if prop["type"] == "boolean":
        return {"true", "false"}
    values = prop.get("enum", [prop["const"]] if "const" in prop else [])
    return {str(v) for v in values}


def _disabled_options_problems(spec: NodeSpec) -> list[str]:
    from vlm_pipeline.i18n import load

    texts = [load("vi"), load("en")]
    problems: list[str] = []
    for name, prop in spec.params_schema().get("properties", {}).items():
        disabled = prop.get("x-ui", {}).get("disabled_options")
        if disabled is None:
            continue
        where = f"{spec.type_id}.{name}"
        if not isinstance(disabled, dict) or not disabled:
            problems.append(f"{where}: disabled_options must be a non-empty mapping")
            continue
        if unknown := sorted(set(disabled) - _values_as_strings(prop)):
            problems.append(f"{where}: {unknown} are not values of the param")
        if str(prop.get("default")).lower() in disabled:
            problems.append(f"{where}: the default value cannot be disabled")
        missing = [r for r in disabled.values() if r is not None and not all(r in t for t in texts)]
        problems += [f"{where}: no vi/en text for {r!r}" for r in missing]
    return problems


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.type_id)
def test_disabled_options_name_real_values_and_real_reasons(spec: NodeSpec) -> None:
    assert _disabled_options_problems(spec) == []


def test_disabled_options_checker_catches_mistakes() -> None:
    from typing import Literal

    from pydantic import Field

    from vlm_pipeline.common.params import Params, ui

    class _P(Params):
        kv: Literal["none", "fp8"] = Field("none", json_schema_extra=ui(disabled_options={"fp8": None}))
        vision: bool = Field(False, json_schema_extra=ui(disabled_options={"true": "no.such.key"}))
        size: Literal[128, 512] = Field(512, json_schema_extra=ui(disabled_options={"512": None, "1024": None}))

    spec = NodeSpec(type_id="test.disabled", version="1.0.0", category="util", icon="x", params_model=_P, runner="host")
    problems = _disabled_options_problems(spec)
    assert not any(".kv:" in p for p in problems)  # a valid use passes
    assert any(".vision: no vi/en text" in p for p in problems)
    assert any(".size: ['1024'] are not values" in p for p in problems)
    assert any(".size: the default value cannot be disabled" in p for p in problems)
