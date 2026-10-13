"""Unit tests for the error catalog loader (text checks live in tests/unit/test_error_catalog_spec.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from vlm_pipeline.common import error_catalog
from vlm_pipeline.common.error_catalog import (
    CatalogError,
    Entry,
    NodeContext,
    PatchOp,
    Range,
    already_optimized,
    apply_patch,
    apply_patches,
    available_actions,
    load_catalog,
)
from vlm_pipeline.common.errors import ExitCode, InsufficientMemoryError


def test_catalog_loads_and_has_fallback() -> None:
    catalog = load_catalog()
    assert "SYS-000" in catalog
    assert len({e.slug for e in catalog.values()}) == len(catalog)


def test_every_error_has_an_action_and_destructive_ops_never_patch() -> None:
    for code, entry in load_catalog().items():
        if entry.severity == "error":
            assert entry.actions, code
        for action in entry.actions:
            if action.id in {"free_space", "review_to_install", "show_admin_command"}:
                assert action.kind == "navigate", (code, action.id)


def test_resume_checkpoint_always_guarded_by_resumable() -> None:
    for code, entry in load_catalog().items():
        for action in entry.actions:
            if action.id in {"resume_checkpoint", "resume_lighter"}:
                assert action.when == {"resumable": True}, code


@pytest.mark.parametrize("code", ["MEM-003", "SYS-003", "SYS-000", "MEM-002", "DISK-001", "NET-001"])
def test_resume_offered_iff_resumable(code: str) -> None:
    yes = available_actions(SimpleNamespace(code=code, params={}, resumable=True))
    no = available_actions(SimpleNamespace(code=code, params={}, resumable=False))
    assert "resume_checkpoint" in yes
    assert "resume_checkpoint" not in no


def test_missing_resumable_defaults_to_false() -> None:
    assert "resume_checkpoint" not in available_actions(SimpleNamespace(code="SYS-000", params={}))


@pytest.mark.parametrize(
    ("params", "offered"),
    [({"reason": "not_installed"}, True), ({"reason": "not_allowed"}, False)],
)
def test_cfg004_install_only_when_not_installed(params: dict[str, str], offered: bool) -> None:
    assert ("review_to_install" in available_actions(SimpleNamespace(code="CFG-004", params=params))) is offered


def test_unknown_code_falls_back_to_sys000_actions() -> None:
    assert available_actions(SimpleNamespace(code="NOPE-999", params={})) == available_actions(
        SimpleNamespace(code="SYS-000", params={})
    )


def test_exception_carries_code_and_params_only() -> None:
    err = InsufficientMemoryError(params={"need_gb": 52.0, "avail_gb": 43.2})
    assert err.code == "MEM-001"
    assert err.code in load_catalog()
    assert err.params == {"need_gb": 52.0, "avail_gb": 43.2}


def test_exit_codes_are_stable() -> None:
    assert [int(c) for c in ExitCode] == [0, 2, 3, 4, 75]


def test_typo_in_entry_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="prevent_in_prefight"):
        Entry.model_validate(
            {"code": "X-001", "slug": "XX", "severity": "info", "prevent_in_preflight": True, "prevent_in_prefight": 1}
        )


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ({"severity": "error", "actions": []}, "at least one action"),
        (
            {"severity": "info", "actions": [{"id": "a", "kind": "patch", "confirm": False, "when": {"zz": True}}]},
            "undeclared",
        ),
        ({"severity": "info", "actions": [{"id": "a", "kind": "delete", "confirm": True}]}, "kind"),
        ({"severity": "info", "variant_by": "nope"}, "variant_by"),
        ({"severity": "info", "params": [{"name": "r", "type": "enum"}]}, "needs values"),
    ],
)
def test_cross_field_rules(body: dict[str, object], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        Entry.model_validate({"code": "X-001", "slug": "XX", "prevent_in_preflight": False, **body})


def test_malformed_yaml_raises_catalog_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(error_catalog.yaml, "safe_load", lambda _text: {"MEM-001": {"slug": "X", "severity": "boom"}})
    error_catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(CatalogError, match="MEM-001"):
            error_catalog.load_catalog()
    finally:
        error_catalog.load_catalog.cache_clear()


# ------------------------------------------------- applicability against the failing node (bao-design request)


class _ParseParams(BaseModel):
    """Shape of the Nemotron-Parse v1.1 NodeSpec (default template = full fine-tune; PEFT untested for Parse)."""

    method: Literal["full"] = "full"
    micro_batch_size: int = Field(1, ge=1)
    gradient_checkpointing: bool = True
    max_seq_len: int = Field(9000, ge=9000, le=9000)  # fixed 2048x1648 images -> fixed token budget


PARSE_TEMPLATE = NodeContext.from_schema(
    {"method": "full", "micro_batch_size": 1, "gradient_checkpointing": True, "max_seq_len": 9000},
    _ParseParams.model_json_schema(),
)


@pytest.mark.parametrize("code", ["MEM-001", "MEM-003"])
def test_parse_template_only_offers_waiting_options(code: str) -> None:
    err = SimpleNamespace(code=code, params={})
    actions = set(available_actions(err, PARSE_TEMPLATE))
    assert {"queue_until_memory", "see_memory_usage"} <= actions
    for useless in ("use_lora", "use_qlora", "reduce_batch", "reduce_image_size", "enable_grad_checkpointing"):
        assert useless not in actions, useless
    assert already_optimized(err, PARSE_TEMPLATE)


def test_lora_offered_only_when_nodespec_allows_it() -> None:
    class Params(BaseModel):
        method: Literal["full", "lora", "qlora"] = "full"

    node = NodeContext.from_schema({"method": "full"}, Params.model_json_schema())
    actions = available_actions(SimpleNamespace(code="MEM-001", params={}), node)
    assert {"use_lora", "use_qlora"} <= set(actions)
    assert not already_optimized(SimpleNamespace(code="MEM-001", params={}), node)


def test_scaling_respects_int_floor_and_bounds() -> None:
    err = SimpleNamespace(code="MEM-003", params={})
    big = NodeContext(params={"micro_batch_size": 4, "max_seq_len": 8192}, ranges={"micro_batch_size": Range(1)})
    assert {"reduce_batch", "reduce_max_length"} <= set(available_actions(err, big))
    one = NodeContext(params={"micro_batch_size": 1}, ranges={"micro_batch_size": Range(1)})
    assert "reduce_batch" not in available_actions(err, one)  # floor(0.5) = 0 < min 1


def test_without_node_context_nothing_is_filtered_by_patch() -> None:
    err = SimpleNamespace(code="MEM-001", params={})
    assert "use_lora" in available_actions(err)
    assert not already_optimized(err, None)


def test_codes_without_memory_actions_are_never_already_optimized() -> None:
    assert not already_optimized(SimpleNamespace(code="CFG-002", params={}), NodeContext(params={}))


def test_scale_never_takes_positive_int_below_one_even_without_bounds() -> None:
    # bao-cleancode repro: batch=1, no bounds passed -> must NOT offer "reduce batch" to 0
    err = SimpleNamespace(code="MEM-003", params={})
    assert "reduce_batch" not in available_actions(err, NodeContext(params={"micro_batch_size": 1}))
    op = PatchOp(param="b", op="scale", value=0.5)
    assert apply_patch(op, 1) == 1
    assert apply_patch(op, 3) == 1
    assert apply_patch(op, 8) == 4
    assert apply_patch(op, 0.8) == pytest.approx(0.4)


def test_bounds_come_from_pydantic_schema() -> None:
    class Params(BaseModel):
        micro_batch_size: int = Field(4, ge=2)
        max_seq_len: int = Field(4096, gt=1024, le=8192)

    node = NodeContext.from_schema({"micro_batch_size": 2, "max_seq_len": 2048}, Params.model_json_schema())
    assert node.ranges == {
        "micro_batch_size": Range(2),
        "max_seq_len": Range(1024, 8192, low_exclusive=True),
    }
    actions = available_actions(SimpleNamespace(code="MEM-003", params={}), node)
    assert "reduce_batch" not in actions  # 2 -> 1 < ge=2
    assert "reduce_max_length" not in actions  # 2048 -> 1024, not > 1024


def test_apply_patches_matches_what_was_offered() -> None:
    action = next(a for a in load_catalog()["MEM-003"].actions if a.id == "reduce_batch")
    assert apply_patches(action, {"micro_batch_size": 8, "other": "x"}) == {"micro_batch_size": 4, "other": "x"}


class _OptionalImageParams(BaseModel):
    max_pixels: int | None = Field(1024, ge=256)  # None = model default
    micro_batch_size: int = Field(4, ge=1)


@pytest.mark.parametrize("build", ["model", "schema"])
def test_optional_field_constraints_are_respected(build: str) -> None:
    # bao-cleancode repro: Optional[int] with ge=256 -> anyOf in JSON schema; 300 * 0.5 = 150 violates ge
    params = {"max_pixels": 300, "micro_batch_size": 4}
    node = (
        NodeContext.from_model(params, _OptionalImageParams)
        if build == "model"
        else NodeContext.from_schema(params, _OptionalImageParams.model_json_schema())
    )
    actions = available_actions(SimpleNamespace(code="MEM-001", params={}), node)
    assert "reduce_image_size" not in actions
    assert "reduce_batch" in actions


def test_model_validation_catches_custom_validators() -> None:
    from pydantic import field_validator

    class Params(BaseModel):
        micro_batch_size: int = 8

        @field_validator("micro_batch_size")
        @classmethod
        def _even(cls, v: int) -> int:
            if v % 2:
                raise ValueError("must be even")
            return v

    ok = NodeContext.from_model({"micro_batch_size": 8}, Params)  # 8 -> 4, even
    odd = NodeContext.from_model({"micro_batch_size": 6}, Params)  # 6 -> 3, rejected by the validator
    err = SimpleNamespace(code="MEM-003", params={})
    assert "reduce_batch" in available_actions(err, ok)
    assert "reduce_batch" not in available_actions(err, odd)


# ------------------------------------------------------- display labels for no-tech users (bao-frontend request)


def _labels() -> object:
    from vlm_pipeline.common.error_catalog import Labels

    nodes = {"n3": "Fine-tune nhẹ (LoRA)", "n1": "Dữ liệu CORD-v2"}
    ports = {("n3", "base_model"): "Model gốc"}
    params = {("n3", "max_steps"): "Số bước huấn luyện"}
    types = {"Dataset": "Dữ liệu", "Model": "Model"}
    return Labels(
        node=nodes.get,
        port=lambda n, p: ports.get((n, p)),
        param=lambda n, p: params.get((n, p)),
        port_type=types.get,
    )


def test_cfg003_shows_node_title_and_port_label_not_raw_ids() -> None:
    from vlm_pipeline.common.error_catalog import display_params

    shown = display_params("CFG-003", {"node": "n3", "port": "base_model"}, _labels(), node_id="n3")  # type: ignore[arg-type]
    assert shown == {"node": "Fine-tune nhẹ (LoRA)", "port": "Model gốc"}


def test_cfg001_and_cfg002_and_cfg005_labels() -> None:
    from vlm_pipeline.common.error_catalog import display_params

    labels = _labels()
    cfg001 = display_params(
        "CFG-001",
        {"a": "n1", "b": "n3", "type_a": "Dataset", "type_b": "Model"},
        labels,
        node_id="n3",  # type: ignore[arg-type]
    )
    assert cfg001 == {"a": "Dữ liệu CORD-v2", "b": "Fine-tune nhẹ (LoRA)", "type_a": "Dữ liệu", "type_b": "Model"}
    assert display_params("CFG-002", {"nodes": "n1, n3"}, labels, node_id="n1") == {  # type: ignore[arg-type]
        "nodes": "Dữ liệu CORD-v2, Fine-tune nhẹ (LoRA)"
    }
    cfg005 = display_params(
        "CFG-005", {"node": "n3", "param": "max_steps", "rule": "greater_than_equal"}, labels, node_id="n3"
    )  # type: ignore[arg-type]
    assert cfg005["param"] == "Số bước huấn luyện"
    assert cfg005["rule"] == "greater_than_equal"  # non-ref params untouched


def test_unknown_ids_fall_back_to_raw_and_input_is_not_mutated() -> None:
    from vlm_pipeline.common.error_catalog import display_params

    raw = {"node": "ghost", "port": "x"}
    shown = display_params("CFG-003", raw, _labels(), node_id="ghost")  # type: ignore[arg-type]
    assert shown == raw
    assert raw == {"node": "ghost", "port": "x"}


@pytest.mark.parametrize(
    ("code", "params", "key"),
    [
        ("CFG-009", {"migratable": True}, "migratable"),
        ("CFG-009", {"migratable": False}, "not_migratable"),
        ("CFG-004", {"reason": "not_installed", "trust_level": "unverified"}, "not_installed_unknown"),
        ("CFG-010", {"supported": "train.sft_vlm"}, "full_finetune_only"),
        ("CFG-001", {"a": "x"}, None),
    ],
)
def test_variant_key_picks_the_i18n_child(code: str, params: dict[str, object], key: str | None) -> None:
    from vlm_pipeline.common.error_catalog import variant_key
    from vlm_pipeline.i18n import load

    assert variant_key(code, params) == key
    if key is not None:
        assert f"error.{code}.what.{key}" in load("vi")


def test_variant_key_coerces_bool_strings_and_never_leaves_a_raw_key() -> None:
    from vlm_pipeline.common.error_catalog import variant_key

    assert variant_key("CFG-009", {"migratable": "true"}) == "migratable"
    assert variant_key("CFG-009", {"migratable": "false"}) == "not_migratable"
    assert variant_key("CFG-010", {"supported": "train.unknown"}) == "full_finetune_only"  # first variant + warning
