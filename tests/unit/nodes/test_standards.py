"""P4-2 standards loader and validate-phase checks (configs/standards/*.yaml -> STD-001 issues)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.spec import Registry
from vlm_pipeline.nodes.standards import STANDARDS_DIR, StandardsError, load_standards, standards_rule

SPECS = REGISTRY.all()


def _rule(**override: Any) -> dict[str, Any]:
    rule = {
        "id": "TEST-LORA-RANK",
        "applies_to": {"node": "train.lora_vlm", "param": "lora_dim"},
        "check": {"kind": "range", "min": 16, "max": 64},
        "level": "info",
        "message": "standards.train.lora_rank",
        "source": "LoRA paper",
    }
    return {**rule, **override}


def _write(root: Path, *rules: dict[str, Any], name: str = "t.yaml") -> Path:
    (root / STANDARDS_DIR).mkdir(parents=True, exist_ok=True)
    path = root / STANDARDS_DIR / name
    path.write_text(yaml.safe_dump({"schema_version": 1, "rules": list(rules)}), encoding="utf-8")
    return root


def _issues(tmp_path: Path, workflow: dict[str, Any], *rules: dict[str, Any]) -> list[Any]:
    loaded = load_standards(_write(tmp_path, *rules), SPECS)
    registry = Registry(SPECS, rules=(standards_rule(loaded),), allow_unpinned=True)
    return [i for i in validate(workflow, registry) if i.code == "STD-001"]


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def _lora(max_samples: int = 0, **train: Any) -> dict[str, Any]:
    nodes = [
        _n("m", "model.hf_pick", model="qwen3.5-0.8b"),
        _n("d", "data.hf_load", max_samples=max_samples),
        _n("t", "train.lora_vlm", **train),
    ]
    edges = [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data")]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


# ------------------------------------------------------------------------------------------------ loading


def test_no_standards_dir_means_no_rules(tmp_path: Path) -> None:
    assert load_standards(tmp_path, SPECS) == ()


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"check": {"kind": "range", "min": 16, "warn_below": 3}}, "warn_below"),  # one threshold, typo-proof
        ({"check": {"kind": "catalog_supports_quant"}}, "kind"),  # kinds without code are refused
        ({"check": {"kind": "range"}}, "min and/or max"),
        ({"level": "hint"}, "level"),  # hint is spelled info
        ({"message": "error.CFG-001.what"}, "message"),  # only the standards.* namespace
        ({"id": "lowercase-id"}, "id"),
        ({"source": ""}, "source"),
        ({"applies_to": {"node": "train.nope", "param": "lora_dim"}}, "unknown node"),
        ({"applies_to": {"node": "train.lora_vlm", "param": "learning_rate"}}, "no param"),  # a property, not a param
        ({"applies_to": {"node": "train.lora_vlm", "port": "eval_data"}}, "no input port"),
        ({"check": {"kind": "ratio_to_param", "other": "rank", "min": 1, "max": 4}}, "no param"),
        ({"source": "see .team/proposals/does-not-exist.md"}, "does not exist"),
    ],
)
def test_bad_rules_fail_loading(tmp_path: Path, override: dict[str, Any], needle: str) -> None:
    with pytest.raises(StandardsError, match=needle):
        load_standards(_write(tmp_path, _rule(**override)), SPECS)


def test_rule_ids_are_unique_across_files(tmp_path: Path) -> None:
    _write(tmp_path, _rule(), name="a.yaml")
    with pytest.raises(StandardsError, match="duplicate"):
        load_standards(_write(tmp_path, _rule(), name="b.yaml"), SPECS)


def test_existing_source_path_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs" / "adr" / "ADR-x.md").write_text("x", encoding="utf-8")
    [rule] = load_standards(_write(tmp_path, _rule(source="docs/adr/ADR-x.md §2")), SPECS)
    assert rule.placeholders() >= {"rule", "node", "value", "min", "max"}


# ------------------------------------------------------------------------------------------------ checks


def test_range_fires_with_code_level_and_params(tmp_path: Path) -> None:
    [issue] = _issues(tmp_path, _lora(lora_dim=8), _rule())
    assert (issue.severity, issue.node_id, issue.param) == ("info", "t", "lora_dim")
    assert issue.params["rule"] == "TEST-LORA-RANK"
    assert issue.params["message"] == "standards.train.lora_rank"
    assert (issue.params["value"], issue.params["min"], issue.params["max"]) == (8, 16, 64)
    assert _issues(tmp_path, _lora(lora_dim=16), _rule()) == []


def test_ratio_to_param(tmp_path: Path) -> None:
    rule = _rule(
        applies_to={"node": "train.lora_vlm", "param": "lora_alpha"},
        check={"kind": "ratio_to_param", "other": "lora_dim", "min": 1.0, "max": 2.0},
    )
    [issue] = _issues(tmp_path, _lora(lora_dim=8, lora_alpha=32), rule)
    assert issue.params["value"] == 4.0
    assert _issues(tmp_path, _lora(lora_dim=16, lora_alpha=32), rule) == []


def test_catalog_field_checks_follow_the_model_wire(tmp_path: Path) -> None:
    on_quant = {"node": "quantize.modelopt_ptq", "param": "format"}
    wf = {
        "format": "vlm-workflow",
        "schema_version": 1,
        "name": "q",
        "nodes": [_n("m", "model.hf_pick", model="nemotron-parse-v1.1"), _n("q", "quantize.modelopt_ptq")],
        "edges": [_e("m", "model", "q", "model")],
    }
    experimental = _rule(
        id="TEST-EXPERIMENTAL",
        level="warn",
        applies_to=on_quant,
        check={"kind": "catalog_field_not_in", "catalog_field": "quant_fp8", "values": ["experimental"]},
    )
    supported = _rule(
        id="TEST-SUPPORTED",
        level="error",
        applies_to=on_quant,
        check={"kind": "catalog_field_in", "catalog_field": "quant_fp8", "allowed": ["official"]},
    )
    issues = _issues(tmp_path, wf, experimental, supported)  # Parse v1.1: quant_fp8 experimental
    assert sorted((i.params["rule"], i.severity) for i in issues) == [
        ("TEST-EXPERIMENTAL", "warn"),
        ("TEST-SUPPORTED", "error"),
    ]


def test_dataset_size_checks_use_the_catalog_and_max_samples(tmp_path: Path) -> None:
    small = _rule(
        id="TEST-SMALL",
        level="warn",
        applies_to={"node": ["data.hf_load"]},
        check={"kind": "sample_count", "below": 200},
    )
    epochs = _rule(
        id="TEST-EPOCHS",
        level="warn",
        applies_to={"node": ["train.lora_vlm"], "param": "epochs"},
        check={"kind": "epochs_vs_samples", "max_epochs": 1, "small_data_below": 1000},
    )
    assert {i.params["rule"] for i in _issues(tmp_path, _lora(max_samples=100, epochs=3), small, epochs)} == {
        "TEST-SMALL",
        "TEST-EPOCHS",
    }
    assert _issues(tmp_path, _lora(max_samples=0, epochs=3), small) == []  # CORD train = 800 >= 200


def test_unwired_val_port_is_fine_when_the_dataset_has_a_validation_split(tmp_path: Path) -> None:
    rule = _rule(
        id="TEST-VAL",
        level="warn",
        applies_to={"node": ["train.lora_vlm"], "port": "val_data"},
        check={"kind": "wired_or_dataset_has_split", "split": "validation"},
    )
    assert _issues(tmp_path, _lora(), rule) == []
    rule["check"]["split"] = "dev"
    [issue] = _issues(tmp_path, _lora(), rule)
    assert issue.port == "val_data"


def test_data_profile_kinds_load_but_never_run_at_validation(tmp_path: Path) -> None:
    rule = _rule(
        id="TEST-DUP",
        level="warn",
        applies_to={"node": ["data.hf_load"]},
        check={"kind": "dup_rate", "method": "sha256_image_text", "above": 0.0},
    )
    [loaded] = load_standards(_write(tmp_path, rule), SPECS)
    assert loaded.phase == "data_profile"
    assert _issues(tmp_path, _lora(), rule) == []


def test_unknown_origin_is_not_guessed(tmp_path: Path) -> None:
    wf = _lora(lora_dim=8)
    wf["edges"] = []  # no model, no data: catalog/dataset checks cannot be evaluated -> no STD-001 from them
    rule = _rule(
        id="TEST-SMALL",
        applies_to={"node": ["train.lora_vlm"], "port": "train_data"},
        check={"kind": "sample_count", "below": 10_000},
    )
    assert _issues(tmp_path, wf, rule) == []


# ------------------------------------------------------------------------------------------------ repo files


def test_repo_standards_load_and_every_message_has_vi_and_en_text() -> None:
    from vlm_pipeline.common.paths import project_root
    from vlm_pipeline.i18n import load, placeholders

    rules = load_standards(project_root(), SPECS)
    for locale in ("vi", "en"):
        texts = load(locale)
        missing = sorted({r.message for r in rules if r.message not in texts})
        assert missing == [], f"{locale}: add {missing}"
        for rule in rules:
            extra = placeholders(texts[rule.message]) - rule.placeholders()
            assert not extra, f"{locale} {rule.message} ({rule.id}) uses {sorted(extra)}: not in the rule's params"


def test_every_check_field_is_a_declared_std001_param() -> None:
    import typing

    from vlm_pipeline.common.error_catalog import load_catalog as load_error_catalog
    from vlm_pipeline.nodes.standards import STANDARD_CODE, Rule

    entry = load_error_catalog()[STANDARD_CODE]
    declared = {p.name for p in entry.params}
    kinds = typing.get_args(Rule.model_fields["check"].annotation)  # pydantic keeps the bare union of check models
    assert len(kinds) >= 10, kinds  # guard: an empty set would make the assertion below vacuous
    fields = {name for kind in kinds for name in kind.model_fields if name != "kind"}
    assert fields | {"rule", "message", "node", "param", "value"} <= declared, sorted(fields - declared)
