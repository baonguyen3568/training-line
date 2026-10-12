"""Unit tests for vlm_pipeline.train.recipe_checks (CPU, no network; real pinned config.json fixtures)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vlm_pipeline.common.error_catalog import load_catalog
from vlm_pipeline.train.recipe_checks import check_nemotron_parse, resolve_parse_class_token_start_idx

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "model_configs" / "nemotron_parse"
LOSS = "nemo_automodel.components.models.nemotron_parse.nemotron_parse_loss.NemotronParseLoss"

# Token ids verified 2026-10-12 against the real tokenizer.json at the pinned revisions (bao-fixcode).
TOKENS = {
    "v1.1": {49999: "ylethanolamine", 50000: "<output_plain>", 50009: "<x_0.0>", 52313: "<class_Text>"},
    "v2.0": {50000: "<output_plain>", 50011: "<x_0.0>", 52314: "<y_0.9992>", 52315: "<class_Text>"},
}


def _model_dir(version: str, tmp_path: Path, *, with_tokenizer: bool = True) -> Path:
    d = tmp_path / version
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / f"config_{version}.json", d / "config.json")
    if with_tokenizer:
        added = [{"id": i, "content": t, "special": t.startswith("<")} for i, t in TOKENS[version].items()]
        (d / "tokenizer.json").write_text(json.dumps({"added_tokens": added, "model": {"vocab": {}}}))
    return d


def _recipe(idx: int | None) -> dict[str, object]:
    loss: dict[str, object] = {"_target_": LOSS, "coordinate_weight": 10.0}
    if idx is not None:
        loss["class_token_start_idx"] = idx
    return {"loss_fn": loss}


@pytest.mark.parametrize(("version", "expected"), [("v1.1", 50000), ("v2.0", 52315)])
def test_threshold_resolved_from_config(version: str, expected: int, tmp_path: Path) -> None:
    assert resolve_parse_class_token_start_idx(_model_dir(version, tmp_path)) == expected


@pytest.mark.parametrize(("version", "idx"), [("v1.1", 50000), ("v1.1", None), ("v2.0", 52315)])
def test_matching_recipe_passes(version: str, idx: int | None, tmp_path: Path) -> None:
    assert check_nemotron_parse(_model_dir(version, tmp_path), _recipe(idx)) == []


@pytest.mark.parametrize(("version", "idx"), [("v2.0", 50000), ("v2.0", None), ("v1.1", 52315)])
def test_mismatching_recipe_fails(version: str, idx: int | None, tmp_path: Path) -> None:
    # (v2.0, None): the loss's own default 50000 would silently weight the wrong tokens.
    [issue] = check_nemotron_parse(_model_dir(version, tmp_path), _recipe(idx))
    assert issue.code == "CFG-005"
    assert issue.params["param"] == "loss_fn.class_token_start_idx"


def test_threshold_must_hit_a_special_token(tmp_path: Path) -> None:
    d = _model_dir("v1.1", tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    cfg["class_token_start_idx"] = 49999  # 'ylethanolamine': a normal word piece
    (d / "config.json").write_text(json.dumps(cfg))
    issues = check_nemotron_parse(d, _recipe(49999))
    assert [i.params["rule"] for i in issues] == ["must point at a special <...> token in tokenizer.json"]


def test_threshold_must_be_inside_vocab(tmp_path: Path) -> None:
    d = _model_dir("v2.0", tmp_path, with_tokenizer=False)
    cfg = json.loads((d / "config.json").read_text())
    cfg["class_token_start_idx"] = 80000
    (d / "config.json").write_text(json.dumps(cfg))
    issues = check_nemotron_parse(d, _recipe(80000))
    assert any("vocab_size" in str(i.params["rule"]) for i in issues)


def test_recipe_without_parse_loss_is_not_checked(tmp_path: Path) -> None:
    assert check_nemotron_parse(_model_dir("v2.0", tmp_path), {"loss_fn": {"_target_": "x.CrossEntropy"}}) == []
    assert check_nemotron_parse(_model_dir("v2.0", tmp_path / "b"), {}) == []


def test_issue_params_match_the_catalog() -> None:
    declared = {p.name for p in load_catalog()["CFG-005"].params}
    assert {"node", "param", "value", "rule"} <= declared


@pytest.mark.parametrize(
    "yaml_name",
    ["nemotron_parse_v1_1_cordv2_smoke.yaml", "nemotron_parse_2_0_cordv2_smoke.yaml"],
)
def test_committed_smoke_recipes_agree_with_their_checkpoints(yaml_name: str, tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    recipe = yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/train/smoke" / yaml_name).read_text())
    version = "v2.0" if "2_0" in yaml_name else "v1.1"
    assert check_nemotron_parse(_model_dir(version, tmp_path), recipe) == []
