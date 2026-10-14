"""ADR-021 / P021 H1: a user-added repo -> family -> CatalogEntry + template, from its JSON files only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.catalog import Catalog, load_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.download import resolver as r
from vlm_pipeline.train.config import LoraParams
from vlm_pipeline.train.recipe import build_recipe, load_template, qlora_skip_modules

SHA = "a" * 40
QWEN = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "torch_dtype": "bfloat16",
        "text_config": {"hidden_size": 2048}, "vision_config": {"model_type": "qwen3_5_vision"}}  # fmt: skip


def _family() -> r.Family:
    family = r.load_families().for_architecture("Qwen3_5ForConditionalGeneration")
    assert family is not None
    return family


def test_research_families_load_and_map_qwen() -> None:
    family = _family()
    assert family.id == "qwen3_5"
    assert family.auto_class == "AutoModelForImageTextToText"


# ------------------------------------------------------------------------------------------- remote code (security §4)
@pytest.mark.parametrize(
    ("configs", "files", "sign"),
    [
        ({"config.json": {**QWEN, "auto_map": {"AutoModel": "modeling_x.X"}}}, [], "auto_map:config.json"),
        ({"config.json": {**QWEN, "text_config": {"auto_map": {"AutoConfig": "c.C"}}}}, [], "auto_map:config.json"),
        ({"config.json": QWEN, "tokenizer_config.json": {"auto_map": {"AutoTokenizer": ["t.T", None]}}}, [],
         "auto_map:tokenizer_config.json"),
        ({"config.json": QWEN, "processor_config.json": {"processor_class": "org/repo--proc.P"}}, [],
         "code_ref:processor_config.json"),
        ({"config.json": {**QWEN, "trust_remote_code": False}}, [], "trust_remote_code:config.json"),
        ({"config.json": QWEN}, ["config.json", "sub/modeling_x.py"], "py_file"),
        ({"config.json": QWEN}, ["config.json", "Modeling.PY"], "py_file"),
        ({"config.json": QWEN}, ["config.json", "__pycache__/x.cpython-312.pyc"], "py_file"),
        ({"config.json": QWEN}, ["x.pyw"], "py_file"),
    ],
)  # fmt: skip
def test_every_static_remote_code_sign_blocks(configs: dict[str, Any], files: list[str], sign: str) -> None:
    assert sign in r.remote_code_signs(configs, files)
    with pytest.raises(r.ModelNeedsRemoteCodeError) as info:
        r.resolve_family(configs, files, repo="org/x", revision=SHA, families=r.load_families())
    assert (info.value.code, info.value.params) == (
        "MODEL-004", {"repo": "org/x", "revision": SHA[:12], "reason": "remote_code"})  # fmt: skip


def test_a_native_repo_has_no_sign_and_resolves() -> None:
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    assert r.remote_code_signs({"config.json": QWEN}, files) == []
    family, facts = r.resolve_family({"config.json": QWEN}, files, repo="org/x", revision=SHA,
                                     families=r.load_families())  # fmt: skip
    assert family.id == "qwen3_5"
    assert (facts.architecture, facts.model_type, facts.torch_dtype) == (
        "Qwen3_5ForConditionalGeneration", "qwen3_5", "bfloat16")  # fmt: skip


def test_remote_code_wins_over_an_unknown_architecture() -> None:
    config = {"architectures": ["InternVLChatModel"], "model_type": "internvl_chat", "auto_map": {"AutoModel": "m.M"}}
    with pytest.raises(r.ModelNeedsRemoteCodeError):
        r.resolve_family({"config.json": config}, [], repo="o/x", revision=SHA, families=r.load_families())


def test_an_unknown_architecture_is_model_003_with_clean_names() -> None:
    config = {"architectures": ["Weird<script>Model" + "x" * 200], "model_type": "we ird"}
    with pytest.raises(r.ModelArchUnsupportedError) as info:
        r.resolve_family({"config.json": config}, [], repo="o/x", revision=SHA, families=r.load_families())
    params = info.value.params
    assert info.value.code == "MODEL-003"
    assert params["architecture"] == ("WeirdscriptModel" + "x" * 200)[: r.MAX_NAME]
    assert params["model_type"] == "weird"


@pytest.mark.parametrize("value", [None, 3, "", "<>", ["a"]])
def test_clean_name_never_returns_an_empty_or_unsafe_string(value: object) -> None:
    assert r.clean_name(value) == "unknown"


# ---------------------------------------------------------------------------------------- reading files (security §5)
@pytest.mark.parametrize(
    "content",
    [
        b'{"a": NaN}',
        b'{"a": Infinity}',
        ("[" * 40 + "]" * 40).encode(),
        json.dumps({"k" * 300: 1}).encode(),
        b"\xff\xfe",
        b"{not json",
    ],
)
def test_untrusted_json_is_refused(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(content)
    with pytest.raises(r.RepoFilesInvalidError):
        r.read_repo_json(path)


def test_a_symlink_or_an_oversized_file_is_refused(tmp_path: Path) -> None:
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"x": "y" * (r.JSON_MAX + 1)}))
    with pytest.raises(r.RepoFilesInvalidError):
        r.read_repo_json(big)
    real = tmp_path / "real.json"
    real.write_text("{}")
    (tmp_path / "link.json").symlink_to(real)
    with pytest.raises(r.RepoFilesInvalidError):
        r.read_repo_json(tmp_path / "link.json")
    assert r.read_repo_json(real) == {}


@pytest.mark.parametrize(
    ("index", "dtype", "params_b"),
    [
        ({"metadata": {"total_size": 4_600_000_000}}, "bfloat16", 2.3),
        ({"metadata": {"total_size": 4_600_000_000}}, "float32", 1.15),
        ({"metadata": {"total_size": True}}, "bfloat16", None),
        ({"metadata": {"total_size": 10**15}}, "bfloat16", None),
        ({"metadata": {}}, "bfloat16", None),
        ([], "bfloat16", None),
    ],
)
def test_params_from_the_safetensors_index_are_bounded(index: object, dtype: str, params_b: float | None) -> None:
    assert r.params_b_from_index(index, dtype) == params_b


# ------------------------------------------------------------------------------------------- the entry
def test_a_user_entry_is_a_catalog_entry_marked_unverified() -> None:
    entry = r.user_entry(repo="HuggingFaceTB/SmolVLM2-2.2B", revision=SHA, family=_family(), params_b=2.2,
                         disk_gb=4.4, license_id="apache-2.0", gated=False)  # fmt: skip
    assert entry.id == "user--huggingfacetb--smolvlm2-2.2b--aaaaaaaa"
    assert (entry.tier, entry.security, entry.ram_source, entry.remote_code) == (
        "user", "unreviewed", "estimated", "none")  # fmt: skip
    assert entry.train_node == "train.lora_vlm"
    assert entry.ram_gb_train == round(1.8 * 2.2 + 25.0, 1)
    assert load_catalog().get(entry.id) is None  # catalog ids never contain "--"


def test_past_the_fitted_sizes_the_ram_bound_gets_a_margin() -> None:
    family = _family()
    assert r.ram_upper_bound(family, 9.0) == round(1.8 * 9 + 25, 1)
    assert r.ram_upper_bound(family, 30.0) == round((1.8 * 30 + 25) * 1.2, 1)


def test_the_catalog_file_can_never_hold_a_user_model() -> None:
    entry = r.user_entry(repo="o/x", revision=SHA, family=_family(), params_b=1.0, disk_gb=2.0, license_id="mit",
                         gated=False)  # fmt: skip
    catalog = load_catalog()
    data = catalog.model_dump()
    data["models"] = [*data["models"], {**entry.model_dump(), "vendor": next(iter(catalog.vendors))}]
    with pytest.raises(ValueError, match="never in the catalog file"):
        Catalog.model_validate(data)


@pytest.mark.parametrize(("repo", "revision"), [("o/x", "main"), ("bad", SHA), ("o/x/y", SHA)])
def test_only_a_pinned_repo_becomes_an_entry(repo: str, revision: str) -> None:
    with pytest.raises(ValueError, match="pinned form"):
        r.user_entry(repo=repo, revision=revision, family=_family(), params_b=1.0, disk_gb=1.0, license_id="mit",
                     gated=False)  # fmt: skip


# ------------------------------------------------------------------------------------------- the template
def test_the_family_template_trains_like_the_catalog_qwen() -> None:
    family = _family()
    template = r.render_template(family)
    catalog_qwen = load_template(project_root() / "configs" / "train" / "templates" / "qwen3.5-2b.yaml")
    assert template == catalog_qwen  # same family recipe: no drift from the vetted template
    assert qlora_skip_modules(template) == ["model.visual"]
    recipe = build_recipe(template, LoraParams(quantize_base="nf4"), model_dir="m", dataset_dir="d", run_dir="r")
    assert recipe["quantization"]["llm_int8_skip_modules"] == ["model.visual"]


@pytest.mark.parametrize(
    "config",
    [
        {**QWEN, "model_type": "llava"},  # native class name, but transformers builds from model_type
        {**QWEN, "architectures": ["Qwen3_5ForConditionalGeneration", "LlavaForConditionalGeneration"]},
        {**QWEN, "architectures": []},
    ],
)
def test_architecture_and_model_type_must_both_be_the_familys(config: dict[str, Any]) -> None:
    with pytest.raises(r.ModelArchUnsupportedError):
        r.resolve_family({"config.json": config}, [], repo="o/x", revision=SHA, families=r.load_families())


def _family_dict(**changes: Any) -> dict[str, Any]:
    return {**_family().model_dump(), **changes}


@pytest.mark.parametrize(
    ("changes", "ok"),
    [
        ({}, True),
        ({"image_policy": "processor_default", "min_pixels": None, "max_pixels": None}, True),
        ({"image_policy": "processor_default"}, False),  # caps of a Qwen recipe on another processor: refused
        ({"min_pixels": None}, False),
        ({"min_pixels": 10, "max_pixels": 5}, False),
        ({"vision_prefixes": ()}, False),
        ({"min_image": "latest"}, False),
    ],
)
def test_family_fields_follow_the_image_policy(changes: dict[str, Any], ok: bool) -> None:
    if ok:
        r.Family.model_validate(_family_dict(**changes))
        return
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic's ValidationError, several rules
        r.Family.model_validate(_family_dict(**changes))


def test_a_processor_default_family_leaves_pixels_to_the_repo() -> None:
    family = r.Family.model_validate(_family_dict(image_policy="processor_default", min_pixels=None,
                                                  max_pixels=None, min_image="vlm-automodel:26.08-8"))  # fmt: skip
    template = r.render_template(family)
    assert "min_pixels" not in template["processor"]
    assert "max_pixels" not in template["processor"]
    entry = r.user_entry(repo="o/x", revision=SHA, family=family, params_b=1.0, disk_gb=1.0, license_id="mit",
                         gated=False)  # fmt: skip
    assert (entry.image_policy, entry.min_pixels, entry.min_image) == ("processor_default", 0, "vlm-automodel:26.08-8")


# ------------------------------------------------------------------------------------------- module blocks (catalog)
def test_a_catalog_family_must_exist_and_blocks_resolve_from_it() -> None:
    known = {f.id for f in r.load_families().families}
    for entry in load_catalog().models:
        assert entry.family is None or entry.family in known, entry.id
        if entry.family is not None:
            assert r.block_prefixes(entry) is not None


def test_every_trainable_catalog_model_names_its_blocks() -> None:
    missing = [e.id for e in load_catalog().models if e.train_node != "none" and r.block_prefixes(e) is None]
    assert missing == []


@pytest.mark.parametrize(
    ("changes", "ok"),
    [
        ({"family": "qwen3_5"}, True),
        ({"vision_prefixes": ("encoder.model_encoder",), "projector_prefixes": ("encoder.conv1",)}, True),
        ({"family": "qwen3_5", "vision_prefixes": ("a",), "projector_prefixes": ("b",)}, False),
        ({"vision_prefixes": ("a",)}, False),
        ({"vision_prefixes": ("",), "projector_prefixes": ("b",)}, False),
    ],
)
def test_blocks_come_one_way_only(changes: dict[str, Any], ok: bool) -> None:
    from vlm_pipeline.common.catalog import CatalogEntry

    base = load_catalog().get("qwen3.5-2b")
    assert base is not None
    data = {**base.model_dump(), "family": None, **changes}  # the catalog entry names its family now
    if ok:
        entry = CatalogEntry.model_validate(data)
        expected = ((("model.visual",), ("model.visual.merger",)) if "family" in changes
                    else (changes["vision_prefixes"], changes["projector_prefixes"]))  # fmt: skip
        assert r.block_prefixes(entry) == expected
        return
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic's ValidationError, several rules
        CatalogEntry.model_validate(data)


def test_a_user_entry_carries_its_family() -> None:
    entry = r.user_entry(repo="o/x", revision=SHA, family=_family(), params_b=1.0, disk_gb=1.0, license_id="mit",
                         gated=False)  # fmt: skip
    assert entry.family == "qwen3_5"
    assert r.block_prefixes(entry) == (("model.visual",), ("model.visual.merger",))


@pytest.mark.parametrize(
    ("repo", "config", "split", "expected"),
    [
        ("lmms-lab/DocVQA", "DocVQA", "validation", "user--ds--lmms-lab--docvqa--aaaaaaaa--docvqa--validation"),
        ("Org/My_Set", "default", "train", "user--ds--org--my-set--aaaaaaaa--default--train"),
        ("o/x", "..", "a b", "user--ds--o--x--aaaaaaaa--x--a-b"),
    ],
)
def test_user_dataset_ids_are_lowercase_and_path_safe(repo: str, config: str, split: str, expected: str) -> None:
    import re

    got = r.user_dataset_id(repo, SHA, config, split)
    assert got == expected
    assert got.startswith(r.user_dataset_id_prefix(repo, SHA))
    assert re.fullmatch(r"user--ds--[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*", got)
    assert ".." not in got
    assert "/" not in got
