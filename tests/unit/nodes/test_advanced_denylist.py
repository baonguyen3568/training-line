"""Gate over the COMMITTED engine options (configs/engine/*.json, P016 §3): the denylist holds after any edit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.nodes.advanced_denylist import denial, exception_problems

ENGINE_FILES = sorted((project_root() / "configs" / "engine").glob("*.json"))


def test_there_is_an_options_file_for_the_pinned_train_image() -> None:
    tag = DEFAULT_TRAIN_IMAGE.rsplit(":", 1)[-1]
    assert project_root() / "configs" / "engine" / f"automodel-{tag}.json" in ENGINE_FILES, "run engine_schema_write"


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda p: p.name)
def test_every_committed_key_passes_the_denylist(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [entry["key"] for entry in data["keys"]]
    assert keys == sorted(set(keys)), "sorted and unique (a regeneration diff must be readable)"
    denied = {entry["key"]: denial(entry["key"], entry["type"]) for entry in data["keys"]}
    assert {k: r for k, r in denied.items() if r is not None} == {}
    assert all(entry["templates"] for entry in data["keys"]), "every key belongs to at least one train template"
    unbounded = [
        e["key"]
        for e in data["keys"]
        if e["type"] in ("int", "float") and (e.get("min") is None or e.get("max") is None)
    ]
    assert unbounded == [], "every offered number needs a range in advanced_denylist.RANGES (shared-host DoS)"
    assert all(
        e["type"] != "list" or e["default"] is None or all(not isinstance(v, str) for v in e["default"])
        for e in data["keys"]
    )


def test_exceptions_are_approved_and_never_cover_code_paths_or_secrets() -> None:
    assert exception_problems() == []


@pytest.mark.parametrize(
    ("key", "kind", "reason"),
    [
        ("model._target_", "str", "runs_code"),
        ("dataloader.collate_fn._post_tokenize_hook", "other", "runs_code"),
        ("model.trust_remote_code", "bool", "runs_code"),
        ("model.pretrained_model_name_or_path", "str", "node_controlled"),
        ("tokenizer.cache_dir", "bool", "denied_word"),  # words, not substrings: "cache", "dir"
        ("hub.revision", "enum", "denied_word"),
        ("vision_config.use_cache", "bool", "denied_word"),  # exceptions are FULL keys, never a word
        ("model.torch_dtype", "other", "type"),
        ("optimizer.lr", "float", "node_controlled"),
        ("distributed.tp_size", "int", "node_controlled"),  # one GB10
        ("model.backend.attn", "enum", None),
        ("optimizer.eps", "float", None),
        ("dataloader.num_workers", "int", None),  # "port" in "support" would be a substring, never a word
    ],
)
def test_denial_reasons(key: str, kind: str, reason: str | None) -> None:
    assert denial(key, kind) == reason


def test_exceptions_are_scoped_to_their_node_types() -> None:
    assert denial("model.text_config.use_cache", "bool", "train.lora_vlm") is None
    assert denial("model.text_config.use_cache", "bool", "quantize.modelopt_ptq") == "denied_word"
