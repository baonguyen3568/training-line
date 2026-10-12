"""Gate: every registered node, port, param and enum option has user-facing text in every locale (P009 §2.2).

Text lives only in src/vlm_pipeline/i18n/{vi,en}.yaml (owner: bao-design); a node added without copy fails here.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import pytest
import yaml

from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.spec import NodeSpec

LOCALES = ("vi", "en")
# Enum params whose option labels come from data, not i18n (model display names from the catalog).
DATA_LABELLED = {("model.hf_pick", "repo"), ("model.hf_pick", "model")}


def _load(locale: str) -> dict[str, Any]:
    text = resources.files("vlm_pipeline.i18n").joinpath(f"{locale}.yaml").read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    return data


def _has(tree: dict[str, Any], *keys: str) -> bool:
    node: Any = tree
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return isinstance(node, str) and bool(node.strip())


def _required_keys(spec: NodeSpec) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = [("node", spec.type_id, "title"), ("node", spec.type_id, "desc")]
    keys += [("node", spec.type_id, "port", port) for port in {**spec.inputs, **spec.outputs}]
    for name, prop in spec.params_schema().get("properties", {}).items():
        keys += [("node", spec.type_id, "param", name, "label"), ("node", spec.type_id, "param", name, "help")]
        if "enum" in prop and (spec.type_id, name) not in DATA_LABELLED:
            option_keys = prop.get("x-ui", {}).get("option_keys")
            assert option_keys, f"{spec.type_id}.{name}: enum param needs x-ui.option_keys"
            keys += [(option_keys, str(value)) for value in prop["enum"]]
    return keys


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("spec", REGISTRY.all(), ids=[s.type_id for s in REGISTRY.all()])
def test_every_registered_text_key_exists(spec: NodeSpec, locale: str) -> None:
    tree = _load(locale)
    missing = [".".join(k) for k in _required_keys(spec) if not _has(tree, *k)]
    assert not missing, f"{locale}: missing {missing}"
