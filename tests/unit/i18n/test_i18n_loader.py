from __future__ import annotations

import logging

import pytest

from vlm_pipeline import i18n


@pytest.mark.parametrize("locale", i18n.LOCALES)
def test_load_is_flat_strings(locale: str) -> None:
    flat = i18n.load(locale)
    assert flat
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in flat.items())


def test_node_type_id_with_dots_is_one_key() -> None:
    flat = i18n.load("vi")
    node_keys = [k for k in flat if k.startswith("node.")]
    assert node_keys
    for key in node_keys:
        parts = i18n.split_key(key)
        assert parts[0] == "node"
        assert parts[-1] in {"title", "desc", "label", "help"} or parts[-2] == "port"


def test_split_key() -> None:
    assert i18n.split_key("node.train.lora_vlm.title") == ("node", "train.lora_vlm", "title")
    assert i18n.split_key("node.model.hf_pick.param.repo.label") == ("node", "model.hf_pick", "param", "repo", "label")
    assert i18n.split_key("node.x.y.port.out") == ("node", "x.y", "port", "out")
    assert i18n.split_key("error.CFG-009.what.migratable") == ("error", "CFG-009", "what", "migratable")


def test_missing_key_returns_key(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert i18n.t("no.such.key") == "no.such.key"
    assert "missing key" in caplog.text


def test_render_and_missing_param_kept(caplog: pytest.LogCaptureFixture) -> None:
    flat = i18n.load("en")
    key, text = next((k, v) for k, v in flat.items() if i18n.placeholders(v))
    names = sorted(i18n.placeholders(text))
    full = i18n.t(key, "en", **dict.fromkeys(names, "X"))
    assert "{" not in full
    with caplog.at_level(logging.WARNING):
        partial = i18n.t(key, "en")
    assert "{" + names[0] + "}" in partial
    assert "missing param" in caplog.text


def test_unsupported_locale() -> None:
    with pytest.raises(ValueError, match="unsupported locale"):
        i18n.load("fr")
