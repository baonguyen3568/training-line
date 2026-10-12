from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_pipeline.common.decoding import DecodeOverrides, decode_overrides_of
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.recipe import TEMPLATE_DIR, RecipeError, load_template


def test_absent_or_null_is_empty() -> None:
    assert decode_overrides_of({}).as_dict() == {}
    assert decode_overrides_of({"decode_overrides": None}).as_dict() == {}


def test_as_dict_keeps_only_set_keys() -> None:
    assert decode_overrides_of({"decode_overrides": {"repetition_penalty": 1.0}}).as_dict() == {
        "repetition_penalty": 1.0
    }
    penalty = DecodeOverrides.model_validate({"repetition_penalty": 1}).as_dict()["repetition_penalty"]
    assert penalty == 1.0
    assert isinstance(penalty, float)  # YAML `1` is stored as a float in the eval record


@pytest.mark.parametrize(
    "raw",
    [
        {"repetition_penaly": 1.0},  # typo: unknown keys are errors, never ignored
        {"top_k": 5},  # not in the allowlist
        {"repetition_penalty": 0.9},
        {"repetition_penalty": 2.1},
        {"no_repeat_ngram_size": -1},
        {"max_new_tokens": 0},
        {"num_beams": 5},
        {"num_beams": 2.0},  # strict: no float -> int coercion
        {"do_sample": 1},  # strict: no int -> bool coercion
        {"temperature": 0.0, "do_sample": True},
        {"temperature": 2.5, "do_sample": True},
        {"temperature": 0.7},  # needs do_sample: true
        {"temperature": 0.7, "do_sample": False},
        ["repetition_penalty", 1.0],
    ],
)
def test_rejects(raw: object) -> None:
    with pytest.raises(ValidationError):
        decode_overrides_of({"decode_overrides": raw})


def test_sampling_with_temperature_ok() -> None:
    assert decode_overrides_of({"decode_overrides": {"do_sample": True, "temperature": 0.7}}).temperature == 0.7


@pytest.mark.parametrize("path", sorted((project_root() / TEMPLATE_DIR).glob("*.yaml")), ids=lambda p: p.stem)
def test_every_template_loads(path: Path) -> None:
    load_template(path)  # validates decode_overrides


def test_bad_template_is_a_recipe_error(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("recipe: R\ndecode_overrides: {repetition_penalty: 3}\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="decode_overrides"):
        load_template(path)
