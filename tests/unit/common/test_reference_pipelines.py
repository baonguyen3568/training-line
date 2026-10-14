"""Gallery schema v2, by training method (ADR-022, P023 §5): strict, one pipeline per method, comparable scores."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from vlm_pipeline.common.reference_pipelines import ReferencePipelines


def _pipeline(pid: str, method: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "method": method,
        "title": {"vi": "x", "en": "x"},
        "summary_key": f"reference_pipeline.{pid}.summary",
        "when_to_use_key": f"method.{method}.when",
        "kind": "app_template",
        "template_id": "t",
        "status": "available",
        "difficulty": "basic",
        "train_arch": {"vision": "frozen", "projector": "frozen", "lm": "lora", "quant": "nf4"},
        "source": {"url": "https://arxiv.org/abs/2305.14314", "publisher": "UW"},
        "license": {"code": "apache-2.0"},
        "steps": [{"node": "train.lora_vlm", "label": {"vi": "LoRA", "en": "LoRA"}}],
        **extra,
    }


def _file(*pipelines: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 2, "pipelines": list(pipelines)}


DPO = _pipeline(
    "dpo-pref",
    "dpo",
    train_arch={"vision": "frozen", "projector": "frozen", "lm": "lora", "extras": ["reference_shared"]},
)


def test_a_qlora_and_a_dpo_card_load() -> None:
    loaded = ReferencePipelines.model_validate(_file(_pipeline("qlora-4bit", "qlora"), DPO))
    qlora, dpo = loaded.pipelines
    assert (qlora.train_arch.lm, qlora.train_arch.quant) == ("lora", "nf4")  # LoRA + 4-bit: quant apart from state
    assert dpo.train_arch.extras == ("reference_shared",)  # LoRA: the reference is the base with the adapter off


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda p: p.update(method="rlhf"), "method"),
        (lambda p: p["train_arch"].update(lm="quantized"), "lm"),  # quantization is `quant`, not a state
        (lambda p: p.update(summary_key="reference_pipeline.other.summary"), "i18n keys"),
        (lambda p: p.update(when_to_use_key="method.lora.when"), "method.qlora.when"),
        (lambda p: p.update(colour="red"), "colour"),
        (lambda p: p.update(measured={"config": "x", "sources": ["benchmarks/x.json"]}), "measured_with"),
        (lambda p: p["train_arch"].update(quant="fp8_fake"), "quant"),  # QAT is quant + fake_quant
        (lambda p: p["train_arch"].update(quant=None, fake_quant=True), "fake_quant needs"),
        (
            lambda p: p["train_arch"].update(
                stages=[
                    {
                        "name_key": "reference_pipeline.qlora-4bit.s1",
                        "vision": "frozen",
                        "projector": "full",
                        "lm": "frozen",
                    },
                    {
                        "name_key": "reference_pipeline.qlora-4bit.s2",
                        "vision": "frozen",
                        "projector": "full",
                        "lm": "full",
                    },
                ]
            ),
            "last stage",
        ),
    ],
)
def test_bad_cards_fail_loading(change: Any, needle: str) -> None:
    card = copy.deepcopy(_pipeline("qlora-4bit", "qlora"))
    change(card)
    with pytest.raises(ValidationError, match=needle):
        ReferencePipelines.model_validate(_file(card))


def test_one_pipeline_per_method() -> None:
    with pytest.raises(ValidationError, match="one pipeline per method"):
        ReferencePipelines.model_validate(_file(_pipeline("a", "qlora"), _pipeline("b", "qlora")))


def test_the_shipped_gallery_has_the_ten_methods_once_it_is_on_v2() -> None:
    """Release gate (ADR-022): self-activates when research moves the file to schema_version 2."""
    import yaml

    from vlm_pipeline.common.paths import project_root
    from vlm_pipeline.common.reference_pipelines import REFERENCE_PIPELINES, Method, load_reference_pipelines

    path = project_root() / REFERENCE_PIPELINES
    if yaml.safe_load(path.read_text(encoding="utf-8")).get("schema_version") != 2:
        pytest.skip("configs/reference_pipelines.yaml still on schema v1 (research migrates it)")
    from typing import get_args

    methods = {p.method for p in load_reference_pipelines(path).pipelines}
    assert methods == set(get_args(Method)), sorted(set(get_args(Method)) - methods)
