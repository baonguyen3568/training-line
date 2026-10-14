"""Patch 0007 (AutoModel freeze_config per block): ``freeze_prefixes`` / ``train_prefixes`` and the PEFT freeze.

Why (ADR-022 #4 two-stage, #5 vision unfrozen; bao-research 2026-10-14): ``freeze_config`` only knew vision / audio /
language-model name patterns. Qwen keeps its projector (``merger``) inside ``visual`` and LLaVA-style models keep
``lm_head`` outside ``language_model``, so "train the projector only" could not be expressed, and with PEFT every
non-LoRA parameter was frozen again after the freeze step, so "projector + LoRA" (stage 2) trained LoRA only.

Runs INSIDE vlm-automodel (offline, CPU, meta device). Must FAIL on 26.08-7 (unknown freeze_config keys) and PASS with
patch 0007. Real configs of the pinned snapshots; the merge test loads Qwen3.5-0.8B weights on CPU (~2 GB).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
HUB = REPO / ".cache/hf/hub"
QWEN = HUB / "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
LLAVA_OV = HUB / "models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf/snapshots/74dd0bf867a4cda7950c17663794267c60cf4b40"


def _meta(path: Path):  # type: ignore[no-untyped-def]
    import torch
    import transformers

    if not (path / "config.json").is_file():
        pytest.skip(f"pinned snapshot not found at {path}")
    cfg = transformers.AutoConfig.from_pretrained(str(path))
    with torch.device("meta"):
        return getattr(transformers, cfg.architectures[0])(cfg)


def _share(model, part: str) -> float:  # type: ignore[no-untyped-def]
    """Trainable share of the parameters whose name contains ``part`` (each tensor once)."""
    params = [p for n, p in model.named_parameters() if part in n]
    return sum(p.numel() for p in params if p.requires_grad) / sum(p.numel() for p in params)


def test_qwen_stage1_trains_the_merger_only() -> None:
    from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

    m = _meta(QWEN)
    stage1 = {"freeze_vision_tower": True, "freeze_language_model": True, "freeze_prefixes": ["lm_head"]}
    apply_parameter_freezing(m, {**stage1, "train_prefixes": ["model.visual.merger"]})
    assert _share(m, "visual.merger.") == 1.0
    assert _share(m, "visual.blocks.") == 0.0
    assert _share(m, "language_model.") == 0.0


def test_llava_stage1_trains_the_projector_only() -> None:
    from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

    m = _meta(LLAVA_OV)
    stage1 = {"freeze_vision_tower": True, "freeze_language_model": True, "freeze_prefixes": ["lm_head"]}
    apply_parameter_freezing(m, {**stage1, "train_prefixes": ["model.multi_modal_projector"]})
    assert _share(m, "multi_modal_projector.") == 1.0
    assert _share(m, "vision_tower.") == 0.0
    assert _share(m, "language_model.") == 0.0


def test_a_prefix_matching_nothing_is_refused() -> None:
    from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

    with pytest.raises(RuntimeError, match="no parameter matched"):
        apply_parameter_freezing(_meta(QWEN), {"train_prefixes": ["model.visual.mergerX"]})


def test_prefix_matches_whole_module_names_only() -> None:
    from nemo_automodel.components.utils.model_utils import param_matches_prefix

    assert param_matches_prefix("_orig_mod.model.visual.merger.linear_fc1.weight", "model.visual.merger")
    assert not param_matches_prefix("model.visual.merger2.linear_fc1.weight", "model.visual.merger")


def test_peft_freeze_keeps_train_prefixes_including_a_tied_lm_head() -> None:
    from nemo_automodel.components.utils.model_utils import freeze_non_lora_params

    m = _meta(QWEN)  # Qwen3.5-0.8B: lm_head tied to embed_tokens (listed once, under embed_tokens)
    assert m.lm_head.weight is m.get_input_embeddings().weight
    freeze_non_lora_params(m, {"train_prefixes": ["model.visual.merger", "lm_head"]})
    assert _share(m, "visual.merger.") == 1.0
    assert m.lm_head.weight.requires_grad
    assert _share(m, "visual.blocks.") == 0.0
    freeze_non_lora_params(m, None)  # the unpatched behaviour: nothing but LoRA stays trainable
    assert not any(p.requires_grad for p in m.parameters())


def test_merge_applies_a_full_trained_block_saved_next_to_lora(tmp_path: Path) -> None:
    """Stage 2 saves the trained projector in adapter_model.safetensors (AutoModel saves every requires_grad tensor);
    merge_lora (PeftModel.from_pretrained + merge_and_unload) must put the TRAINED tensor in the merged model."""
    import sys

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file

    if not (QWEN / "config.json").is_file():
        pytest.skip(f"pinned snapshot not found at {QWEN}")
    sys.path.insert(0, str(REPO / "src"))
    from vlm_pipeline.train.merge_lora import merge

    key = "model.visual.merger.linear_fc1.weight"
    base = transformers.AutoModelForImageTextToText.from_pretrained(str(QWEN), dtype=torch.bfloat16, device_map="cpu")
    trained = (base.get_parameter(key).detach().float() + 0.5).to(torch.bfloat16)
    peft_model = get_peft_model(base, LoraConfig(r=4, target_modules=["model.language_model.layers.0.mlp.down_proj"]))
    adapter = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter))
    sd = load_file(str(adapter / "adapter_model.safetensors"))
    sd[f"base_model.model.{key}"] = trained
    save_file(sd, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    del base, peft_model

    merge(QWEN, adapter, tmp_path / "merged", auto_class="AutoModelForImageTextToText", dtype="bf16")
    merged = {}
    for f in sorted((tmp_path / "merged").glob("*.safetensors")):
        merged.update(load_file(str(f)))
    assert torch.equal(merged[key], trained)
