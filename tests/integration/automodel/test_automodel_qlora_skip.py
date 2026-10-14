"""Patch 0006 (AutoModel streaming BnB QLoRA): a tied lm_head is never 4-bit; recipe skips reach the loader.

Bug (bao-research 2026-10-14, run 20261014-0842-q35-2b-qlora-nf4-ab-cd64; root cause bao-fixcode): the streaming BnB
path in ``model_init._init_model_bnb_streaming`` skipped only ``llm_int8_skip_modules`` / ``_keep_in_fp32_modules``,
not what HF's loader always adds (tied weights, output embeddings), so Qwen3.5's tied lm_head became Linear4bit and the
first step failed in ``fix_4bit_weight_quant_state_from_module``; ``create_bnb_config`` also dropped the recipe's
``llm_int8_skip_modules``, so the vision tower could not be kept in bf16.

Runs INSIDE vlm-automodel (offline, CPU, meta device: no weights loaded, no GPU). Must FAIL on 26.08-6 and PASS with
patch 0006. Real config of the pinned Qwen3.5-0.8B snapshot (tie_word_embeddings: true).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")
pytest.importorskip("bitsandbytes")

REPO = Path(__file__).resolve().parents[3]
QWEN = REPO / ".cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"


def _skeleton(monkeypatch: pytest.MonkeyPatch, skip: list[str] | None):  # type: ignore[no-untyped-def]
    """The model `_init_model_bnb_streaming` builds, with weight streaming stubbed out (meta tensors, CPU only)."""
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig

    from nemo_automodel._transformers import model_init
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.quantization.qlora import create_bnb_config

    if not (QWEN / "config.json").is_file():
        pytest.skip(f"pinned Qwen3.5-0.8B snapshot not found at {QWEN}")
    monkeypatch.setattr(model_init, "_download_model_weights", lambda *a, **k: None)
    monkeypatch.setattr(model_init, "_stream_load_bnb_weights", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    quant = {"load_in_4bit": True, "load_in_8bit": False, "bnb_4bit_quant_type": "nf4"}
    if skip is not None:
        quant["llm_int8_skip_modules"] = skip
    bnb_config = create_bnb_config(OmegaConf.create(quant))
    hf_config = AutoConfig.from_pretrained(str(QWEN))
    _, model = model_init._init_model_bnb_streaming(
        NeMoAutoModelForImageTextToText, str(QWEN), hf_config, "sdpa", torch.bfloat16, bnb_config
    )
    return model


def _kinds(model) -> dict[str, str]:  # type: ignore[no-untyped-def]
    import bitsandbytes as bnb
    import torch

    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            out[name] = "4bit"
        elif isinstance(mod, torch.nn.Linear):
            out[name] = "linear"
    return out


def test_tied_lm_head_stays_unquantized_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds = _kinds(_skeleton(monkeypatch, skip=None))
    assert kinds["lm_head"] == "linear"  # tied to embed_tokens: its weight is the bf16 embedding
    assert any(k.endswith("mlp.down_proj") and v == "4bit" for k, v in kinds.items())  # the LLM is still 4-bit


def test_recipe_skips_reach_the_loader_and_keep_the_vision_tower_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds = _kinds(_skeleton(monkeypatch, skip=["model.visual"]))
    visual = {k: v for k, v in kinds.items() if k.startswith("model.visual.")}
    assert visual
    assert set(visual.values()) == {"linear"}  # vision tower + merger (projector) kept in bf16
    assert kinds["lm_head"] == "linear"  # the defaults still apply next to the recipe's skips
    assert any(k.endswith("mlp.down_proj") and v == "4bit" for k, v in kinds.items())
