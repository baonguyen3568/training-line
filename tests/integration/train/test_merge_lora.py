"""B-1: merge a real LoRA adapter into its base inside the training image (CPU, offline)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("peft")

REPO = Path(os.environ.get("VLM_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BASE = REPO / ".cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
ADAPTER = REPO / "outputs/runs/20261012-1630-lora-qwen3.5-0.8b-s3a/checkpoints/epoch_0_step_19/model"


def test_merged_checkpoint_loads_and_changed_only_lora_targets(tmp_path: Path) -> None:
    if not (BASE / "config.json").is_file() or not (ADAPTER / "adapter_config.json").is_file():
        pytest.skip("Qwen3.5-0.8B snapshot or the S-3 adapter is not on this machine")
    from safetensors import safe_open

    from vlm_pipeline.train.merge_lora import merge

    report = merge(BASE, ADAPTER, tmp_path, auto_class="AutoModelForImageTextToText")
    assert report["lora_r"]
    expected = [n for n in ("tokenizer.json", "generation_config.json", "chat_template.jinja") if (BASE / n).is_file()]
    assert set(expected) <= set(report["copied"])  # processor + decoding files come from the base snapshot
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as f:
        keys = set(f.keys())
    assert not [k for k in keys if "lora_" in k]  # merged, no adapter tensors left
    assert any(k.startswith("model.language_model.") for k in keys)
    assert json.loads((tmp_path / "config.json").read_text())["architectures"]


def test_merged_logits_match_the_unmerged_adapter(tmp_path: Path) -> None:
    """Parity (bao-fixcode): catches a wrong alpha/r scale, save dtype, or a base tensor restored over a merged one."""
    if not (BASE / "config.json").is_file() or not (ADAPTER / "adapter_config.json").is_file():
        pytest.skip("Qwen3.5-0.8B snapshot or the S-3 adapter is not on this machine")
    import torch
    import transformers
    from peft import PeftModel

    from vlm_pipeline.train.ckpt_complete import complete_from_base
    from vlm_pipeline.train.merge_lora import merge

    report = merge(BASE, ADAPTER, tmp_path, auto_class="AutoModelForImageTextToText", dtype="fp32")
    assert report["saved_dtype"] == "bfloat16"  # the base file's dtype, not the fp32 merge dtype
    complete_from_base(BASE, tmp_path)  # what the worker does after the merge: must not overwrite merged tensors
    tok = transformers.AutoTokenizer.from_pretrained(BASE)
    ids = tok("Total 12,500 VND. Thank you!", return_tensors="pt").input_ids
    cls = transformers.AutoModelForImageTextToText
    ref = PeftModel.from_pretrained(cls.from_pretrained(BASE, dtype=torch.float32, device_map="cpu"), ADAPTER).eval()
    merged = cls.from_pretrained(tmp_path, dtype=torch.float32, device_map="cpu").eval()
    with torch.no_grad():
        a, b = ref(input_ids=ids).logits, merged(input_ids=ids).logits
    base = cls.from_pretrained(BASE, dtype=torch.float32, device_map="cpu").eval()
    with torch.no_grad():
        c = base(input_ids=ids).logits
    effect = (a - c).abs()
    assert effect.max().item() > 1e-3  # the adapter really changes the output, so the comparison means something
    # Saved in bf16 (the base dtype): rounding W + delta costs a little of the LoRA effect, never its decisions.
    # Measured 2026-10-13: max 0.104 / mean 0.0072 vs LoRA effect mean 0.28; argmax 100% equal (base alone: 89%).
    diff = (a - b).abs()
    assert diff.mean().item() < 0.1 * effect.mean().item()
    assert (a.argmax(-1) == b.argmax(-1)).all()
    with torch.no_grad():  # the merge math itself, before any bf16 rounding: exact up to fp32 noise
        exact = PeftModel.from_pretrained(cls.from_pretrained(BASE, dtype=torch.float32, device_map="cpu"), ADAPTER)
        exact = exact.merge_and_unload().eval()
        assert (exact(input_ids=ids).logits - a).abs().max().item() < 1e-3
