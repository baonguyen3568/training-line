"""Merge a LoRA adapter into its base model (node ``train.merge_lora``, M1b B-1) — runs INSIDE the training image.

``python -m vlm_pipeline.train.merge_lora --base DIR --adapter DIR --out DIR --auto-class NAME [--dtype bf16|fp32]``

Loads the pinned base snapshot with the catalog's transformers auto class, applies the PEFT adapter written by the
LoRA node (``adapter_config.json`` + ``adapter_model.safetensors``), ``merge_and_unload()`` and saves a plain HF
checkpoint (safetensors) + the base processor files + the base ``generation_config.json`` (decoding parity with the
base, as for full fine-tune exports). The merge math runs in ``--dtype`` (fp32 = W + scale*B@A without rounding the
delta to bf16 first; the worker picks fp32 when the model fits); the saved weights keep the dtype recorded in the
base ``config.json`` (read from the file, not from the loaded config, which reports the load dtype).
Offline (image ENV), CPU only: no GPU needed for a merge. Imports torch/transformers/peft lazily.

Known limitation (measured 2026-10-13, Qwen3.5-0.8B + the S-3 adapter): the transformers class does not build the
multi-token-prediction head, so saving drops the 15 ``mtp.*`` tensors of the base snapshot, and LoRA weights trained on
``mtp.layers`` are not applied. The worker restores ``mtp.*`` from the base (``ckpt_complete.complete_from_base``), so
the merged model's MTP head equals the base one: transformers inference does not use it; vLLM speculative decoding
stays correct (drafts are verified) but may accept fewer draft tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

AUTO_CLASSES = ("AutoModelForImageTextToText",)  # remote-code models (AutoModel) never take the LoRA path (CFG-010)
PROCESSOR_FILES = (
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "vocab.json",
    "merges.txt", "generation_config.json",
)  # fmt: skip


def base_dtype(base: Path) -> str:
    """Weight dtype the base snapshot declares (top level or text_config; ``bfloat16`` when not recorded)."""
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    for scope in (cfg, cfg.get("text_config") or {}):
        value = scope.get("dtype") or scope.get("torch_dtype")
        if isinstance(value, str):
            return value.removeprefix("torch.")
    return "bfloat16"


def merge(base: Path, adapter: Path, out: Path, *, auto_class: str, dtype: str = "bf16") -> dict[str, object]:
    import torch
    import transformers
    from peft import PeftModel

    if auto_class not in AUTO_CLASSES:
        raise ValueError(f"unsupported auto class {auto_class!r}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError("adapter_config.json missing: not a PEFT adapter")
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = getattr(transformers, auto_class).from_pretrained(base, dtype=torch_dtype, device_map="cpu")
    saved_dtype = getattr(torch, base_dtype(base))
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.to(saved_dtype).save_pretrained(out, safe_serialization=True)
    copied = [name for name in PROCESSOR_FILES if (base / name).is_file()]
    for name in copied:
        shutil.copy2(base / name, out / name)
    lora = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    report = {
        "auto_class": auto_class,
        "merge_dtype": dtype,
        "saved_dtype": str(saved_dtype).removeprefix("torch."),
        "copied": copied,
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("lora_alpha"),
        "target_modules": lora.get("target_modules"),
    }
    (out / "merge_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.train.merge_lora")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--auto-class", required=True, choices=AUTO_CLASSES)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args(argv)
    print(json.dumps(merge(a.base, a.adapter, a.out, auto_class=a.auto_class, dtype=a.dtype), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
