# Upstream issue draft — NeMo AutoModel VLM QLoRA quantizes the tied lm_head and the vision tower (Qwen3.5)

- **Status: DRAFT, not posted.** Posting needs PM/user approval (no agent posts upstream). Found 2026-10-14 by
  bao-research during the QLoRA vs LoRA A/B for the app's `train.lora_vlm` `quantize_base: nf4` option.
- Where: https://github.com/NVIDIA-NeMo/Automodel/issues/new (image `vlm-automodel:26.08-6`, AutoModel r0.6.0,
  bitsandbytes 0.49.2, transformers 5.12, GB10 arm64).

## Reproduction (facts we have)

- Recipe `FinetuneRecipeForVLM`, model `Qwen/Qwen3.5-2B` @15852e8c (`tie_word_embeddings: true`), PEFT LoRA dim 8,
  and the `quantization:` block of AutoModel's own Spark QLoRA recipe (load_in_4bit, nf4, double quant, bf16 compute,
  bf16 quant storage). Config: `configs/train/smoke/qwen35_2b_qlora_nf4_ab.yaml` (this repo).
- `components/quantization/qlora.py::create_bnb_config` builds `BitsAndBytesConfig` with no
  `llm_int8_skip_modules`, so every Linear becomes `Linear4bit`, including the vision tower (`visual.blocks.*.attn.qkv`)
  and `lm_head`, whose weight is tied to `embed_tokens`.
- First training step: `UserWarning: FP4 quantization state not initialized` then `AssertionError` in
  `bitsandbytes/nn/modules.py:415 fix_4bit_weight_quant_state_from_module`, called from `self.lm_head(...)` at
  `transformers/models/qwen3_5/modeling_qwen3_5.py:1833`. Run `20261014-0842-q35-2b-qlora-nf4-ab-cd64` (local).
- The same config without the `quantization:` block trains fine (40 steps, val_loss 0.051).

## Expected

`create_bnb_config` passes through `llm_int8_skip_modules` (HF's name, also used for 4-bit), defaulting for VLMs to
`["lm_head", <vision tower>, <projector>]`, or the recipe skips tied output embeddings automatically.
