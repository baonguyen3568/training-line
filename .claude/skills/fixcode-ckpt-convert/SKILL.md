---
name: fixcode-ckpt-convert
description: Use when converting VLM checkpoints HF ⇄ Megatron (Megatron-Bridge), merging LoRA, or exporting quantized checkpoints (ModelOpt → TRT-LLM/vLLM), and whenever a converted/exported model fails to load (KeyError, missing/unexpected keys, shape mismatch) or loads but produces garbage — verifies key mapping, tied embeddings, vision tower prefixes, and runs a numerical round-trip check.
---

# Checkpoint conversion debugging & verification

Background: `.team/knowledge/fixcode.md` §2.6–2.7. "It loads" ≠ "it is correct". Always compare numbers.

## 1. Inspect both sides (no GPU needed)
```python
import json, safetensors
idx = json.load(open(f"{hf_dir}/model.safetensors.index.json"))["weight_map"]
prefixes = sorted({".".join(k.split(".")[:3]) for k in idx})   # e.g. model.visual / model.language_model.visual
cfg = json.load(open(f"{hf_dir}/config.json"))
print(cfg.get("architectures"), cfg.get("transformers_version"), cfg.get("tie_word_embeddings"))
print("lm_head present:", any(k.startswith("lm_head") for k in idx))
```
Checklist:
- [ ] Vision tower prefix matches what the bridge/loader expects (`model.visual.*` vs `model.language_model.visual.*` changed across transformers versions). Fix = re-key, never touch weights.
- [ ] `tie_word_embeddings` consistent with presence of `lm_head.weight`; Megatron side untie flag matches.
- [ ] Projector / merger keys mapped (often dropped silently by text-only bridges).
- [ ] dtype recorded (bf16 vs fp32); vision encoder not down-cast unexpectedly.
- [ ] `transformers` version used for convert == version pinned for train/eval (per `config.json`).
- [ ] GQA / QKV fused layout: num_heads, num_kv_heads, head_dim identical both sides.

## 2. Round-trip test (mandatory for any converter we ship)
1. HF → Megatron → HF (Megatron-Bridge `AutoBridge` import/export).
2. Tensor-wise: every key present both sides; `torch.equal` for lossless dtype, else `allclose(atol=1e-3 bf16)`. List mismatches.
3. Functional: same fixed batch (1 image + prompt, seed fixed), compare logits of original vs round-tripped: `max_abs_diff`, top-1 token agreement ≥ 99%.
4. Generate 32 tokens greedy on 3 fixed prompts — outputs must match (bf16 tolerance: allow late divergence only).
Use a small model (≤4B) or `tiny-random` for the CI version; real model run needs GPU lock.

## 3. LoRA merge / quantized export
- After `merge_and_unload()`: logits(merged) ≈ logits(base+adapter) on same batch.
- Quantize (ModelOpt): print quantized module names; assert vision tower + projector NOT quantized; compare a mini-eval (e.g. 50 samples) vs bf16 before full benchmark.
- Export: load the exported ckpt with the exact serving runtime version (TRT-LLM / vLLM) inside the pinned container; pin ModelOpt ↔ TRT-LLM ↔ vLLM together.

## 4. Typical errors
| Error | First check |
|---|---|
| `KeyError: model.visual.patch_embed...` | prefix mismatch (§1) |
| `size mismatch for ... qkv` | head counts / fused layout / TP split |
| Loads, outputs gibberish | QKV interleave order, rope config (`rope_theta`, mrope sections), tied embedding copied wrong |
| Loads, quality slightly lower | dtype cast, missing norm bias, image processor config not copied (`preprocessor_config.json`, `chat_template`) |
| Exported model missing chat template | copy `tokenizer_config.json`/`chat_template.jinja`/`preprocessor_config.json` with the weights |
