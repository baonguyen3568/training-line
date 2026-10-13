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
- [ ] Silent config renames: every `getattr(config, name, default)` in the model class — does `config.json` (incl. sub-configs) use that exact name? Check alias table in KB §2.6b (e.g. `image_processor_normalizes` vs `processor_normalizes`).
- [ ] GQA / QKV fused layout: num_heads, num_kv_heads, head_dim identical both sides.

## 2. Round-trip / equivalence test (mandatory; thresholds = Proposal 007 R6, agreed with bao-tester)
1. HF → Megatron → HF same dtype: **bit-exact** — `torch.equal` per tensor, no missing/extra keys (atol would hide a wrong transpose).
2. Logits on a fixed batch (1 image + prompt, seed fixed, same kernel + `attn_implementation`): gate on cosine ≥ 0.999
   or mean KL ≤ 1e-3; log `max_abs_diff` only (bf16 logits of 10–30 make absolute atol meaningless).
3. tiny-random models (CPU, `tests/integration/`): bit-exact weights + fp32 logits allclose. Never gate on top-1 there
   (random weights give near-ties).
4. Real model (`tests/gpu/`, GPU lock): greedy decode on ~16 prompts **with images**, short + long; pass if identical or
   the first divergence is a near-tie (top1 − top2 < 0.1 logit).

## 3. LoRA merge / quantized export
- After `merge_and_unload()`: W_merged vs fp32 reference `W + scale·B·A` (relative tol ~1e-2 for bf16); logits merged vs
  unmerged with the criteria of §2.2 + §2.4.
- Quantize (ModelOpt): print quantized module names; assert the actual precision of vision tower + projector **equals
  `vision_precision` in the config** (default bf16 — P002 allows ModelOpt's FP8 vision recipe). Top-1 agreement is logged,
  never a gate. A 50-sample mini-eval (CI95 ≈ ±14 pt) is only a smoke test (score ≥ base − 10 pt, non-empty output);
  the real gate is P002 G3.
- Export: load the exported ckpt with the exact serving runtime version (TRT-LLM / vLLM) inside the pinned container; pin ModelOpt ↔ TRT-LLM ↔ vLLM together.

## 4. Typical errors
| Error | First check |
|---|---|
| `KeyError: model.visual.patch_embed...` | prefix mismatch (§1) |
| `size mismatch for ... qkv` | head counts / fused layout / TP split |
| Loads, outputs gibberish | QKV interleave order, rope config (`rope_theta`, mrope sections), tied embedding copied wrong |
| Loads, quality slightly lower | dtype cast, missing norm bias, image processor config not copied (`preprocessor_config.json`, `chat_template`) |
| Exported model missing chat template | copy `tokenizer_config.json`/`chat_template.jinja`/`preprocessor_config.json` with the weights |
