---
name: research-model-selection
description: Use when picking or re-evaluating a base VLM (Nemotron VL/Omni, Qwen3.x-VL, InternVL, Gemma, ...) or a fine-tuning recipe (LoRA/QLoRA/full SFT/GRPO/DPO) for this project — scoring matrix, memory budget math for the shared GB10, and hyperparameter starting points.
---

# Base model + recipe selection (single GB10, shared machine)

## 1. Hard constraints (fail = drop candidate)
- Open weights on HF, official org (`nvidia/`, `Qwen/`, `OpenGVLab/`, `google/`...), not a re-upload.
- License allows our use: Apache-2.0 / MIT / NVIDIA Open Model License OK; Gemma terms, Llama community license → flag for `bao-security`; non-commercial (CC-BY-NC, research-only) → reject unless PM/user explicitly accepts.
- Supported by the training lib we pinned (AutoModel `examples/vlm_finetune/<family>` exists) **and** by at least one serving path (vLLM or TRT-LLM) on arm64.
- Fits memory budget (below) with LoRA in **≤ 40 GB** (real free RAM on this shared box; re-check `free -g`).

## 2. Memory budget math (BF16, rule of thumb — verify with a smoke run)
| Item | Formula |
|---|---|
| Weights BF16 | 2 B × params (MoE: *total* params, not active) |
| LoRA trainable + Adam | ~ (r-dependent, usually <1% params) × 16 B |
| Full SFT (Adam, BF16+fp32 master) | ~16 B × params → 8B model ≈ 128 GB → **not feasible** |
| QLoRA (NF4 base) | ~0.55 B × params + LoRA |
| Activations | grows with seq_len × batch; image tokens dominate (cap `max_pixels`, use grad checkpointing) |
Reference measurements (bao-backend, AutoModel 26.08 Spark recipes): Qwen3-8B LoRA ≈ 21 GiB, Qwen3-14B LoRA ≈ 43 GiB, 8B full SFT ≈ 78 GiB.
⇒ MoE 30-35B (e.g. 35B-A3B) BF16 weights alone ≈ 70 GB → only with QLoRA/FP8 base or when the box is freed.

## 3. Scoring matrix (weights → total /100)
| Criterion | Weight |
|---|---|
| Published quality on our target benchmarks (MMMU, MMBench, DocVQA, ChartQA, OCRBench, MathVista, Video-MME) | 30 |
| Stack support (AutoModel recipe, ModelOpt PTQ, vLLM/TRT-LLM, official FP8/NVFP4 ckpt) | 25 |
| Fits budget with headroom (LoRA ≤ 30 GB ideal) | 20 |
| License clarity | 15 |
| Ecosystem / community recipes / tokenizer & chat template sanity | 10 |
Always pick: **1 tiny dev model (≤4B)** for smoke/CI + **1 main target** + 1 fallback.

## 4. Recipe starting points (tune per model, cite source in proposal)
- LoRA SFT: r=16-64, alpha=2r, dropout 0.05, targets = LLM attn+MLP linear (vision tower frozen, projector optionally trainable), lr 1e-4 to 2e-4 cosine, warmup 3%, 1-3 epochs, grad checkpointing on, BF16, global batch 16-64 via grad accumulation, seed fixed.
- Full-LLM SFT only for ≤4B models.
- Vision encoder: freeze by default; unfreeze only with evidence (domain shift, OCR-heavy) and lr ×0.1.
- RL (GRPO, NeMo-RL): after SFT, verifiable rewards (exact match / format), group size 4-8, KL 0.0-0.04, small model first; rollout engine (vLLM) shares UMA → budget both.
- DPO: beta 0.1, lr 5e-7 to 5e-6.
- Data: HF chat-messages JSONL with `{"messages":[{"role":"user","content":[{"type":"image","image":...},{"type":"text","text":...}]},...]}`; dedupe, check image decode, cap resolution; keep a held-out split; never train on benchmark test sets (contamination check).

## 5. Output
Write comparison table + decision + risks into `.team/proposals/001-*.md`; send `bao-security` the license list, `bao-backend` the recipe YAML sketch, `bao-cleanup` the disk estimate.

## Remote-code bridge check (mandatory for every model; missed twice on 2026-10-12)
- Walk config.json **recursively**: every `auto_map` and `_name_or_path` at ANY level (encoder/vision_config/text_config/...),
  not just the root. `org/repo--module.Class` or a foreign `_name_or_path` = code pulled from ANOTHER repo at `main`.
- Grep the repo's `.py` for `get_class_from_dynamic_module`, `trust_remote_code=True`, `torch.load`, `urlretrieve`.
- Tool: `python3 -I scripts/research/hf_model_meta.py org/repo` (prints subconfig_auto_map). Gated repos: config unreadable
  without an accepted token -> verdict must stay CONDITIONAL.
- Preprocessing flags can be silently DROPPED when a model's config builds a sub-config from another repo
  (e.g. Nemotron Parse: `image_processor_normalizes` lost because RADIOConfig lacks the attr → double normalization).
  Verify numerically (synthetic white image at patch-embed input: once = (1-m)/s), and check train/serve parity
  of pixel_values (AutoModel path vs vLLM/HF path) before trusting any accuracy number.
- Before blaming memory on a component, inspect the BUILT model (meta device is enough): trainable-param count per
  top-level prefix, which modules got activation checkpointing (AutoModel logs `selected N/N trainable layers`), and
  attention kernel flags. Docstrings/recipe YAML are not evidence: AutoModel's freeze_config matches module NAMES
  (`vision_tower`/`vision`/`visual`/`image_encoder`) and silently ignores unknown keys (e.g. `freeze_embeddings`).
- Train/serve prompt parity: the decoder/input prefix used at eval and in "Chạy thử" must equal the one the collate
  built at train time. Derive it by running the collate on one sample and cutting at the first labelled token -
  never infer it from the label mask (masked tokens are still in the input; e.g. Nemotron Parse keeps its task
  tokens `</s><s><predict_bbox><predict_classes><output_markdown>` in input but masks them from labels).
- Precision on ONE GPU: AutoModel skips FSDP2 at world size 1, so `distributed.mp_policy` (bf16 compute / fp32 master)
  is NEVER applied. `torch_dtype: bfloat16` + `torch.optim.AdamW` = pure bf16 params: Adam steps (~lr) below half a
  bf16 ulp are rounded away (Parse P-3, lr 1e-5: only 5.3% of decoder elements changed in 300 steps; probe: ~7x update
  lost). Use TE `FusedAdam` with `master_weights: true, store_param_remainders: true` (upstream pattern). Check
  "fraction of trainable elements changed vs base" after any run whose quality looks wrong - val_loss alone hides it.
- Budget sanity: compare samples seen (GBS × steps) with the upstream recipe before setting presets (P-3 = 15% of it).
- Memory numbers on GB10 (UMA): the peak is ALWAYS `run.json` `peak_mem_gb` (jobrun memguard, host-wide), never
  `torch.cuda.max_memory_allocated` (allocator only: B-4 probe showed 6.3 GB torch vs 16.6 GB real). Label any torch
  number `torch_alloc_peak_gb`. The "no invented numbers" site gate checks a number exists in a source, not its meaning.
