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
