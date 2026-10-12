---
name: research-benchmark-compare
description: Use when defining the eval suite, running/interpreting VLM benchmarks (MMMU, MMBench, DocVQA, ChartQA, OCRBench, MathVista, AI2D, Video-MME) or comparing our numbers to model-card/published numbers — protocol, reproduction checklist, and how to explain discrepancies.
---

# VLM benchmark protocol & comparison with published numbers

## 1. Standard suite (Proposal 002)
| Tier | Benchmarks | When |
|---|---|---|
| Smoke (minutes) | 100-200 sample fixed subsets of MMBench-dev, ChartQA-test, OCRBench | every PR / CI, fine-tune sanity |
| Core (≤ ~2h) | MMMU-val, MMBench-EN-dev, DocVQA-val, ChartQA-test, OCRBench, AI2D-test, MathVista-testmini | every candidate checkpoint, every quant variant |
| Extended | Video-MME (w/o subs), MMStar, RealWorldQA, domain task set | release candidates |
| Perf | TTFT, TPOT/tokens/s, req/s at fixed concurrency, peak memory, model size on disk | every serve format (BF16/FP8/NVFP4) |
Domain eval (our fine-tune target) always reported next to general benchmarks → shows gain + forgetting.

## 2. Reproducing a published number — checklist
Before claiming "our base model = X on benchmark Y", fix and record:
- [ ] Harness + version (VLMEvalKit / lmms-eval / NeMo Evaluator), commit hash.
- [ ] Split exactly as model card (MMMU **val**, MathVista **testmini**, DocVQA **val** vs test-server).
- [ ] Prompt template & system prompt (model card / official eval script), chat template applied.
- [ ] Thinking/reasoning mode on or off; max_new_tokens big enough for thinking (≥ 8k-32k) else answers truncated.
- [ ] Image resolution / `max_pixels` / tiles, video frames & fps.
- [ ] Decoding: greedy (temp 0) unless card says otherwise; seed.
- [ ] Answer extraction: exact match vs LLM-judge (MathVista, MMBench option-matching often use GPT judge) — record judge model; use a local judge (NIM on this box) if no API, and note it.
- [ ] Metric: ANLS (DocVQA), relaxed accuracy (ChartQA), OCRBench score /1000.
Acceptable deviation for base model: within ±1.5 points (±2-3 for judge-based). Larger → investigate before trusting any fine-tune delta.

## 3. Comparing variants
- Same harness, same config, same subset for base / fine-tuned / FP8 / NVFP4.
- Report mean and, for small subsets, 95% CI (bootstrap) — differences < CI are noise.
- Quantization acceptance (proposal default): FP8 ≤ 1 pt drop, NVFP4/INT4 ≤ 2-3 pt drop averaged over core suite; otherwise flag.

## 4. Result record
Use the flat contract owned by `bao-frontend` (skill `frontend-metrics-contract`): one flat record per
(model, variant, precision, benchmark, metric). Key fields: `value` on 0-100 scale + `scale`, `source`
(measured|published), harness + version, prompt_hash, config_hash, `thinking`, `judge_model`, `max_new_tokens`,
n_samples, ci_low/ci_high, git_commit, date. Perf = separate records (`benchmark: "perf"`).

## 5. Rules
- Heavy eval = GPU job → `.team/GPU_LOCK`.
- Never fine-tune on benchmark test data; run n-gram/image-hash contamination check on training sets.
- Agree any protocol change with `bao-tester` and log it.
