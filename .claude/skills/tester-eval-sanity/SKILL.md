---
name: tester-eval-sanity
description: Use before trusting or reporting ANY accuracy number (base, fine-tuned, or quantized VLM) in this project — verifies the eval harness reproduces the base model's published numbers, the config is fully pinned, there is no train/eval leakage, and comparisons are paired with confidence intervals. Also use when someone says "our score is lower/higher than the model card".
---

# Eval sanity gate (VLM accuracy)

Rule: no fine-tune / quantize result is reported until the **base model** passes this gate on the same harness.

## 1. Pin the eval config (must be in the result JSON)
Required fields: `harness` (name+version/commit), `task` + split (e.g. MMMU val, MathVista testmini), `container_digest`,
`git_sha`, `model` (path + revision/hash), `precision` (bf16/fp8/nvfp4), `chat_template` source, `system_prompt`,
`prompt_template`, `max_new_tokens`, `temperature=0`/greedy, `seed`, image `min_pixels/max_pixels`/tiles,
`answer_extractor` (rule | llm-judge + judge model), `n_samples`, `thinking` on/off.
Missing a field ⇒ result is INVALID.

## 2. Reproduce the published number (base model)
1. Get the reference number + protocol from model card / tech report / OpenVLM leaderboard (ask bao-research).
2. Smoke first: run with `--limit 20` to check the pipeline (parsing, template) — inspect raw outputs by eye.
3. Full run (GPU lock — see below). Compare:
   - MC/rule-scored benchmarks: |ours − published| ≤ 1.5 pt  → PASS
   - judge-scored benchmarks: ≤ 2 pt → PASS
   - otherwise FAIL → debug order: chat template → image resolution → prompt/post-prompt → max_new_tokens
     (thinking truncated?) → answer extraction → split/version → transformers/vLLM version.
4. Record outcome in `benchmarks/sanity/<model>-<task>.json` + 1 line in `.team/log/<date>.md`.

## 3. Leakage / shortcut checks
- Dedup train vs eval: image hash (sha256 + pHash) and normalized question text; report overlap count (target 0).
- Blind test: run eval without images on a subset; if score ≫ chance, flag the benchmark as text-solvable.

## 4. Comparing two models (finetune vs base, quant vs bf16)
- Same harness commit, same config except the one variable under test.
- Per-sample paired comparison; report delta with 95% CI (bootstrap 1000x or lmms-eval's paired t-test).
- Delta within CI ⇒ "no significant difference". Never compare against numbers copied from a paper.
- vLLM non-determinism: run twice for at least one benchmark, report spread.
- Spot-check ≥20 samples where the two models disagree.

Quick bootstrap snippet:
```python
import numpy as np
def paired_ci(a, b, n=1000, seed=0):  # a, b: per-sample 0/1 correctness, same order
    rng = np.random.default_rng(seed); d = np.asarray(b, float) - np.asarray(a, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, [2.5, 97.5])
```

## 5. Quantization acceptance (proposal-level defaults, final values in ADR for P002)
- FP8: mean drop ≤ 1 pt; NVFP4: mean drop ≤ 2–3 pt and no single benchmark > 5 pt.
- Calibration data must be disjoint from eval data.
- Vision tower precision must be recorded (bf16 / fp8 / nvfp4).

## GPU lock (mandatory for every GPU job)
Never read or write the lock file by hand (its path moved to `.team/lock/GPU_LOCK` in 28c2088; hand-written locks
count as corrupt). Run GPU work through `jobrun_cli`, which does preflight (ADR-005 80% rule), takes and releases the
lock, runs memguard and writes run.json:
```bash
cd /home/ntcai/Trainning-model
uv run python -m vlm_pipeline.common.gpu_lock status        # who holds it; "held": false = free
uv run python -m vlm_pipeline.common.jobrun_cli --image vlm-automodel:<tag> --tag <short-name> --stage eval \
  --est-peak-gb <N> --who bao-tester -- python /workspace/scripts/qa/<worker>.py ... --out RUN_DIR/pred.jsonl
```
To queue behind another job: `until uv run python -m vlm_pipeline.common.gpu_lock status | grep -q '"held": false';
do sleep 30; done` then the jobrun command. Never kill another member's job.
