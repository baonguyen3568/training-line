# 020 — GRPO training node (`train.grpo_vlm`), worker side

- Author: bao-backend · 2026-10-14 · Status: DRAFT, gated on research's NeMo-RL probe (PM: ~1 day, then 1.0 or 1.1)
- Split (agreed with bao-cleancode): cleancode owns the node spec / params / i18n / standards in `nodes/`;
  backend owns the worker, recipe, check_support, memory estimate and job runner.
- Inputs used: research 2026-10-14 (NeMo-RL README, DGX Spark VLM playbook), security release-1.0 review §2
  (built-in rewards only), PM decision 2026-10-14 (GRPO in 1.0 only if the probe passes; DPO on VLM is 1.1).

## 1. What the user gets

One node "Huấn luyện tăng cường (GRPO)" after a model pick and a dataset, before Merge / Eval / Export like the
LoRA node. It trains a LoRA adapter with GRPO: for each prompt the model writes several answers, a **built-in
reward** scores them, and the model moves towards the better ones. No code anywhere.

## 2. Data contract (no new port type)

The node takes the ordinary `Dataset` port, so filter / dedupe / mix / split / DATA-V keep working. Each record
needs `image`, `prompt` (question) and `answer` (reference used by the reward). `data.local_folder` reads it with
the existing `input_format` (a GRPO record is an SFT record whose target is the reference answer). The hub
reference dataset is `hiyouga/geometry3k` (security: APPROVED-WITH-CONDITIONS, research).

check_support refuses, before Run (canvas and run use the same check):
- a model that is not in the GRPO list measured by research (start: Qwen3.5-0.8B, then 2B) → CFG-011;
- a dataset whose DATA-V preview shows no `answer` / `target` field → DATA-00x (new code, cleancode + design);
- `reward=json_field_f1` with records whose answer is not a JSON object → same code.

## 3. Rewards (built-in only, security §2)

`reward` enum, each a small pure function in our code (unit-tested, no `eval`, no user Python):

| id | score in [0, 1] |
|---|---|
| `exact_match` | normalized answer == reference (case, spaces, punctuation folded) |
| `numeric_tolerance` | first number in the answer within `tolerance` (default 1e-2 relative) of the reference |
| `json_field_f1` | field-level F1 of a JSON answer vs the reference (the CORD scorer, reused) |
| `format_regex` | answer fully matches a user regex (≤ 200 chars, compiled with a timeout in the worker) |

Total reward = `format_weight` × format bonus (answer has the expected shape: a number / a JSON object / matches
the regex) + `correctness_weight` × correctness, defaults 2.0 / 5.0 as in the DGX Spark playbook.

## 4. Engine and recipe

- Image: NeMo-RL v0.7.0 arm64, pinned by digest in IMAGES.lock after security vets it (PM approved the pull).
- The worker renders a NeMo-RL GRPO config from a per-model template in `configs/train/templates/grpo/`
  (research writes the first one from the probe): `num_generations` (default 4), `max_new_tokens`, `kl_coef`,
  LoRA rank 8–64, learning rate, epochs. Advanced keys follow P016 §3 (allowlist from the template).
- The reward runs inside the job: the worker mounts our reward module read-only (code from `src/`, like every
  job) and passes the reward id + weights + regex through the config. The job stays `--network none`, narrow
  mounts, non-root, no secret.
- Output: a LoRA `Adapter` artifact like `train.lora_vlm`, so Merge / Export / Eval work unchanged; metrics.jsonl
  gets `reward_mean`, `reward_std`, `kl`, `loss` per step (frontend contract: new metric names, ask frontend).

## 5. Memory and time

GRPO holds the policy, a reference model (or LoRA-off pass) and `num_generations` rollouts in memory. The estimate
comes from research's probe numbers (peak RAM at 0.8B, then 2B), scaled like the LoRA estimate; until measured, the
node is refused by check_support (no guessing on a shared 128 GB machine).

## 6. Done when

1. Research probe PASS (NeMo-RL v0.7.0 on GB10, 1 GPU, CUDA 13.2 on driver 580) and a GRPO smoke on Qwen3.5-0.8B
   whose reward goes up.
2. Node spec (cleancode) + worker + rewards + tests; the reference pipeline R5 (DGX Spark GRPO) rebuilt with our
   nodes and run once on GB10 with its numbers recorded.
3. Security review of the job (image, mounts, regex timeout).

If 1 fails within the PM's time box: the node ships as "sắp có" (disabled, real reason shown), all of this moves
to 1.1, and DPO joins it there.
