# 020 — GRPO training node (`train.grpo_vlm`), worker side

- Author: bao-backend · 2026-10-14 · Status: DRAFT v2 (cleancode review folded in), gated on research's NeMo-RL probe
  (PM: ~1 day, then 1.0 or 1.1)
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
needs `image`, `prompt` (question) and **`target`** (the reference the reward compares with). There is no new
field: "answer" is only the UI's word for `Record.target` (cleancode). `data.local_folder` reads it with the
existing `input_format`. The hub reference dataset is `hiyouga/geometry3k` (security: APPROVED-WITH-CONDITIONS).

check_support refuses, before Run (canvas and run use the same check):
- a model whose catalog entry does not list GRPO (`rl_modes: [grpo]`, next to `train_node`, like CFG-010; research
  edits the YAML as models are measured: 0.8B first, then 2B) → CFG-011;
- a dataset without the fields the node needs → **`DATA-008 DATASET_NOT_FOR_NODE`** `{node, dataset, missing}`, one
  code shared with DPO in 1.1 (cleancode + design write it). Local data: DATA-V preview. Hub data: the catalog's new
  `fields: [image, prompt, target]` (DATA-V cannot preview a hub dataset);
- `reward=json_field_f1` with targets that are not JSON objects → the same DATA-008 (`missing: json_target`).

## 3. Rewards (built-in only, security §2)

`reward` enum, each a small pure function in our code (unit-tested, no `eval`, no user Python):

| id | score in [0, 1] |
|---|---|
| `exact_match` | normalized answer == reference (case, spaces, punctuation folded) |
| `numeric_tolerance` | first number in the answer within `tolerance` (default 1e-2 relative) of the reference |
| `json_field_f1` | field-level F1 of a JSON answer vs the reference (the CORD scorer, reused) |
| `format_regex` | answer fully matches a user regex (rules below) |

`format_regex` safety (stdlib `re` has no timeout; no new dependency in two images for 1.0): at most 200 chars,
checked at validate time to have no nested quantifiers (`(a+)+`, `(a*)*`…), no backreferences and no lookarounds,
compiled once; the answer it is matched against is cut to 2,000 chars (answers are short: `max_new_tokens`).
A pattern that breaks a rule is a param error with a plain message (cleancode: params model).

Total reward = `format_weight` × format bonus (answer has the expected shape: a number / a JSON object / matches
the regex) + `correctness_weight` × correctness, defaults 2.0 / 5.0 as in the DGX Spark playbook.

## 4. Engine and recipe

- Image: NeMo-RL v0.7.0 arm64, pinned by digest in IMAGES.lock after security vets it (PM approved the pull).
- The worker renders a NeMo-RL GRPO config from a per-model template in `configs/train/templates/grpo/`
  (research writes the first one from the probe): `num_generations` (default 4), `max_new_tokens`, `kl_coef`,
  LoRA rank 8–64, learning rate, epochs.
- **No engine Advanced for GRPO in 1.0**: P016 §3's machinery probes AutoModel only (`configs/engine/automodel-*`);
  NeMo-RL would need its own probe and engine file. Only the node's own params (cleancode's list, including the
  reward weights as node fields) are offered.
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
2. The adapter NeMo-RL writes is read by our existing Merge / Export / Eval (PEFT format, module names): a smoke
   GRPO 0.8B → merge → eval. If the format differs, a converter (or Merge support) is part of this work.
3. Node spec (cleancode) + worker + rewards + tests; the reference pipeline R5 (DGX Spark GRPO) rebuilt with our
   nodes and run once on GB10 with its numbers recorded.
4. Security review of the job (image, mounts, regex rules).

If 1 fails within the PM's time box: the node ships as "sắp có" (disabled, real reason shown), all of this moves
to 1.1, and DPO joins it there.
