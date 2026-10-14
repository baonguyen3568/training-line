# 020 — GRPO training node (`train.grpo_vlm`), worker side

- Author: bao-backend · 2026-10-14 · Status: DRAFT v3 (rewards = NeMo-RL built-ins; 1-GPU limit), gated on research's NeMo-RL probe
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
- (`json_field_f1` is not in the first version: §3.)

## 3. Rewards (built-in only, security §2) — mapped onto NeMo-RL's own

NeMo-RL v0.7.0 builds the reward in `VLMEnvironment` (`nemo_rl/environments/vlm_environment.py:66-89`, research
2026-10-14) from `env.<name>.reward_functions: [{name, weight, kwargs?}]`, a FIXED if/elif over five names:
`format`, `exact_alnum`, `math_expr`, `bbox_giou`, `geo3k`; any other name is a ValueError, and no outside module is
loaded. So the node's `reward` enum maps straight onto those names, and **no reward code of ours runs in the job**
(the safest option):

| node `reward` | NeMo-RL reward_functions | note |
|---|---|---|
| `exact_match` | `format` + `exact_alnum` | letters/digits inside `<answer>…</answer>`, case-insensitive |
| `numeric` | `format` + `math_expr` | sympy equivalence (approximate numbers, expressions) |
| `format` | `format` only | the answer has the `<think>` / `<answer>` shape |

Weights default to the engine's examples (clevr: format 0.2 + exact_alnum 0.8; geometry3k: format 0.1 +
math_expr 0.9); the node exposes them as `format_weight` / `correctness_weight`. The prompt template always tells the
model to answer inside `<answer>…</answer>` (the rewards read nothing else). `json_field_f1` and a user regex need
an environment of our own (our Python loaded into the Ray workers: a derived image and its own security review):
not in the first version.

## 4. Engine and recipe

- Image: NeMo-RL v0.7.0 arm64 (python 3.13, torch 2.11+cu130, transformers 5.8.1, vLLM in its own Ray venv),
  pinned by digest; the derived non-root `vlm-nemo-rl:0.7.0-1` is PROBE ONLY until the probe passes.
- Engine limit found by the probe: the smallest VLM GRPO recipe is 1 node × 2 GPUs (Qwen2.5-VL-3B); none for 1 GPU,
  and Qwen3.5 only as 35B-A3B on 2 × 8 GPUs. A 1-GPU GB10 run needs a colocated vLLM + policy configuration of our
  own: the reason the node ships "coming soon" in 1.0 unless the probe passes by 2026-10-15 morning (PM).
- The worker renders a NeMo-RL GRPO config from a per-model template in `configs/train/templates/grpo/`
  (research writes the first one from the probe): `num_generations` (default 4), `max_new_tokens`, `kl_coef`,
  LoRA rank 8–64, learning rate, epochs.
- **No engine Advanced for GRPO in 1.0**: P016 §3's machinery probes AutoModel only (`configs/engine/automodel-*`);
  NeMo-RL would need its own probe and engine file. Only the node's own params (cleancode's list, including the
  reward weights as node fields) are offered.
- The reward is NeMo-RL's own (§3): the worker only writes `reward_functions` names and weights into the config.
  The job stays `--network none`, narrow mounts, non-root (`vlm-nemo-rl:0.7.0-1`: derived image, `chmod 711 /root`,
  security 2026-10-14), no secret.
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
4. Security review of the job (image, mounts).

If 1 fails within the PM's time box: the node ships as "sắp có" (disabled, real reason shown), all of this moves
to 1.1, and DPO joins it there.
