# 023 — Nodes and gallery schema for the 10-method library (ADR-022)

- Author: bao-cleancode · 2026-10-14 · Status: DRAFT. Reviewers: research (lead, engine facts), backend (workers),
  fixcode (#6, #10), design + frontend (gallery card), security (new engines).
- Scope: node specs / params in `nodes/` + `*/config.py` (cleancode), and the schema of `configs/reference_pipelines.yaml`
  (research owns the content). Workers, recipes and engines are the owners listed in ADR-022.
- Principle: a method is a PIPELINE of existing nodes plus the smallest new knobs; a new node only when the engine job is
  different (DPO, QAT). Every new knob or node is refused by its support check until its engine path is proven
  (canvas == worker, the support invariant).

## 1. Train by block (#1, #4, #5): one param, no new node
Train nodes describe WHAT is trained with one enum instead of booleans that can contradict each other:

| param | values | meaning |
|---|---|---|
| `train_blocks` (sft) | `lm` (default, today's behaviour) · `projector` · `projector_lm` · `all` | blocks that are updated; the others are frozen |
| `vision_lr_scale` (sft, Advanced, only when `train_blocks: all`) | 0.01–1.0, default 0.1 | vision tower LR = node LR × scale (#5: "ViT low LR") |

- `freeze_vision_tower` (today's bool) becomes `train_blocks` (`true` → `lm`, `false` → `all`); old files are migrated
  by the params validator (one release), the bool is then removed. LoRA keeps adapters on the LM only (unchanged).
- Engine facts NEEDED from research before code: (a) does AutoModel 26.08 freeze the projector separately
  (`freeze_config` has vision / audio / language_model only)? (b) per-block LR = optimizer param groups: native or a patch?
  If (b) needs a patch, `vision_lr_scale` ships disabled with its reason (like nf4 did).
- #4 (LLaVA 2-stage) needs NO stage node: two `train.sft_vlm` in a row (Checkpoint feeds Model), stage 1
  `train_blocks: projector` on caption data, stage 2 `projector_lm` on instruction data. Shipped as a first-party GROUP
  (P4-4) "Train 2 giai đoạn" so it is one block in the library.

## 2. `train.dpo_vlm` (#7) — mirrors `train.grpo_vlm`
- Inputs base_model / train_data / val_data; output `adapter` (LoRA), so Merge / Eval / Export are reused.
- Data: ordinary `Dataset`; records need `chosen` + `rejected` (and `prompt`, `image`). Missing fields → DATA-008
  DATASET_NOT_FOR_NODE (shared with GRPO, P020). `data.local_folder input_format: preference` reads them.
- Params (ranges confirmed by research from the engine): `loss` (`dpo` | `mpo` if the engine has it), `beta` 0.1
  [0.01, 1.0], `epochs`, `lr_preset`, `lora_dim` (Advanced), `seed`. No engine Advanced in 1.0 (same as GRPO).

## 3. `quantize.qat` (#10) — a short fine-tune with fake quantization
- Inputs `model` (Model), `train_data` (Dataset), `calib_data` (Dataset, optional); output `quantized` (QuantizedModel),
  so the Eval / Benchmark / Export after a PTQ node work unchanged.
- Params: `format` (same enum as PTQ, same disabled options), `epochs` 1, `lr_preset` (QAT recipes use a low LR),
  `calib_size`. Shares `quantize/config.py` types with PTQ (one source).

## 4. Distillation (#6) — a pipeline, at most one new data node
- Teacher answers = `data.caption_vlm` with a task: add `task: caption | qa` (qa: the teacher writes a question AND its
  answer per image, two prompts) — or a separate `data.synth_qa` if fixcode's worker differs too much. fixcode decides;
  either way the generated target keeps `generated_by` (provenance) and is untrusted text (security §B3).
- Student = the existing LoRA / SFT nodes.

## 5. Gallery schema (`configs/reference_pipelines.yaml`, schema_version 2)
A pydantic model `common/reference_pipelines.py` (strict, loaded by a test and by the frontend exporter) — fields added
to today's per-pipeline record:

| field | type | note |
|---|---|---|
| `method` | enum of the 10 ADR-022 ids: `full_sft, lora, qlora, two_stage, vision_unfrozen, distill, dpo, grpo, ptq, qat` | one pipeline per method in 1.0 |
| `when_to_use_key` | i18n key `reference_pipeline.<id>.when` | one sentence, design writes vi/en |
| `train_arch` | `{vision, projector, lm}` each `frozen | lora | full | quantized | n/a`, + optional `stages: [{name_key, vision, projector, lm}]` | drives the card's architecture diagram |
| `source` | `{url, publisher, paper_url?}` | replaces `source_url` + `publisher` |
| `license`, `difficulty`, `measured`, `steps`, `template_id`, `status` | unchanged | `measured` rules unchanged (sources cited) |

The 5 pipelines that are about tasks (receipts, …) become examples INSIDE their method entry (`examples: [...]`) or
move to templates; research decides which. The exporter validates with the model instead of its REQUIRED list.

## 6. Open questions
1. research: engine facts of §1 (a)(b); DPO/MPO in NeMo-RL v0.7.0 for VLM on 1 GPU (shares the GRPO setup).
2. fixcode: §4 one task param or a new node?
3. design/frontend: is `train_arch` (3 blocks × 5 states, optional stages) enough for the diagram?
