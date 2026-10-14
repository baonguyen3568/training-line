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
- fixcode (engine facts, probed 2026-10-14 in `vlm-automodel:26.08-7`, no new deps): ModelOpt 0.44.0 ships
  `modelopt.torch.quantization.plugins.transformers_trainer.{QATTrainer, QADTrainer}`, `mtq.FP8_DEFAULT_CFG` /
  `NVFP4_DEFAULT_CFG`, peft 0.18.1, transformers 5.12.1. Recipe (ModelOpt `examples/llm_qat`): calibrate (= the PTQ step,
  same `SKIP_PATTERNS` scope: vision / projector / lm_head stay BF16) → train the fake-quant model at a LOW LR for a short
  run → export the same unified HF checkpoint as PTQ. Worker = `vlm_pipeline.quantize.qat` in the train image, own
  argv in `job_kinds` (`quantize` kind), not AutoModel (AutoModel has no ModelOpt QAT path).
- Knobs I need on top of the four above (all with safe defaults, Advanced unless noted):
  - `adapter: lora | full` (main panel, default `lora`): ModelOpt QAT + LoRA keeps a 2B within the GB10 envelope; `full`
    is refused by check_support above the full-SFT size limit (same rule as `train.*`).
  - `max_steps` (default 0 = by epochs): QAT needs ~10% of the SFT steps; a cap stops a big dataset from turning QAT
    into a full training run.
  - `lr_preset`: `low` = 1e-5 (default), `lower` = 5e-6. No free LR field in 1.0.
  - Not in 1.0: QAD (distill from the BF16 model = a second copy in memory), INT4/W4A8 formats.
- `format`: `fp8` enabled; `nvfp4` disabled with the PTQ reason until research's #9 check says GB10 can SERVE NVFP4
  (fake-quant training itself runs in plain PyTorch, the open question is export + serving).
- Output report = `quant_report.json` of PTQ plus `qat: {steps, lr, adapter, train_loss_first/last}`, so #10's card
  compares with #9 on the same eval.

## 4. Distillation (#6) — a pipeline, at most one new data node
- **Decision (fixcode): one param on `data.caption_vlm`, no new node.** `task: caption | qa` (default `caption`).
  Same worker, same vLLM batch job, same GPU lock / memory queue / `--network none`, same provenance and untrusted-text
  rules. The difference is small:
  - `qa`: ONE prompt per image asks the teacher for a question and its answer as JSON `{question, answer}`, enforced by
    vLLM structured output (JSON schema), not two passes (half the GPU time, no question/answer mismatch).
  - The record gets `prompt = question`, `target = answer`, `generated_by = {model, revision, task}`. Both strings are
    untrusted (stripped, length-capped, JSON-serialized). A reply that does not parse or is empty drops the record and
    is counted in `caption_report.json` (`dropped_unparsed`); none left → DATA-006.
  - `qa_focus` (Advanced, enum `general | text_reading | counting | charts`, default `general`) picks one of our fixed
    prompt templates. No free prompt for `qa` in 1.0 (the JSON contract depends on the template).
  - One QA pair per image in 1.0 (records are one-per-image today).
- Quality filtering = the existing shared filter / dedupe steps (P4-4), not a new node.
- Teacher = a catalog model in `trial_worker.VLLM_PARITY_MODELS`. Today only qwen3.5-2b; I add qwen3.5-4b (bf16, fits:
  ~28 GB est.) after tester's parity gate passes, then try qwen3.5-9b FP8 if memory allows. check_support refuses any
  model outside the parity set (CFG-011, as today); a teacher not bigger than the student is a warning (STD), not a block.
- Student = the existing LoRA / SFT nodes. Pipeline: source images → caption_vlm(task=qa, teacher 4B) → filter/dedupe →
  LoRA student 0.8B → eval; card compares with the same student on the original labels.

## 5. Gallery schema (`configs/reference_pipelines.yaml`, schema_version 2)
A pydantic model `common/reference_pipelines.py` (strict, loaded by a test and by the frontend exporter) — fields added
to today's per-pipeline record:

| field | type | note |
|---|---|---|
| `method` | enum of the 10 ADR-022 ids: `full_sft, lora, qlora, two_stage, vision_unfrozen, distill, dpo, grpo, ptq, qat` | one pipeline per method in 1.0 |
| `when_to_use_key` | i18n key `reference_pipeline.<id>.when` | one sentence, design writes vi/en |
| `train_arch` | `{vision, projector, lm}` each `frozen | lora | full | full_low_lr | n/a`, + `quant: null | nf4 | fp8 | nvfp4 | fp8_fake` (separate from the state: QLoRA is `lm: lora` + `quant: nf4`; PTQ is all `frozen` + `quant: fp8`), + `extras: [teacher | reference | rollout | reward]`, + optional `stages: [{name_key, vision, projector, lm, quant}]` | drives the card's architecture diagram (design spec I §2–3: `full_low_lr` = method #5 vision tower at a low LR; extras draw side blocks for distill/DPO/GRPO) |
| `measured_with` | `{model, data}` display names | the card says "Đo với: {model}"; the method itself is model-agnostic (ADR-021) |
| `source` | `{url, publisher, paper_url?}` | replaces `source_url` + `publisher` |
| `license`, `difficulty`, `measured`, `steps`, `template_id`, `status` | unchanged | `measured` rules unchanged (sources cited) |

The 5 pipelines that are about tasks (receipts, …) become examples INSIDE their method entry (`examples: [...]`) or
move to templates; research decides which. The exporter validates with the model instead of its REQUIRED list.

## 6. Open questions
1. research: engine facts of §1 (a)(b); DPO/MPO in NeMo-RL v0.7.0 for VLM on 1 GPU (shares the GRPO setup).
2. fixcode: §4 one task param or a new node? → **one param** (`task`), see §4.
3. design/frontend: is `train_arch` (3 blocks × 5 states, optional stages) enough for the diagram?
