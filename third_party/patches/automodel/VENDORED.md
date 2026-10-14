# Patches applied to NeMo AutoModel inside `vlm-automodel` images

Policy: P004 §8.3 — minimal patches, never a fork; every changed block starts with a `PATCH(vlm)` comment (or the
line carries it) so `grep -n 'PATCH(vlm)'` finds every change on rebase; code under
`/opt/Automodel` is not reformatted. Owner: bao-fixcode. Build: bao-backend (`docker/automodel.Dockerfile`).

| | |
|---|---|
| Upstream | https://github.com/NVIDIA-NeMo/Automodel (Apache-2.0, LICENSE kept in the image at `/opt/Automodel/LICENSE`) |
| Base inside the image | `/opt/Automodel` git `922e83e4207c6aaa0ae1654c4621975dffac6849` (branch r0.6.0, 2026-08-25) of `nvcr.io/nvidia/nemo-automodel:26.08` @ `sha256:70b4e9df…7b55a` (arm64). **Not** `main`: `main@829445d5` differs only in type-hint style for this file. |
| Apply | `cd /opt/Automodel && sha256sum -c PATCHES.sha256 && patch -p1 --forward --dry-run < 0001-…patch && patch -p1 --forward < 0001-…patch` — wired into `docker/automodel.Dockerfile` by bao-backend for `vlm-automodel:26.08-2`; the build must fail if it does not apply cleanly |
| Checksums | `PATCHES.sha256` |

## 0001-nemotron-parse-normalize-flag-and-tied-embeddings.patch
File: `nemo_automodel/components/models/nemotron_parse/model.py`

1. **Encoder normalize flag dropped → pixels normalized twice (all Parse versions, silent accuracy loss).**
   `NemotronParseConfig.__init__` builds the encoder config from C-RADIOv2-H and copies the Parse `encoder` keys
   only `if hasattr(self.encoder, key)`. `RADIOConfig` has no normalize attribute, so `image_processor_normalizes`
   was dropped even for v1.1; `RadioWithNeck` then kept the RADIO input conditioner while the image processor
   (`do_normalize: true`) had already normalized. v1.2/2.0 additionally renamed the flag to `processor_normalizes`.
   Fix: carry the flag explicitly; accept both names (new wins, warn on disagreement) — `_resolve_processor_normalizes`.
   Measured 2026-10-12 in vlm-automodel:26.08-1 (offline, config only): v1.1 flag seen by the model = `False`.
2. **Parse 2.0 rejected (`TieSupport.UNTIED_ONLY`) although its checkpoint is tied.** 2.0 has
   `tie_word_embeddings: true` and no `lm_head.weight` in `model.safetensors` (only `decoder.embed_tokens.weight`
   [72256, 1024]); its reference `hf_nemotron_parse_modeling.py` uses
   `_tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}`. Fix: `TieSupport.BOTH`, the same
   `_tied_weights_keys`, and a `tie_weights()` override that first calls `super().tie_weights(...)` and then ties
   `lm_head.weight` to `decoder.embed_tokens.weight` when the config asks for it (transformers v5 does not reliably
   tie from the dict-shaped keys alone; cf. AutoModel's `Qwen2ForCausalLM`). `post_init()` calls it. v1.x untied.

Not patched here (by design): `NemotronParseLoss.class_token_start_idx` — resolved from the model's `config.json` in
our own adapter, not by editing NVIDIA code.

Regression tests: `tests/integration/automodel/test_nemotron_parse_patch.py` (10 tests, marker `integration`).
Verified 2026-10-12: **10 FAIL** on vlm-automodel:26.08-1, **10 PASS** with the patched file.
Still to do (tester, nightly, GPU lock): white-image probe at the RADIO patch-embed input and golden logits vs the
reference implementation for v1.1 / v1.2 / 2.0; tied-weights save/load round trip with real weights.

Upstream: draft issue `.team/upstream/automodel-nemotron-parse-normalize.md` (ON HOLD until rewritten with the
root cause above; the user posts it). The patch may be offered as a PR later — the user decides.
Remove this patch when the AutoModel version in our image contains an equivalent fix (re-run the regression tests
against the unpatched image first).

## 0002-nemotron-parse-freeze-backbone-and-trainable-share-guard.patch (on top of 0001)
Files: `nemo_automodel/components/utils/model_utils.py`, `nemo_automodel/components/models/nemotron_parse/model.py`

P-1 incident (2026-10-12, 001a §9): Parse v1.1 full SFT trained **100 %** of 936.3M params and drove host
MemAvailable to 6.34 GB.
1. **`freeze_vision_tower` matched nothing in Nemotron-Parse.** `apply_parameter_freezing` freezes by attribute
   `vision_tower` / names containing vision|visual|image_encoder; the Parse backbone is `encoder.model_encoder`.
   Fix: models may declare `_vlm_frozen_vision_prefixes`; Parse declares `("encoder.model_encoder.",)`, matched at
   a module boundary (survives `model.` / FSDP prefixes); a declared prefix that matches nothing is an error.
   Plan A (bao-research): RADIO backbone frozen, neck + decoder + embeddings + head train ->
   v1.1 **30.40 %** (284.6M / 936.3M), 2.0 **27.84 %** (251.4M / 903.0M, tied head). Frozen backbone also drops
   the activations of 32 RADIO blocks that activation checkpointing never covered.
2. **Unknown `freeze_config` keys were ignored** (e.g. `freeze_embeddings`): now a `ValueError`.
3. **Trainable share guard**: `print_trainable_parameters` raises `VLM_TRAINABLE_PCT_MISMATCH` when the share is
   outside `VLM_EXPECTED_TRAINABLE_PCT="<lo>:<hi>"` (set by our runner from the recipe catalog; unset = no check).
   A malformed value (not two numbers, `lo > hi`, outside 0..100) is a `ValueError` naming the variable, so a typo
   cannot disable the guard (regenerated 2026-10-12 after bao-cleancode's review; prefix match is now
   `f".{p}" in f".{pname}"`, same behaviour).

Not a bug (measured, so not patched): RADIO attention is already `fused_attn=True` (timm default with SDPA), and the
Parse label mask is correct (prompt and padding masked, target + EOS learned) despite the "BPE pattern-match"
warning — both locked by tests.
Regression tests: `tests/integration/automodel/test_nemotron_parse_freeze.py` — 4 FAIL on 26.08-2 (behavioural), all
PASS with 0001+0002; the 5 malformed-env cases FAIL on 26.08-3 (old 0002) and PASS with this 0002.

## 0003-nemotron-parse-generate-without-kv-cache.patch (on top of 0001+0002) — BUG-001
File: `nemo_automodel/components/models/nemotron_parse/model.py`
`NemotronParseDecoder` ignores the KV cache (layers get `past_key_values=None, use_cache=False`, it returns
`past_key_values=None`). `generate()` with the default `use_cache=True` fed only the newest token at position 0 each
step -> `<x_0.3906>` repeated forever (golden gate (b) on 26.08-2). Fix: `generation_config.use_cache=False` by
default and `prepare_inputs_for_generation` always feeds the whole decoder sequence (O(n^2), correct). Proper KV-cache
support is upstream work (tech debt, see `.team/BOARD.md` "KV cache for the Parse decoder"). Cost: each step
re-runs the decoder over the whole prefix; measured at golden gate (b) on 26.08-5 (GB10, bf16): 28 tokens in 2.45 s warm
(~11 tok/s), max_alloc 2.21 GiB; forced lengths (min_new_tokens, same image): 256 tok 5.85 s,
512 tok 10.66 s, 1024 tok 24.07 s (x2.26 for 512->1024; memory flat at 2.21 GiB). Decision TD-1 (bao-fixcode): keep
no-cache for eval v1 (~1 h per 100 CORD pages); revisit when pages exceed ~2k tokens or eval exceeds its budget. v2 (2026-10-12, golden gate (b) on 26.08-4 failed with `model_kwargs are not used by the model: ['attention_mask']`): the override must keep the parent's explicit parameters (`attention_mask`, `inputs_embeds`, `is_first_iteration`), because `generate()` validates caller kwargs against this signature and Parse's `forward` only takes `decoder_attention_mask`. Tests: `test_nemotron_parse_generate.py` (v1: 2 FAIL on 26.08-3; v2: 2 more FAIL on 26.08-4 with the exact golden error; all 4 PASS patched); end-to-end
proof = golden gate (b) on GPU.

## 0004-nemotron-parse-nvidia-decoder-prefix-and-positional-labels.patch — decoder prefix parity
File: `nemo_automodel/components/datasets/vlm/collate_fns.py` (`nemotron_parse_collate_fn`)
Training let the tokenizer add its BOS: decoder input `[0, 2, 0, 50004, 50008, 50001, ...]`, while NVIDIA's reference
(README, test_golden.py, decoder_start_token_id = 2) uses `add_special_tokens=False`:
`[2, 0, 50004, 50008, 50001]` (bao-tester; decision 001a §10.5: fix training, not serving). Fix: encode
`task_prompt + target + </s>` with `add_special_tokens=False` and build labels by position (mask the prompt from the
first real token, mask padding) instead of the chat-template BPE fallback. Because the mask is positional, the collate
raises `ValueError` if a sample does not start with the task prompt ids (review bao-cleancode); the prompt is encoded
with `padding=False` (Parse 2.0's `tokenizer.json` enables padding to 9000). Tests: `test_nemotron_parse_prefix.py`
(3 FAIL on 26.08-3, PASS patched) + bao-tester's `test_parse_prefix_parity.py` (xfail removed); label lock tests
unchanged and green. Measured: the prompt ids are `[2, 0, 50004, 50008, 50001]` in v1.1 **and** 2.0 (vocab 72249),
and stay the same when joined with a target.

**Known limitation (bao-security review):** the target text is tokenized with special tokens enabled, so a target
containing the literal strings `</s>`, `<s>`, `<predict_bbox>`, … becomes those special tokens (a fake EOS / task
token inside the label). CORD/synthetic data never contains them; for user datasets, tokenize the target with
`split_special_tokens=True` or reject/escape these strings at data validation (owner: data validate node).

## 0005-nemotron-parse-externalize-radio-conditioner-at-build.patch (on top of 0001–0004) — BUG-003
File: `nemo_automodel/components/models/nemotron_parse/model.py` (`RadioWithNeck.__init__`)
With `image_processor_normalizes=True` the RADIO input conditioner (`norm_mean`/`norm_std` buffers) was swapped for
`nn.Identity` at the **first forward** (`_externalize_radio_input_conditioner`). Training checkpoints are saved after
that, so the model shard has 665 keys; a model built for resume has not run a forward yet and asks for 667: DCP's
strict load fails with `Missing key in checkpoint state_dict: ...input_conditioner.norm_mean` (resume smoke
2026-10-13, run `20261013-0851-resume-b2-a93e`). Fix: externalize once in `__init__` (the forward call stays, it is
idempotent). The training state_dict is then the same before and after any forward; the base checkpoint's
conditioner keys are not requested (DCP loads only the model's keys; `allow_checkpoint_key_subset` is never set), and
the math is unchanged (the image processor normalizes, the conditioner was Identity from step 1 anyway). Checkpoints
saved by 26.08-5 (665 keys) become resumable with it: fresh patched model keys == shard keys, measured on
`20261013-0849-resume-b1-0335/checkpoints/epoch_0_step_1`. HF exports still need `ckpt_complete` (BUG-002) because
NVIDIA's remote class expects the conditioner buffers. Tests: `test_nemotron_parse_resume.py` (1 FAIL on 26.08-5,
3 PASS patched, incl. the base file carrying the 2 keys); end-to-end proof on the rebuilt image, BOTH required
before bumping the default image (review bao-cleancode/bao-backend): (a) a fresh run from the base (the base load
now meets 2 extra file keys; any "unexpected keys" warning is expected and must be recorded here), (b) the
FusedAdam resume smoke from the 26.08-5 checkpoint. Run ids go into `.team/bugs/BUG-003-*.md`.
Result on 26.08-6 (`sha256:dc9da48f…`, bao-backend 2026-10-13): (a) `20261013-0924-resume6-a-6da1` and (b)
`20261013-0928-resume6-b2-6b5b` both completed with no "unexpected"/"missing key" log line; resumed weights and loss
within run-to-run noise (details in BUG-003).

## 0006-qlora-keep-tied-lm-head-and-recipe-skips-unquantized.patch
Files: `nemo_automodel/_transformers/model_init.py`, `nemo_automodel/components/quantization/qlora.py`

QLoRA (nf4) on Qwen3.5 VLM failed at step 0 (bao-research 2026-10-14, run `20261014-0842-q35-2b-qlora-nf4-ab-cd64`,
draft `.team/upstream/automodel-vlm-qlora-tied-lm-head.md`): `AssertionError` in bitsandbytes
`fix_4bit_weight_quant_state_from_module` from `self.lm_head(...)`.
1. **Root cause: the streaming BnB loader did not skip what HF's own BnB loader never quantizes.** On a single GPU,
   AutoModel loads QLoRA through `_init_model_bnb_streaming` (meta skeleton → `replace_with_bnb_linear` → shard
   streaming). Its skip list was `llm_int8_skip_modules` or `_keep_in_fp32_modules` only; HF's loader always adds
   `get_keys_to_not_convert` (tied weights, output embeddings, last module). Qwen3.5 has `tie_word_embeddings: true`,
   so `lm_head` became `Linear4bit` and `tie_weights()` then gave it the bf16 `embed_tokens.weight`. Fix:
   `_bnb_modules_to_not_convert()` = recipe skips + `_keep_in_fp32_modules` + HF's defaults, always (a tied output
   embedding can never be 4-bit, whatever the recipe lists).
2. **`create_bnb_config` dropped `llm_int8_skip_modules`** from the recipe's `quantization:` block, so a VLM could not
   keep its vision tower / projector in bf16. Fix: passed through for 4-bit (None keeps the defaults; 8-bit untouched).
   Our recipes set `llm_int8_skip_modules: [model.visual]` for Qwen3.5 (HF matches a prefix followed by `.`; the
   merger/projector lives under `model.visual`).

Regression tests: `tests/integration/automodel/test_automodel_qlora_skip.py` (2 tests, CPU, meta device, real
Qwen3.5-0.8B config). Verified 2026-10-14: **2 FAIL** on vlm-automodel:26.08-6 (lm_head `Linear4bit`; vision tower
`Linear4bit` despite the recipe skip), **2 PASS** with the patched files bind-mounted. `patch -p1 --forward --dry-run`
clean on the 26.08-6 tree. Still to do: research's QLoRA vs LoRA A/B on the patched image (GPU).
Remove when AutoModel's streaming loader adds HF's default skips upstream.

## 0007-freeze-config-block-prefixes.patch
Files: `nemo_automodel/components/utils/model_utils.py`, `nemo_automodel/_transformers/infrastructure.py`

Per-block training for ADR-022 #4 (two-stage LLaVA-style: stage 1 projector only, stage 2 projector + LoRA) and #5
(vision tower unfrozen at its own LR) (bao-research 2026-10-14). The existing `freeze_config` keys freeze by name
pattern only (Qwen keeps its projector `merger` inside `visual`; LLaVA-style `lm_head` sits outside `language_model`).
1. `freeze_config.freeze_prefixes` / `train_prefixes`: module paths matched as whole module-name runs anywhere in the
   parameter name (`param_matches_prefix`: wrappers such as `_orig_mod.` still match, `merger2` never matches
   `merger`; a short name matches every module of that name, so recipes give full paths). `train_prefixes` applies
   last and wins. Parameters are walked with `remove_duplicate=False` (a tied `lm_head` matches under its own name).
   A prefix that matches no parameter raises (`RuntimeError`): a typo never trains the wrong blocks silently.
2. The PEFT pass that freezes everything not `lora_` is `freeze_non_lora_params(model, freeze_config)`; it keeps the
   `train_prefixes` parameters by `id(param)` collected with `remove_duplicate=False` (a tied lm_head is kept too).
   Without it stage 2 "projector + LoRA" was impossible.
Proven: a full-trained block next to LoRA is saved in the adapter file (`_get_peft_state_dict` takes every
`requires_grad` parameter) and `train/merge_lora.py` (peft `PeftModel.from_pretrained` + `merge_and_unload`) keeps
its trained values in the merged model.
Regression tests: `tests/integration/automodel/test_automodel_block_freeze.py` (6, CPU/meta: Qwen3.5 stage 1 = merger
only, LLaVA-OV stage 1 = projector only, wrong prefix refused, `merger2` boundary, tied lm_head through the PEFT
pass, merge keeps the trained projector). All fail on 26.08-7 (keys unknown), pass with the patch.
Review: bao-fixcode (2 rounds). Remove when AutoModel offers block-level freezing upstream.

## 0008-vlm-labels-from-processor-chat-template.patch
Files: `nemo_automodel/components/datasets/vlm/collate_fns.py`

Families outside Qwen trained on nothing (bao-research 2026-10-14, measured with the real collate on CORD): SmolVLM
(idefics3) and LLaVA-OneVision got 0 label tokens, so loss 0 / nan and a run that "completes" without learning.
1. **Root cause: assistant-turn markers were derived from the TOKENIZER's chat template**, while the collate renders
   with the PROCESSOR's. LLaVA-OV's processor renders `<|im_start|>assistant \n` (a space), its tokenizer `assistant\n`;
   SmolVLM's tokenizer template renders nothing for plain-string content. Fix: `_derive_turn_markers_from_processor`
   renders a sentinel reply through the processor's template and takes the marker tokens by character offsets (a BPE
   merge with the space before the sentinel cannot hide it); tried first, then the tokenizer path, then the old
   pattern fallback. Qwen3.5 / Gemma-4 / InternVL results unchanged.
2. **A batch with no label token is no longer silent**: `build_labels_from_template` calls `_require_labels` exactly
   once per batch: the first batch must have labels (a broken template stops the run there); later empty batches
   (an answer truncated away) are counted and warned about, and the run stops once more than 5 % of at least 20
   batches were empty (counted per process).
Regression tests: `tests/integration/automodel/test_automodel_vlm_labels.py` (labels == the answer for smolvlm,
llava_onevision, internvl, gemma4, qwen3_5; empty-batch behaviour incl. all three paths empty -> first batch raises).
On 26.08-7: smolvlm, llava_onevision and the empty-batch tests fail; with the patch all pass (13 with 0007's).
Review: bao-fixcode (2 rounds; round 1 found the double count that let a broken template pass its first batch).
Draft upstream: `.team/upstream/` (research). Remove when AutoModel derives markers from the processor upstream.
