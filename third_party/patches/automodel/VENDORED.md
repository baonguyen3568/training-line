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
(~11 tok/s), max_alloc 2.21 GiB; a full CORD page (up to 1536 tokens) is still to be measured before eval budgets rely on it. v2 (2026-10-12, golden gate (b) on 26.08-4 failed with `model_kwargs are not used by the model: ['attention_mask']`): the override must keep the parent's explicit parameters (`attention_mask`, `inputs_embeds`, `is_first_iteration`), because `generate()` validates caller kwargs against this signature and Parse's `forward` only takes `decoder_attention_mask`. Tests: `test_nemotron_parse_generate.py` (v1: 2 FAIL on 26.08-3; v2: 2 more FAIL on 26.08-4 with the exact golden error; all 4 PASS patched); end-to-end
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
