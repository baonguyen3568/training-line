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

Not a bug (measured, so not patched): RADIO attention is already `fused_attn=True` (timm default with SDPA), and the
Parse label mask is correct (prompt and padding masked, target + EOS learned) despite the "BPE pattern-match"
warning — both locked by tests.
Regression tests: `tests/integration/automodel/test_nemotron_parse_freeze.py` — 4 FAIL on 26.08-2 (behavioural), all
5 PASS with 0001+0002; 0001+0002 apply cleanly on the pristine NGC base.

Pending for the next regeneration of 0002 (review bao-cleancode, no behaviour change): `f".{p}" in f".{pname}"` for
the boundary match; validate `VLM_EXPECTED_TRAINABLE_PCT` format/`lo <= hi` with a clear message.
