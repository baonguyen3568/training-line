# BUG-001: Nemotron-Parse native AutoModel class loops when generating (KV cache ignored)
Severity: major (eval/inference with the native class produces garbage; training unaffected)
Found by: bao-backend (golden gate §8.6, 26.08-2, outputs/probe/golden-b-parse-v1.1.log)   Date: 2026-10-12
Component: train/eval (third-party: NeMo AutoModel `nemotron_parse/model.py`)   Owner: bao-fixcode
Status: CLOSED 2026-10-12 (patch 0003 v2, image vlm-automodel:26.08-5; golden gate (b) PASS)

## Symptom
Native `NeMoAutoModelForImageTextToText` + Parse v1.1: pixel/encoder/forward top-k match golden, but `generate()`
repeats `<x_0.3906>` (50409) forever from the 6th token.

## Root cause (static reading of r0.6.0 @ 922e83e4, bao-fixcode)
`NemotronParseDecoder.forward` ignores the KV cache: it calls every layer with `past_key_values=None, use_cache=False`,
uses `past_key_values_length = 0`, and returns `past_key_values=None`. `generate()` with the default `use_cache=True`
feeds only the newest token each step (it assumes history is in the cache) -> the decoder sees one token at position 0
with no context -> repetition. The `layer_idx` warning is a red herring. Training (teacher forcing on the full
sequence, no cache) is unaffected — consistent with forward matching golden.

## Confirm (needs GPU lock + weights; bao-backend has the harness)
`generate(..., use_cache=False)` with the native class must reproduce golden_outputs.json (slower: O(n^2)).

## Fix options
1. Call-site workaround in our eval/inference adapters: always `use_cache=False` for the native Parse class
   (`# WORKAROUND(BUG-001)`), or use the HF-remote / vLLM path for inference (bao-backend's interim decision).
2. Patch 0002: make the native class refuse/neutralize caching (e.g. `generation_config.use_cache=False` +
   `prepare_inputs_for_generation` passing the full decoder ids). Proper KV-cache support is upstream work.
## Verification
Golden gate (b) native: 50-token generation + text equal to golden; regression test in the container (gpu marker).

## Resolution (2026-10-12)
- f42f91c patch 0003 v1 (no KV cache: `generation_config.use_cache=False`, feed the whole decoder sequence).
- Golden gate (b) on 26.08-4 FAILED before generating: `model_kwargs are not used by the model: ['attention_mask']`.
  Regression introduced by v1: the `prepare_inputs_for_generation` override dropped the parent's explicit parameters,
  and transformers validates generate kwargs against that signature. Fixed in 3a768ca (0003 v2) + 2 tests that
  reproduce the exact error on 26.08-4.
- Golden gate (b) on 26.08-5 (bao-backend, outputs/runs/20261012-1926-golden-b-26.08-5-cf22): 24/25 PASS;
  generated token ids and decoded text match NVIDIA's golden exactly (28 tokens). The only miss is top-k order among
  near-equal logits, identical on the HF remote path -> accepted.
- Cost: 28 tokens in 2.45 s warm (3.54 s cold), ~8-11 tok/s, max_alloc 2.21 GiB. Long pages (O(n^2), CORD,
  max_new_tokens 1536) still to be measured from bao-tester's eval -> TD-1 on BOARD.
- Timing data (forced lengths 256/512/1024 tokens, VENDORED.md §0003) comes from `scripts/probe/parse_generate_timing.py`
  (native class, NVIDIA's golden image; runs in vlm-automodel on GPU under the lock).
