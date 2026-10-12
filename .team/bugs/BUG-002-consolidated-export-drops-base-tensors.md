# BUG-002: consolidated export lacks 3 tensors of the Nemotron-Parse base checkpoint
Severity: major for downstream runtimes (HF remote / vLLM / export load RADIO's normalization as 0 -> divide by zero);
harmless on the native AutoModel path (recomputed in __init__)
Found by: bao-tester (O-1b parity, run 20261012-1930-o1b-parity-ae1a, image 26.08-4)   Date: 2026-10-12
Component: train export (AutoModel consolidated save)   Owner: bao-fixcode
Status: FIXED, pending bao-tester confirmation (7b671f2 + generation config; wired by bao-backend in RealWorker and smoke CLI)

## Symptom
Strict load of `checkpoints/LATEST/model/consolidated` against NVIDIA's checkpoint: 665 vs 668 keys. Missing:
`encoder.model_encoder.radio_model.input_conditioner.norm_mean`, `...norm_std` (F32 [3,1,1]),
`decoder.embed_positions.weight` (F32 [20002,1024]). NVIDIA's modeling code then has norm_mean = norm_std = 0.

## Root cause (bao-fixcode, verified on 26.08-5)
- RADIO `radio_model.make_preprocessor_external()` (C-RADIOv2-H radio_model.py:130) REPLACES `input_conditioner` with
  `nn.Identity()`; AutoModel's Parse encoder calls it at the first forward (processor already normalizes). The two
  buffers leave the live state_dict, and AutoModel's consolidated save keeps only live keys
  (`_maybe_build_consolidated_index`: base keys - model keys are dropped).
- `decoder.embed_positions.weight`: in NVIDIA's file but neither the native class nor NVIDIA's
  hf_nemotron_parse_modeling.py has the module (native state_dict 667 keys incl. the 2 buffers before externalizing).

## Fix
`src/vlm_pipeline/train/ckpt_complete.py` `complete_from_base(base, export)`: copy base tensors missing from the export
byte-for-byte into `model-vlm-restored-from-base.safetensors`, update the index, skip tied aliases (`lm_head.weight`
when tie_word_embeddings), never overwrite trained tensors, idempotent; then gate export ⊇ base.
Tests: `tests/unit/train/test_ckpt_complete.py` (9).
Real checkpoint (copy at outputs/fixcode/bug002/consolidated): 3 keys restored, torch.equal to base, index 668 keys.

## Verification pending
- Wire after training in app/workers.py (bao-backend) so every downstream node gets the complete export.
- bao-tester parity script on the completed copy: expect strict_load_ok true, remote norm_mean/std = CLIP values.

## Also: generation_config.json (bao-tester, same class of bug)
The consolidated save rewrites generation_config.json from the model config and drops NVIDIA's decoding defaults
(do_sample, forced_eos_token_id 2, max_new_tokens 9000, num_beams 1, repetition_penalty 1.1) -> looping decode from the
export (O-1b sample #5). `complete_from_base` now restores them: base wins except `use_cache` (patch 0003 needs False)
and `transformers_version`. CLI for exports made elsewhere: `python -m vlm_pipeline.train.ckpt_complete BASE EXPORT`.

## Verification (2026-10-12, 26.08-5, GPU via jobrun)
- bao-tester's parity script on the completed copy (run 20261012-1939-bug002-parity-8068): remote norm_mean/std =
  CLIP values (were 0), logprob diff 0.0, argmax 100 %, parity_ok true. strict_load_ok false ONLY for
  `unexpected_keys: [decoder.embed_positions.weight]`.
- Same script on NVIDIA's own base dir (run 20261012-1941-bug002-parity-base-91f3): identical load report. So the
  export now loads exactly like the base; the right gate is "load report of export == load report of base", not {}.
- transformers GenerationConfig.from_pretrained on the completed copy: repetition_penalty 1.1, forced_eos 2,
  max_new_tokens 9000, num_beams 1, do_sample False, use_cache False.
