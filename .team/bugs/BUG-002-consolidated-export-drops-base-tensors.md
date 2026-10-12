# BUG-002: consolidated export lacks 3 tensors of the Nemotron-Parse base checkpoint
Severity: major for downstream runtimes (HF remote / vLLM / export load RADIO's normalization as 0 -> divide by zero);
harmless on the native AutoModel path (recomputed in __init__)
Found by: bao-tester (O-1b parity, run 20261012-1930-o1b-parity-ae1a, image 26.08-4)   Date: 2026-10-12
Component: train export (AutoModel consolidated save)   Owner: bao-fixcode
Status: FIXING (code ready, wiring into app/workers.py + parity re-run pending)

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
