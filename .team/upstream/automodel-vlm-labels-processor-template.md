# Upstream issue draft — NeMo AutoModel VLM collate builds no labels for LLaVA-OneVision and SmolVLM

- **Status: DRAFT, not posted.** Posting needs PM/user approval. Found 2026-10-14 by bao-research (ADR-021 family checks).
- Where: https://github.com/NVIDIA-NeMo/Automodel/issues/new — image vlm-automodel:26.08-7 (AutoModel r0.6.0 + local
  patches 0001–0006), transformers 5.12.1.

## Facts (CORD v2 validation, 2 samples, AutoModel's own collate functions)
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (llava_onevision_collate_fn) and `HuggingFaceTB/SmolVLM-256M-Instruct`
  (default_collate_fn): every label is -100 → the loss trains on nothing. InternVL3-1B-hf, Gemma-4-E2B, Qwen3.5-2B: fine.
- Cause, `build_labels_from_template`: markers are derived from the TOKENIZER's chat template, but the batch is rendered
  with the PROCESSOR's. LLaVA-OV's processor renders `<|im_start|>assistant \n` (space), its tokenizer `assistant\n`.
  SmolVLM's tokenizer template renders nothing for string content, so the sentinel is not found and the BPE fallback
  (`build_labels`) also returns no label.
- No error or warning reaches the user beyond a log line; the run "trains".

## Our local patch
`0008-vlm-labels-from-processor-chat-template.patch` (outputs/research/, pending fixcode review): derive markers from
the processor's template (sentinel located by character offsets) first, and raise when a batch has no label token.
