# Upstream issue draft — vLLM ModelOpt FP8 block-wise (`fp8_pb_wo`) cannot load a merged column layer whose shards are smaller than one block

- **Status: DRAFT, not posted.** Posting needs PM/user approval (no agent posts upstream). Found 2026-10-13 by
  bao-fixcode (root cause from the safetensors header), written up by bao-research.
- Where: https://github.com/vllm-project/vllm/issues/new (vLLM 0.31.0, image `vlm-vllm:0.31.0`, GB10 arm64).
- Our workaround (in the repo): `vlm_pipeline.quantize.ptq.BLOCK_EXTRA_SKIP` keeps `linear_attn.in_proj_a/b` BF16 in
  the block recipe; they are tiny, so nothing measurable is lost.

## Reproduction (facts we have)

- Model: `Qwen/Qwen3.5-0.8B` @2fc06364, quantized with NVIDIA ModelOpt 0.44 `FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG`
  (128x128 weight blocks), exported with `export_hf_checkpoint` → `hf_quant_config.json` `quant_algo: "fp8_pb_wo"`.
- `linear_attn.in_proj_a.weight_scale` / `in_proj_b.weight_scale` have shape `[1, 1, 8, 1]` in the exported
  safetensors (36 tensors): their output dim is < 128, so one block scale each.
- vLLM fuses `in_proj_b` + `in_proj_a` into one `MergedColumnParallelLinear` (`in_proj_ba`), sizes its block-scale
  parameter as `ceil(out_total / 128) = 1`, and loading the second shard narrows at offset 1:
  `RuntimeError: start (1) + length (1) exceeds dimension size (1)` in `parameter.load_merged_column_weight`.
- vLLM's source mentions padding non-128 outputs (#53132), but the merged-column path still fails.
- Run: `outputs/runs/20261013-1736-vllm-batch-q08-block-013e` (local, not shareable).

## Expected

Either load per-shard block scales for merged layers whose shards are not 128-aligned, or reject the checkpoint with a
clear message naming the layer.
