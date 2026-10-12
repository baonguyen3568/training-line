---
name: backend-nvidia-spike
description: Use when exercising or wrapping the NVIDIA tool APIs for this project on DGX Spark — NeMo AutoModel VLM fine-tune (automodel CLI + YAML recipe), ModelOpt hf_ptq quantization (FP8/NVFP4/AWQ, VLM support matrix), vLLM / TensorRT-LLM serving — exact commands, versions verified 2026-10, GB10 pitfalls, and what to measure.
---
# NVIDIA stack cheat-sheet (bao-backend). Knowledge + sources: `.team/knowledge/backend.md`
Always pair with `backend-gpu-job` (lock + memory budget) before running anything below.

## Versions (GitHub releases, checked 2026-10-12) — re-check before pinning
AutoModel v0.6.0 (`nvcr.io/nvidia/nemo-automodel:26.08`), Megatron-Bridge v0.6.2, NeMo-RL v0.7.0,
ModelOpt 0.47.0, TensorRT-LLM v1.2.1 (`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`), vLLM v0.31.0.
```bash
for r in NVIDIA-NeMo/Automodel NVIDIA/Model-Optimizer NVIDIA/TensorRT-LLM NVIDIA-NeMo/RL vllm-project/vllm; do
  curl -s https://api.github.com/repos/$r/releases/latest | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["tag_name"],d["published_at"])'; done
```

## 1. Train — NeMo AutoModel (VLM)
- CLI: `automodel <recipe.yaml> --nproc-per-node 1 [--dotted.key value ...]` (examples/*/finetune.py is deprecated).
- Smallest VLM recipe: `examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml` (set `model.pretrained_model_name_or_path` + `processor...` to `Qwen/Qwen3.5-0.8B` or `-2B` for spikes; dataset MedPix-VQA).
- Spike overrides: `--step_scheduler.max_steps 20 --step_scheduler.global_batch_size 4 --step_scheduler.ckpt_every_steps 1000
  --checkpoint.checkpoint_dir /workspace/outputs/runs/$RUN_ID/checkpoints`.
- Keys that matter: `freeze_config.{freeze_vision_tower,freeze_language_model}`, `peft:` block for LoRA (see *_lora.yaml),
  `checkpoint.save_consolidated`, `checkpoint.max_recent_checkpoints: 2`, `model.backend.attn: sdpa` (flash-attn unreliable on aarch64).
- Measure: peak `torch.cuda.max_memory_allocated()`, host MemAvailable min, s/step, tok/s, time-to-first-step (model load + page cache).

## 2. Quantize — ModelOpt `examples/hf_ptq`
```bash
python hf_ptq.py --pyt_ckpt_path <hf_ckpt> --qformat fp8|nvfp4|nvfp4_mlp_only|int4_awq \
  --kv_cache_qformat fp8_cast|nvfp4|none --calib_size 512 --export_path <out>
# Qwen3-VL / Qwen3.5 FP8 incl. vision tower:
python hf_ptq.py --pyt_ckpt_path <ckpt> --recipe model_type/qwen3_5/ptq/fp8_vision_lm-kv_fp8_cast \
  --calib_with_images --calib_size 512 --skip_generate --export_path <out>
```
VLM support (default = LLM only, vision stays BF16): Qwen2/2.5-VL fp8+awq+nvfp4 · Qwen3-VL/3.5 fp8 only · Gemma3 fp8 · Nemotron VL fp8+nvfp4 · Llava fp8+awq.
Output = unified HF checkpoint → vLLM / TRT-LLM PyTorch backend.

## 3. Serve / perf
- vLLM: `vllm serve <ckpt> --host 127.0.0.1 --port 7871 --gpu-memory-utilization 0.3 --max-model-len 8192 --max-num-seqs 4`
  (quantization auto-detected from config). Default 0.9 would grab most of the shared 128 GB — never use it here.
- TRT-LLM: `trtllm-serve <ckpt> --backend pytorch --host 127.0.0.1 --port 7872` (VLM path on Spark not yet verified).
- On GB10, NVFP4 is NOT guaranteed faster than FP8 (kernel selection, sm_121a) → always benchmark both.

## Pitfalls
`torch.cuda.mem_get_info` under-reports on UMA (use /proc/meminfo) · CUDA-12 wheels break on CUDA 13 host → use containers ·
container runs as root → files root-owned · version string inside container may lag the tag → record digest + `pip list` in run.json.
