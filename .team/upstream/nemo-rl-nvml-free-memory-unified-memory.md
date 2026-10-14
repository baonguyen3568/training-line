# Upstream issue draft — NeMo-RL GRPO fails on unified-memory GPUs (DGX Spark GB10): NVML has no free-memory info

- **Status: DRAFT, not posted.** Posting needs PM/user approval. Found 2026-10-14 by bao-research (GRPO probe #3).
- Where: https://github.com/NVIDIA-NeMo/RL/issues/new — NeMo-RL v0.7.0 container (nvcr.io/nvidia/nemo-rl@sha256:b0732799…),
  1x GB10, driver 580.142, `cluster.gpus_per_node=1`, vLLM colocated.

## Facts
- `examples/run_vlm_grpo.py` (Qwen3.5-0.8B, LoRA, geometry3k) initialises vLLM colocated fine (load, compile, CUDA graphs,
  sleep frees 9.76 GiB), then dies at step 1: `DTensorPolicyWorkerV2.get_free_memory_bytes()` →
  `nemo_rl/utils/nvml.py:get_free_memory_bytes` → `pynvml.NVMLError_NotSupported` → `RuntimeError: Failed to get free
  memory for device 0`. Caller: `nemo_rl/algorithms/grpo.py:1910` (refit buffer = free × NRL_REFIT_BUFFER_MEMORY_RATIO).
- On GB10 NVML reports no framebuffer memory (`nvidia-smi` shows "Not Supported"); `torch.cuda.mem_get_info()` works.

## Our local patch
`third_party/patches/nemo-rl/0001-nvml-free-memory-fallback-for-unified-memory.patch`: on `NVMLError_NotSupported`
return `torch.cuda.mem_get_info(device_idx)[0]`. We also set `NRL_REFIT_BUFFER_MEMORY_RATIO=0.1` because "free" on a
unified-memory system is shared host RAM.

## Ask
Fall back to `cudaMemGetInfo` when NVML memory queries are unsupported (also DGX Spark playbook targets).
