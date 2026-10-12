---
name: fixcode-uma-oom
description: Use when a train/quantize/eval/serve job on the shared DGX Spark (GB10, 128GB unified memory) hits CUDA OOM, gets OOM-killed, makes the host sluggish/frozen, or when sizing memory before a heavy run — measure the real budget (not nvidia-smi), find what consumes memory, and apply safe mitigations without touching other services.
---

# UMA memory / OOM debugging on DGX Spark

Background: `.team/knowledge/fixcode.md` §2.2. Key facts:
- `nvidia-smi` memory = "Not Supported" (expected). `torch.cuda.mem_get_info()` UNDER-reports (ignores reclaimable cache/swap).
- CPU RAM, GPU tensors, page cache, other containers all share ONE pool. Host already uses ~70-80GB → our budget ≈ 35-40GB.
- Severe pressure may **freeze the host** instead of raising OOM. Never "just retry with bigger batch".

## 1. Measure the real budget (before every heavy run)
```bash
grep -E '^(MemAvailable|Cached|SwapFree|SwapTotal):' /proc/meminfo
free -g; cat .team/GPU_LOCK 2>/dev/null
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' | sort -k2 -h | tail   # read-only view of neighbours
```
Budget = `MemAvailable` − 8GB safety margin. If budget < planned peak → do NOT start; tell PM.

## 2. Watch during run (separate terminal)
```bash
watch -n 2 "grep -E 'MemAvailable|SwapFree|Cached' /proc/meminfo; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
```
In-process: log `torch.cuda.max_memory_allocated()`, `memory_reserved()` per N steps; on OOM dump
`torch.cuda.memory._record_memory_history(max_entries=100000)` → `_dump_snapshot("outputs/debug/<id>/mem.pickle")`
(view at https://pytorch.org/memory_viz). Abort cleanly if `MemAvailable < 8GB`.

## 3. Find the consumer
| Peak shape | Likely cause | Mitigation (our side only) |
|---|---|---|
| Spike at load | full-precision load then cast; safetensors mmap → page cache | `torch_dtype=torch.bfloat16`, `low_cpu_mem_usage=True`, `device_map="cuda"`; load once |
| Grows per step | activations; long seq; many image tiles | gradient checkpointing; cap `max_pixels`/tiles; smaller micro-batch + grad accum; filter overlong samples |
| Fragmentation (reserved ≫ allocated) | allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Optimizer step spike | fp32 Adam states (full FT) | LoRA/QLoRA; 8-bit Adam (bitsandbytes); paged optimizers |
| DataLoader workers balloon RSS | worker copies of dataset, big prefetch | fewer workers, `prefetch_factor=2`, lazy decode, `--shm-size`/`--ipc=host` |
| vLLM/serve eats everything | `gpu_memory_utilization` is fraction of whole 128GB pool | `≤0.25`, `--max-model-len` minimal, `--max-num-seqs ≤4`, `--enforce-eager` (-~13GB) |
| Startup transient +20-30GB | Triton/FlashInfer/Inductor JIT (no prebuilt sm_121) | image with prebuilt kernels; disable compile; warm cache once |
| Eval accumulates | keeping logits/tensors in Python lists | `.detach().cpu()` / `.item()`, `torch.no_grad()`, `del` + `gc.collect()` |
| Cached grows while reading data | page cache from dataset/ckpt reads | stream data; avoid re-reading huge ckpts. `drop_caches` needs sudo → ask PM, never run yourself |

Quick estimates (bf16): weights = 2B/param; LoRA train ≈ weights + activations + small; full FT Adam ≈ 16-18B/param → full FT >2B params is unrealistic within ~35GB.

## 4. Do NOT
- Kill/stop processes or containers not prefixed `vlm-` / not ours. Report to PM instead.
- `sudo` sysctl/swap/drop_caches; changing docker daemon settings.
- Rely on `docker --memory` as protection — reports show freeze still happens.

## 5. Report
Write findings in the bug file: budget at start, peak (`max_memory_allocated`, MemAvailable min), consumer, mitigation, before/after numbers.
