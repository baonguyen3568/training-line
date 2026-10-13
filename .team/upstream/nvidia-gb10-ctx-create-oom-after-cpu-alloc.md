# GB10: cuDevicePrimaryCtxRetain returns OUT_OF_MEMORY when the process allocated ~2 GB on the CPU first

Status: DRAFT, not posted (needs PM / user approval). Owner: bao-fixcode. Found 2026-10-13.

## Environment
- NVIDIA DGX Spark (GB10, sm_121, unified memory 121.7 GiB), driver 580.142, CUDA 13.0 (container vlm-automodel:26.08-6,
  torch 2.13), aarch64.
- Shared host: two other vLLM EngineCore processes (other containers) hold ~58 GB of GPU memory; MemFree 30-36 GB,
  MemAvailable 50+ GB at the time of the failures.

## Minimal reproduction (container, `--gpus all`)
```python
import torch
from safetensors import safe_open
with safe_open("<3.7 GB fp32 checkpoint>.safetensors", framework="pt") as f:
    ts = [f.get_tensor(k).to(torch.bfloat16) for k in f.keys()]   # ~1.9 GB materialized on the CPU
g = [t.to("cuda") for t in ts]                                    # first CUDA use -> context creation
```
- About 55% of runs (5/9) fail with `torch.AcceleratorError: CUDA error: out of memory`. With `CUDA_LOG_FILE=stderr`:
  `Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain`.
- Same script with `torch.empty(1, device="cuda")` as its FIRST line (context created before the CPU allocations): 0/5.
- Allocations of 4-20 GiB right after start, with no prior CPU load: 0/36. Whether another GPU container just exited
  does not matter (the failure reproduces with no previous job and a 45 s gap).

## Expected
Context creation succeeds when tens of GB are free, regardless of how much the process allocated on the CPU before.

## Our workaround
`vlm_pipeline.common.cuda_init.ensure_cuda_context()`: create the context on the first line of every GPU job entry
point (3 in-process attempts), plus one executor retry 120 s later for load-time CUDA OOM.
Evidence: `.team/knowledge/fixcode.md` §2.2b; scripts round7-10 (fixcode scratchpad); runs `outputs/runs/20261013-22{47..59}-oomexp{7,8,9,10}-*`.
