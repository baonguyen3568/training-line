---
name: research-stack-verify
description: Use when choosing or upgrading any NVIDIA stack component (NeMo AutoModel, Megatron-Bridge, NeMo-RL, ModelOpt, TensorRT-LLM, vLLM, NeMo Evaluator, NeMo Curator, NGC container tag) for this project — verifies latest version, arm64/GB10 (sm_121) support and pins it with evidence before it goes into a proposal/ADR.
---

# Verify an NVIDIA stack component for DGX Spark (GB10, aarch64, CUDA 13, UMA 128GB)

Goal: never put a version into a proposal/ADR that nobody has checked against *this* machine.

## 1. Gather evidence (sources, newest first)
1. Release notes / changelog: GitHub `NVIDIA-NeMo/<repo>/releases`, `NVIDIA/TensorRT-Model-Optimizer` (now `NVIDIA/Model-Optimizer`), `NVIDIA/TensorRT-LLM/releases`.
2. NGC catalog tag page: `catalog.ngc.nvidia.com/orgs/nvidia/containers/<name>` → tag list, check **arm64 / linux/arm64** in the "Architecture" column.
3. DGX Spark playbooks: `build.nvidia.com/spark/<playbook>` and `github.com/NVIDIA/dgx-spark-playbooks` (what tag they actually pin, memory numbers).
4. Docs support matrix (supported models list for VLM, e.g. AutoModel `examples/vlm_finetune/`).
5. NVIDIA developer forum "DGX Spark / GB10" for known issues.
Record each fact with URL + access date. Mark anything not confirmed by an official source as `UNVERIFIED`.

## 2. Check locally (read-only, no GPU lock needed)
```bash
uname -m                                # must be aarch64
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'nvcr|vllm|nemo|tensorrt'
docker manifest inspect nvcr.io/nvidia/<name>:<tag> | grep -A2 '"architecture"'   # has arm64?
free -g                                 # UMA budget; nvidia-smi shows "Not Supported" for memory on GB10
```
Check CUDA/driver baseline of an arm64 NGC image WITHOUT pulling (reads config blob from nvcr.io):
```bash
scripts/research/ngc_image_env.sh nvidia/nemo-automodel 26.08   # -> CUDA_VERSION, CUDA_DRIVER_VERSION, torch
```
Host driver is **580.142 (CUDA 13.0)**. Measured 2026-10-12: CUDA 13.0 images run native; CUDA 13.1 images run via
"CUDA Forward Compatibility mode ENABLED"; CUDA >= 13.2 images (driver baseline 595+) reported NOT to work on 580
(forum #379168) — must be smoke-tested before pinning. Driver upgrade = system change → PM/user only.

Inside a container (only when needed, name it `vlm-probe-*`, `--rm`):
```bash
docker run --rm --gpus all --name vlm-probe-x nvcr.io/nvidia/<name>:<tag> \
  python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"
```
Expect capability `(12, 1)`. If a wheel has no sm_121/sm_120 kernels → note PTX JIT fallback or failure.

## 3. Decision record (paste into proposal)
| Component | Version/tag | Digest | arm64 | GB10 tested? (by whom, date) | Source |
Rules:
- Prefer NGC container with arm64 manifest over pip install on host.
- Pin tag **and** digest (`docker inspect --format '{{index .RepoDigests 0}}'`).
- Pulls of images ~20-30GB: tell `bao-cleanup`; models/datasets >20GB: ask PM first.
- If a smoke run is needed → ask `bao-backend` (or take `.team/GPU_LOCK` yourself per CLAUDE.md).

## 4. Red flags (escalate to PM / write in proposal "Risks")
- No arm64 image → building from source on aarch64 (flash-attn, TE, deepspeed builds can take hours).
- Requires `trust_remote_code=True` → loop in `bao-security`.
- Docs "latest" page lags nightly — check both.
- Docker `--memory` does not cap CUDA allocations on UMA → memory guard must be in code.
