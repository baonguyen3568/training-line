#!/usr/bin/env bash
# Print an environment fingerprint for bug reports.
# Env vars: strict allowlist (never prefix/denylist) + value redaction, per bao-security review. HF_TOKEN is never printed.
# Usage: bash env_fingerprint.sh [python-bin]
PY="${1:-python}"
echo "== host"; date -Iseconds; uname -m
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader 2>&1
echo "== container"; { [ -f /.dockerenv ] || [ -f /run/.containerenv ]; } && echo "in-container" || echo "host"
echo "== git"; git rev-parse --short HEAD 2>/dev/null; git status --porcelain 2>/dev/null | head -5
echo "== memory"; grep -E '^(MemTotal|MemAvailable|Cached|SwapTotal|SwapFree):' /proc/meminfo
df -h /dev/shm 2>/dev/null | tail -1
echo "== gpu lock"; cat .team/GPU_LOCK 2>/dev/null || echo "free"
echo "== python"
"$PY" - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
        print("arch_list", torch.cuda.get_arch_list())
        free, total = torch.cuda.mem_get_info()
        print(f"mem_get_info free={free/2**30:.1f}G total={total/2**30:.1f}G (UNDER-reports on UMA)")
    else:
        print("cuda NOT available")
except Exception as e:
    print("torch import failed:", type(e).__name__, e)
for m in ["transformers", "accelerate", "peft", "trl", "datasets", "transformer_engine",
          "megatron.core", "megatron.bridge", "nemo_automodel", "nemo_rl", "modelopt",
          "tensorrt_llm", "vllm", "triton", "flash_attn", "qwen_vl_utils"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:20s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:20s} N/A ({type(e).__name__})")
EOF
echo "== env (allowlist only; values with credentials redacted)"
for v in CUDA_VISIBLE_DEVICES CUDA_HOME CUDA_LAUNCH_BLOCKING CUDA_CACHE_PATH NCCL_DEBUG NCCL_SOCKET_IFNAME \
         NCCL_P2P_DISABLE NCCL_IB_DISABLE PYTORCH_CUDA_ALLOC_CONF TORCH_CUDA_ARCH_LIST TORCH_NCCL_ASYNC_ERROR_HANDLING \
         TORCH_DISTRIBUTED_DEBUG TORCHINDUCTOR_CACHE_DIR TRITON_PTXAS_PATH TRITON_CACHE_DIR HF_HOME HF_HUB_CACHE \
         HF_HUB_OFFLINE HF_HUB_ENABLE_HF_TRANSFER HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE VLLM_USE_V1 \
         NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES OMP_NUM_THREADS; do
  [ -n "${!v+x}" ] || continue
  val="${!v}"
  if printf '%s' "$val" | grep -qiE '://[^/]*@|hf_[A-Za-z0-9]{8,}|nvapi-|(^|[^a-z])sk-|token|key|secret|pass|auth|cred|cookie|session'; then
    val="<redacted>"
  fi
  echo "$v=$val"
done
