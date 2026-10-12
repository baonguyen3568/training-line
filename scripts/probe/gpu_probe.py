"""Minimal GPU environment probe for an NGC container (no training).

Checks CUDA availability, device info, and runs a bf16 matmul + SDPA. Prints one JSON line.
"""

import json
import os
import time

import torch
import torch.nn.functional as F

out: dict[str, object] = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if out["cuda_available"]:
    props = torch.cuda.get_device_properties(0)
    out.update(
        device=props.name,
        capability=f"{props.major}.{props.minor}",
        arch_list=torch.cuda.get_arch_list(),
        driver=os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip(),
    )
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):  # warmup: cuBLAS heuristics / lazy init
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out["matmul_bf16_tflops"] = round(20 * 2 * 4096**3 / dt / 1e12, 2)
    out["matmul_finite"] = bool(torch.isfinite(c).all().item())
    q = torch.randn(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16)
    o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    out["sdpa_ok"] = bool(torch.isfinite(o).all().item())
    out["max_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
for mod in ("nemo_automodel", "transformers", "transformer_engine"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001 - probe reports any import failure
        out[mod] = f"IMPORT_FAIL: {type(e).__name__}: {e}"[:200]
print(json.dumps(out))
