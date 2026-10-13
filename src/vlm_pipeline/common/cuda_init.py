"""Create the CUDA context FIRST in a GPU job, before any large CPU-side allocation (GB10 workaround).

Measured on GB10 / driver 580.142 (2026-10-13, KB fixcode §2.2b): a process that first materializes ~2 GB of tensors on
the CPU (a checkpoint loaded by HF / safetensors) and only then touches CUDA fails to create its primary context about
half the time (``CUDA_ERROR_OUT_OF_MEMORY from cuDevicePrimaryCtxRetain``, shown by torch as "CUDA error: out of
memory") with 30+ GB free. The same process creating the context first never failed (0/5 against 5/9). The driver-side
mechanism is unknown; this only orders the work so the job does not hit it.

Container-side helper (imports torch lazily); call it on the first line of a GPU job's entrypoint.
"""

from __future__ import annotations

ATTEMPTS = 3
PAUSE_S = 5.0


def ensure_cuda_context(*, attempts: int = ATTEMPTS, pause_s: float = PAUSE_S) -> bool:
    """Create the primary CUDA context now (a 1-element allocation); False when no GPU is visible.

    A rare failure even this early is retried in-process (``attempts`` x ``pause_s``) before the job gives up; the
    executor's one load-time retry still covers a job that never gets a context. NOT verified: torch may cache a
    lazy-init failure and re-raise it without calling the driver again, in which case the extra attempts only cost
    ``pause_s`` each and the executor retry is the real safety net (tester).
    """
    import time

    import torch

    if not torch.cuda.is_available():
        return False
    for attempt in range(1, attempts + 1):
        try:
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            return True
        except RuntimeError as exc:  # torch.AcceleratorError is a RuntimeError
            if attempt == attempts:
                raise
            print(f"CUDA context not created ({str(exc).splitlines()[0]}), retry {attempt}/{attempts - 1}", flush=True)
            time.sleep(pause_s)
    return False  # pragma: no cover - the loop returns or raises


__all__ = ["ATTEMPTS", "PAUSE_S", "ensure_cuda_context"]
