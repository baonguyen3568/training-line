"""What GPU memory model this host has (UMA like GB10, or discrete VRAM) and the budgets/limits that follow (D-4, M3).

Stdlib only, host-safe: one ``nvidia-smi --query-gpu`` call, parsed into ``GpuInfo``. On GB10 (DGX Spark) the memory
columns read ``[N/A]``: CPU and GPU share one pool, so the host ``MemAvailable`` is the only budget (memguard). On a
discrete GPU the job needs VRAM (``memory.free``) AND host RAM, two budgets checked with the same rule as memguard
(ADR-005 A1: a job may take at most ``max_free_frac`` of what is free when it starts).

NOT tested on real discrete hardware (the team only has a GB10): the parsing follows nvidia-smi's documented CSV
output and is covered by fixtures; re-check on the first amd64/discrete machine.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

MemoryModel = Literal["uma", "discrete", "none"]
logger = logging.getLogger(__name__)
QUERY_TIMEOUT_S = 5
QUERY = "name,memory.total,memory.used,memory.free,compute_cap,driver_version"
MIB_PER_GIB = 1024
# CUDA 13.3 images (vlm-automodel 26.08) need driver >= 610 natively. Data-center GPUs (and GB10) reach older drivers
# through forward compatibility; GeForce/consumer RTX cards have none (research, multiarch-gpu.md §2).
MIN_NATIVE_DRIVER_CUDA_13_3 = 610
_NO_FORWARD_COMPAT = ("GeForce", "TITAN")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_gb: float | None  # None on UMA ([N/A]): the host pool is the budget
    used_gb: float | None
    free_gb: float | None
    compute_capability: tuple[int, int]
    driver_version: str

    @property
    def driver_major(self) -> int:
        return int(self.driver_version.split(".", 1)[0])

    @property
    def forward_compatible(self) -> bool:
        """Whether CUDA forward compatibility exists for this card (data center / GB10 yes, GeForce no)."""
        return not any(tag in self.name for tag in _NO_FORWARD_COMPAT)


def _gb(field: str) -> float | None:
    field = field.strip()
    if not field or field.startswith("[") or field.upper() in {"N/A", "NA"}:
        return None
    return float(field) / MIB_PER_GIB


def parse_query(csv_text: str) -> list[GpuInfo]:
    """Rows of ``nvidia-smi --query-gpu=<QUERY> --format=csv,noheader,nounits`` (memory in MiB)."""
    gpus = []
    for line in csv_text.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != len(QUERY.split(",")):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        name, total, used, free, cc, driver = cols
        major, _, minor = cc.partition(".")
        gpus.append(GpuInfo(name, _gb(total), _gb(used), _gb(free), (int(major), int(minor or 0)), driver))
    return gpus


def query_gpus(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[GpuInfo]:
    """GPUs seen by the driver; [] without nvidia-smi or GPU (CPU-only host).

    On every job's path (preflight): a hung or broken nvidia-smi (driver trouble) must never hang the executor, so it
    gets ``QUERY_TIMEOUT_S`` and any failure degrades to [] ("none": host budget only) with a warning.
    """
    try:
        out = run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=QUERY_TIMEOUT_S,
        )  # fmt: skip
        return parse_query(out.stdout) if out.stdout.strip() else []
    except FileNotFoundError:
        return []  # no NVIDIA driver: a CPU-only host, nothing to warn about
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        logger.warning("nvidia-smi query failed (%s): GPU checks skipped, host memory budget only", exc)
        return []


def memory_model(gpus: list[GpuInfo]) -> MemoryModel:
    """``uma`` when the driver reports no dedicated memory (GB10), ``discrete`` with VRAM figures, else ``none``."""
    if not gpus:
        return "none"
    return "uma" if gpus[0].total_gb is None else "discrete"


def vram_required_gb(est_vram_gb: float, max_free_frac: float) -> float:
    """Free VRAM needed for a job of ``est_vram_gb`` under ADR-005 A1 (exactly at the limit is allowed)."""
    return est_vram_gb / max_free_frac


def driver_too_old(gpu: GpuInfo, min_native: int = MIN_NATIVE_DRIVER_CUDA_13_3) -> bool:
    """A card without forward compatibility needs a native driver for the image's CUDA (GeForce + driver < 610)."""
    return not gpu.forward_compatible and gpu.driver_major < min_native


def supports_fp8(gpu: GpuInfo) -> bool:
    """FP8 tensor cores: sm >= 8.9 (Ada, Hopper, Blackwell)."""
    return gpu.compute_capability >= (8, 9)


def supports_nvfp4(gpu: GpuInfo) -> bool:
    """NVFP4: Blackwell, sm >= 10.0 (GB10 is 12.1)."""
    return gpu.compute_capability >= (10, 0)


__all__ = [
    "MIN_NATIVE_DRIVER_CUDA_13_3",
    "GpuInfo",
    "MemoryModel",
    "driver_too_old",
    "memory_model",
    "parse_query",
    "query_gpus",
    "supports_fp8",
    "supports_nvfp4",
    "vram_required_gb",
]
