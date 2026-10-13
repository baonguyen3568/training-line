"""D-4: UMA vs discrete detection and per-GPU limits from nvidia-smi CSV fixtures.

GB10 row is the real output of this host (2026-10-13). The discrete rows follow nvidia-smi's documented format;
they are NOT from real hardware (no amd64 machine in the team).
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from vlm_pipeline.common.gpu_mem import (
    driver_too_old,
    memory_model,
    parse_query,
    query_gpus,
    supports_fp8,
    supports_nvfp4,
    vram_required_gb,
)

GB10 = "NVIDIA GB10, [N/A], [N/A], [N/A], 12.1, 580.142"
H100 = "NVIDIA H100 80GB HBM3, 81559, 1024, 80535, 9.0, 580.95.05"
L40S = "NVIDIA L40S, 46068, 512, 45556, 8.9, 580.95.05"
RTX4090_OLD = "NVIDIA GeForce RTX 4090, 24564, 1200, 23364, 8.9, 580.95.05"
RTX5090_NEW = "NVIDIA GeForce RTX 5090, 32607, 900, 31707, 12.0, 610.12"
A100 = "NVIDIA A100-SXM4-40GB, 40960, 0, 40960, 8.0, 580.95.05"


def test_gb10_is_uma_with_no_vram_figures() -> None:
    [gpu] = parse_query(GB10)
    assert (gpu.total_gb, gpu.free_gb) == (None, None)
    assert gpu.compute_capability == (12, 1)
    assert memory_model([gpu]) == "uma"
    assert gpu.forward_compatible
    assert not driver_too_old(gpu)
    assert supports_fp8(gpu)
    assert supports_nvfp4(gpu)


def test_h100_is_discrete_in_gib() -> None:
    [gpu] = parse_query(H100)
    assert memory_model([gpu]) == "discrete"
    assert gpu.total_gb == pytest.approx(81559 / 1024)
    assert gpu.free_gb == pytest.approx(80535 / 1024)
    assert supports_fp8(gpu)
    assert not supports_nvfp4(gpu)


@pytest.mark.parametrize(("row", "too_old"), [(RTX4090_OLD, True), (RTX5090_NEW, False), (L40S, False), (H100, False)])
def test_consumer_geforce_needs_a_native_driver(row: str, too_old: bool) -> None:
    [gpu] = parse_query(row)
    assert driver_too_old(gpu) is too_old


def test_quant_formats_follow_compute_capability() -> None:
    a100, l40s = parse_query(f"{A100}\n{L40S}")
    assert not supports_fp8(a100)  # sm 8.0: no FP8 tensor cores -> hide the FP8 option
    assert supports_fp8(l40s)
    assert not supports_nvfp4(l40s)


def test_vram_budget_is_80_percent_of_free() -> None:
    assert vram_required_gb(20.0, 0.8) == pytest.approx(25.0)


def test_no_nvidia_smi_means_no_gpu() -> None:
    def missing(*a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    assert query_gpus(run=missing) == []
    assert memory_model([]) == "none"


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired("nvidia-smi", 5), subprocess.CalledProcessError(9, "nvidia-smi"), PermissionError()],
)
def test_broken_or_hung_nvidia_smi_never_blocks_a_job(failure: Exception, caplog: pytest.LogCaptureFixture) -> None:
    def broken(*a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        assert kw["timeout"] <= 5
        raise failure

    assert query_gpus(run=broken) == []
    assert "GPU checks skipped" in caplog.text


def test_garbage_output_degrades_to_no_gpu() -> None:
    def garbage(*a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(a, 0, stdout="Failed to initialize NVML: Driver/library mismatch\n")

    assert query_gpus(run=garbage) == []


def test_malformed_row_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="unexpected nvidia-smi row"):
        parse_query("NVIDIA X, 1, 2")


# ------------------------------------------------------------------------------- preflight (discrete branch)


def _preflight(gpus_csv: str, est: float, est_vram: float | None = None):  # type: ignore[no-untyped-def]
    from vlm_pipeline.common.diskguard import DiskSnapshot
    from vlm_pipeline.common.memguard import MemSnapshot
    from vlm_pipeline.common.preflight import preflight

    host = MemSnapshot(mem_total_gb=256.0, mem_available_gb=200.0, cached_gb=0.0, swap_free_gb=8.0)
    disk = DiskSnapshot(path="/x", total_gb=2000.0, free_gb=1000.0)
    return preflight(est, snapshot=host, disk=disk, lock_holder=lambda: None,
                     gpus=parse_query(gpus_csv) if gpus_csv else [], est_vram_gb=est_vram)  # fmt: skip


def test_discrete_vram_is_a_second_budget() -> None:
    from vlm_pipeline.common.errors import ExitCode

    assert _preflight(L40S, 20.0, est_vram=35.0).ok  # 35 <= 0.8 x 44.5 GiB free
    v = _preflight(L40S, 20.0, est_vram=40.0)  # 40 / 0.8 = 50 > 44.5
    assert v.exit_code is ExitCode.NO_MEM
    [f] = v.findings
    assert (f.code, f.params["memory"]) == ("MEM-001", "vram")
    assert v.to_json()  # details carry the GPU list


def test_geforce_with_an_old_driver_is_refused_first() -> None:
    from vlm_pipeline.common.errors import ExitCode

    v = _preflight(RTX4090_OLD, 10.0, est_vram=40.0)
    assert v.exit_code is ExitCode.GPU_ENV  # permanent: wins over the (transient) VRAM refusal
    assert {f.code for f in v.findings} == {"GPU-001", "MEM-001"}


def test_uma_and_cpu_only_hosts_keep_the_host_budget_only() -> None:
    assert _preflight(GB10, 20.0, est_vram=500.0).ok  # GB10: VRAM is the host pool, already checked
    assert _preflight("", 20.0, est_vram=500.0).ok  # no GPU: nothing more to check
