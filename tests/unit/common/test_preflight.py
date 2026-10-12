"""Unit tests for vlm_pipeline.common.preflight (CPU only, all I/O injected)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.common.diskguard import DiskSnapshot, LockStatus
from vlm_pipeline.common.error_catalog import load_catalog
from vlm_pipeline.common.errors import ExitCode
from vlm_pipeline.common.memguard import MemSnapshot
from vlm_pipeline.common.preflight import Finding, check_memory, combine, preflight


def _snap(avail: float, swap_free: float = 6.0) -> MemSnapshot:
    return MemSnapshot(mem_total_gb=121.7, mem_available_gb=avail, cached_gb=20.0, swap_free_gb=swap_free)


DISK_OK = DiskSnapshot(path="/x", total_gb=3700.0, free_gb=2000.0)
DISK_FULL = DiskSnapshot(path="/x", total_gb=3700.0, free_gb=10.0)


def _no_lock() -> LockStatus | None:
    return None


def _busy_lock() -> LockStatus | None:
    return LockStatus(who="bao-backend", job="smoke", run_id=None, host_pid=123, started_at="16:00", alive=True)


def test_adr005_examples_with_43gb_available() -> None:
    # ADR-005 D8: LoRA 4B (~16 GB) OK, full-FT 8B (~70 GB) refused, at MemAvailable 43 GB.
    assert check_memory(16.0, snapshot=_snap(43.0)) == []
    [f] = check_memory(70.0, snapshot=_snap(43.0))
    assert f.exit_code is ExitCode.NO_MEM
    assert f.params["need_gb"] == pytest.approx(84.0)


def test_low_swap_refuses_even_with_ram() -> None:
    [f] = check_memory(10.0, snapshot=_snap(100.0, swap_free=1.5))
    assert f.exit_code is ExitCode.NO_MEM
    assert "SwapFree" in f.reason


def test_swap_is_never_counted_as_available() -> None:
    # 15 GB RAM + 15 GB swap free must still refuse a 10 GB job (needs 20 GB RAM).
    assert check_memory(10.0, snapshot=_snap(15.0, swap_free=15.0))


def test_negative_peak_rejected() -> None:
    with pytest.raises(ValueError, match="est_peak_gb"):
        check_memory(-1.0, snapshot=_snap(43.0))


# ------------------------------------------------------------------------------------------ combine


def test_combine_priority_lock_then_mem_then_disk_and_keeps_all_reasons() -> None:
    findings = [
        Finding("DISK-001", ExitCode.NO_DISK, "disk"),
        Finding("MEM-001", ExitCode.NO_MEM, "mem"),
        Finding("SYS-001", ExitCode.LOCK_BUSY, "lock"),
    ]
    v = combine(findings)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert v.reasons == ["disk", "mem", "lock"]
    assert combine(findings[:2]).exit_code is ExitCode.NO_MEM
    assert combine(findings[:1]).exit_code is ExitCode.NO_DISK
    ok = combine([])
    assert ok.ok
    assert ok.exit_code is ExitCode.OK


def test_combine_rejects_unknown_exit_code() -> None:
    with pytest.raises(ValueError, match="unknown"):
        combine([Finding("MEM-002", ExitCode.MEM_PREEMPT, "x")])


# ---------------------------------------------------------------------------------------- preflight


def test_preflight_ok_json_shape() -> None:
    v = preflight(16.0, snapshot=_snap(43.0), disk=DISK_OK, lock_holder=_no_lock)
    assert v.ok
    data = json.loads(v.to_json())
    assert data["ok"] is True
    assert data["exit_code"] == 0
    assert data["reasons"] == []
    assert {"mem", "disk", "lock", "thresholds", "findings"} <= data.keys()
    assert "details" not in data  # one source: details are flattened at top level
    assert data["mem"]["required_gb"] == pytest.approx(26.0)
    assert data["lock"] is None


def test_preflight_reports_everything_with_lock_priority() -> None:
    v = preflight(70.0, ckpt_gb=20.0, snapshot=_snap(43.0), disk=DISK_FULL, lock_holder=_busy_lock)
    assert v.exit_code is ExitCode.LOCK_BUSY
    codes = {f.exit_code for f in v.findings}
    assert codes == {ExitCode.LOCK_BUSY, ExitCode.NO_MEM, ExitCode.NO_DISK}
    assert len(v.reasons) == 3


def test_preflight_disk_needs_headroom_even_without_checkpoint() -> None:
    v = preflight(10.0, ckpt_gb=0.0, snapshot=_snap(43.0), disk=DISK_FULL, lock_holder=_no_lock)
    assert v.exit_code is ExitCode.NO_DISK


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (LockStatus("a", "j", None, 99, None, alive=False), "stale"),
        (LockStatus("?", "?", None, None, None, alive=True, corrupt=True), "unreadable"),
    ],
)
def test_preflight_stale_or_corrupt_lock_is_still_busy(status: LockStatus, needle: str) -> None:
    v = preflight(10.0, snapshot=_snap(43.0), disk=DISK_OK, lock_holder=lambda: status)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert needle in v.reasons[0]
    assert "never delete" in v.reasons[0]


def test_preflight_reads_lock_file_under_root_without_touching_it(tmp_path: Path) -> None:
    lock = tmp_path / ".team" / "GPU_LOCK"
    lock.parent.mkdir()
    lock.write_text("bao-x | job | 2026-10-12T16:00 | PID 1\n")  # legacy text format -> corrupt -> busy
    before = lock.read_bytes()
    v = preflight(10.0, snapshot=_snap(43.0), disk=DISK_OK, root=tmp_path)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert lock.read_bytes() == before


def test_every_finding_code_is_in_the_catalog_with_its_params() -> None:
    v = preflight(70.0, ckpt_gb=20.0, snapshot=_snap(43.0, swap_free=1.0), disk=DISK_FULL, lock_holder=_busy_lock)
    catalog = load_catalog()
    assert {f.code for f in v.findings} == {"SYS-001", "MEM-001", "DISK-001"}
    for f in v.findings:
        declared = {p.name for p in catalog[f.code].params}
        assert set(f.params) <= declared, (f.code, set(f.params) - declared)
    assert all("code" in f for f in json.loads(v.to_json())["findings"])
