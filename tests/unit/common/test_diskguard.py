"""Unit tests for ``vlm_pipeline.common.diskguard`` (ADR-005 D8). No real disk/lock state is touched."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vlm_pipeline.common.diskguard import (
    GIB,
    DiskSnapshot,
    check_disk,
    check_lock,
    read_disk,
    required_disk_gb,
)


def _fake_statvfs(blocks: int, bavail: int, frsize: int = 4096):  # type: ignore[no-untyped-def]
    def statvfs(_path: str) -> os.statvfs_result:
        # f_bsize, f_frsize, f_blocks, f_bfree, f_bavail, f_files, f_ffree, f_favail, f_flag, f_namemax
        return os.statvfs_result((frsize, frsize, blocks, bavail + 10, bavail, 0, 0, 0, 0, 255))

    return statvfs


def test_read_disk_uses_bavail_not_bfree() -> None:
    frsize = 4096
    snap = read_disk(Path("/x"), statvfs=_fake_statvfs(blocks=1000 * GIB // frsize, bavail=100 * GIB // frsize))
    assert snap.total_gb == pytest.approx(1000)
    assert snap.free_gb == pytest.approx(100)
    assert snap.path == "/x"


def test_required_disk_gb_window_plus_headroom() -> None:
    assert required_disk_gb(19.0) == pytest.approx(19.0 * 3 + 50.0)
    assert required_disk_gb(0.0) == pytest.approx(50.0)  # every job still writes logs/caches
    assert required_disk_gb(10.0, max_recent=0, headroom_gb=0) == pytest.approx(10.0)


def test_required_disk_gb_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        required_disk_gb(-1.0)


def test_check_disk_ok_and_fail() -> None:
    assert check_disk(50.0, DiskSnapshot("/p", 3700.0, 2000.0)) == []
    reasons = check_disk(107.0, DiskSnapshot("/p", 3700.0, 100.0))
    assert len(reasons) == 1
    assert "100.0" in reasons[0]
    assert "107.0" in reasons[0]


def _lock_json(**over: object) -> str:
    data: dict[str, object] = {
        "who": "bao-backend",
        "job": "smoke",
        "run_id": "r1",
        "host_pid": 4242,
        "started_at": "t0",
        "token": "secret",
    }
    data.update(over)
    return json.dumps(data)


def test_check_lock_absent(tmp_path: Path) -> None:
    assert check_lock(tmp_path / "GPU_LOCK") is None


def test_check_lock_alive_and_stale_never_deletes(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    lock.write_text(_lock_json())

    alive = check_lock(lock, pid_alive=lambda pid: pid == 4242)
    assert alive is not None
    assert alive.alive
    assert not alive.corrupt
    assert (alive.who, alive.job, alive.run_id, alive.host_pid, alive.started_at) == (
        "bao-backend",
        "smoke",
        "r1",
        4242,
        "t0",
    )

    stale = check_lock(lock, pid_alive=lambda _pid: False)
    assert stale is not None
    assert not stale.alive
    assert lock.exists()  # read-only: stale lock is reported, never removed


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[1, 2]",
        _lock_json(host_pid="abc"),
        json.dumps({"who": "x", "job": "manual"}),  # missing fields: not written by gpu_lock.acquire
        "bao-backend | smoke LoRA | 2026-10-12T17:00 | 4242\n",  # legacy text lock
    ],
)
def test_check_lock_malformed_is_treated_busy(tmp_path: Path, content: str) -> None:
    lock = tmp_path / "GPU_LOCK"
    lock.write_text(content)
    status = check_lock(lock, pid_alive=lambda _pid: False)
    assert status is not None
    assert status.corrupt
    assert status.alive
