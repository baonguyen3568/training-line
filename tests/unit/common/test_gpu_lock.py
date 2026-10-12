from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.gpu_lock import GpuLockError


def _acquire(path: Path, **kw: object) -> gpu_lock.LockHandle:
    return gpu_lock.acquire(path, who="bao-backend", job="test", run_id="r1", host_pid=4242, **kw)  # type: ignore[arg-type]


def test_acquire_writes_complete_json(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    h = _acquire(lock)
    data = json.loads(lock.read_text())
    assert data["who"] == "bao-backend"
    assert data["host_pid"] == 4242
    assert data["token"] == h.info.token
    assert "pid" not in data
    assert list(tmp_path.iterdir()) == [lock]  # no tmp leftovers


def test_second_acquire_fails_with_holder_info_without_token(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    h = _acquire(lock)
    with pytest.raises(GpuLockError) as ei:
        gpu_lock.acquire(lock, who="other", job="x", run_id="r2", host_pid=1)
    params = ei.value.params
    assert ei.value.code == "SYS-001"
    assert params["who"] == "bao-backend"
    assert params["state"] in {"busy", "stale"}
    assert set(params) == {"who", "job", "since", "state"}
    assert h.info.token not in json.dumps(params)
    assert json.loads(lock.read_text())["token"] == h.info.token  # untouched


def _race(path: str, q: mp.Queue) -> None:  # type: ignore[type-arg]
    try:
        gpu_lock.acquire(Path(path), who="p", job="race", run_id="r", host_pid=1)
        q.put("won")
    except GpuLockError:
        q.put("lost")


def test_concurrent_acquire_exactly_one_wins(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    q: mp.Queue = mp.Queue()  # type: ignore[type-arg]
    procs = [mp.Process(target=_race, args=(str(lock), q)) for _ in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(10)
    results = [q.get(timeout=5) for _ in procs]
    assert results.count("won") == 1


def test_release_requires_matching_token(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    h = _acquire(lock)
    forged = gpu_lock.LockHandle(path=lock, info=gpu_lock.LockInfo(**{**h.info.__dict__, "token": "0" * 32}))
    with pytest.raises(GpuLockError):
        gpu_lock.release(forged)
    assert lock.exists()
    gpu_lock.release(h)
    assert not lock.exists()
    gpu_lock.release(h)  # idempotent once gone


def test_release_does_not_remove_someone_elses_newer_lock(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    h = _acquire(lock)
    lock.unlink()  # e.g. removed by its owner, then someone else acquired
    other = gpu_lock.acquire(lock, who="other", job="x", run_id="r2", host_pid=1)
    with pytest.raises(GpuLockError):
        gpu_lock.release(h)
    assert json.loads(lock.read_text())["token"] == other.info.token


@pytest.mark.parametrize("content", ["", "{", "[]", '{"who": "x"}', "ai | job | 10:00 | 123"])
def test_corrupt_or_legacy_lock_raises_and_is_not_removed(tmp_path: Path, content: str) -> None:
    lock = tmp_path / "GPU_LOCK"
    lock.write_text(content)
    with pytest.raises(GpuLockError):
        gpu_lock.read_lock(lock)
    with pytest.raises(GpuLockError):
        _acquire(lock)
    assert lock.read_text() == content


def test_missing_lock_reads_none(tmp_path: Path) -> None:
    assert gpu_lock.read_lock(tmp_path / "GPU_LOCK") is None


def test_stale_detection_is_report_only(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    _acquire(lock)
    info = gpu_lock.read_lock(lock)
    assert info is not None
    assert gpu_lock.is_stale(info, pid_alive=lambda _pid: False)
    assert not gpu_lock.is_stale(info, pid_alive=lambda _pid: True)
    assert lock.exists()


def test_verify_token_and_holds_lock(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    assert not gpu_lock.holds_lock("x", lock)
    h = _acquire(lock)
    gpu_lock.verify_token(lock, h.info.token)
    assert gpu_lock.holds_lock(h.info.token, lock)
    assert not gpu_lock.holds_lock("deadbeef", lock)
    with pytest.raises(GpuLockError):
        gpu_lock.verify_token(lock, "deadbeef")


def test_default_lock_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_lock.PATH_ENV, str(tmp_path / "L"))
    assert gpu_lock.default_lock_path() == tmp_path / "L"
    monkeypatch.delenv(gpu_lock.PATH_ENV)
    assert gpu_lock.default_lock_path().parts[-2:] == (".team", "GPU_LOCK")


def test_error_params_never_contain_paths(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    lock.write_text("garbage")
    with pytest.raises(GpuLockError) as ei:
        gpu_lock.read_lock(lock)
    assert ei.value.params["state"] == "corrupt"
    assert str(tmp_path) not in json.dumps(ei.value.params)


def test_cli_acquire_status_release(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lock = tmp_path / "GPU_LOCK"
    base = ["--path", str(lock)]
    assert gpu_lock.main([*base, "status"]) == 0
    assert gpu_lock.main([*base, "acquire", "--who", "w", "--job", "j", "--run-id", "r", "--host-pid", "1"]) == 0
    token = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["token"]
    assert gpu_lock.main([*base, "status"]) == 4
    assert "token" not in capsys.readouterr().out
    assert gpu_lock.main([*base, "release", "--token", "bad"]) == 4
    assert lock.exists()
    assert gpu_lock.main([*base, "release", "--token", token]) == 0
    assert not lock.exists()
