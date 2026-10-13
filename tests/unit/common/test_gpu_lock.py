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


def test_verify_token(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    with pytest.raises(gpu_lock.GpuLockNotHeldError):
        gpu_lock.verify_token(lock, "x")
    h = _acquire(lock)
    gpu_lock.verify_token(lock, h.info.token)
    with pytest.raises(gpu_lock.GpuLockNotHeldError):
        gpu_lock.verify_token(lock, "\u00e1bc-non-ascii")
    with pytest.raises(gpu_lock.GpuLockNotHeldError) as ei:
        gpu_lock.verify_token(lock, "deadbeef")
    assert ei.value.code == "SYS-004"
    assert ei.value.params == {}


def test_default_lock_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_lock.PATH_ENV, str(tmp_path / "L"))
    assert gpu_lock.default_lock_path() == tmp_path / "L"
    monkeypatch.delenv(gpu_lock.PATH_ENV)
    assert gpu_lock.default_lock_path().parts[-3:] == (".team", "lock", "GPU_LOCK")  # its own folder (2026-10-13)


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


def test_newer_schema_version_is_rejected_as_corrupt(tmp_path: Path) -> None:
    lock = tmp_path / "GPU_LOCK"
    _acquire(lock)
    data = json.loads(lock.read_text())
    data["schema_version"] = gpu_lock.LOCK_SCHEMA_VERSION + 1
    lock.write_text(json.dumps(data))
    with pytest.raises(GpuLockError) as ei:
        gpu_lock.read_lock(lock)
    assert ei.value.params["state"] == "corrupt"


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        ("outputs/graph/runs/n4-0123456789abcdef", "outputs/graph/runs/n4-0123456789abcdef"),
        ("/etc/passwd", None),
        ("outputs/../.env", None),
        ("data/x", None),  # only under outputs/
        (None, None),
        (5, None),
    ],
)
def test_the_lock_names_the_run_dir_only_when_it_is_safe(tmp_path: Path, given: object, kept: str | None) -> None:
    """tester: tools read the holder's run dir from the lock instead of searching; garbage is never trusted."""
    import json

    from vlm_pipeline.common import gpu_lock

    path = tmp_path / "GPU_LOCK"
    gpu_lock.acquire(path, who="t", job="j", run_id="r", run_dir=given if isinstance(given, str) else None)
    raw = json.loads(path.read_text())
    raw["run_dir"] = given  # also what a hand-edited lock could hold
    path.write_text(json.dumps(raw))
    info = gpu_lock.read_lock(path)
    assert info is not None
    assert info.run_dir == kept


def test_an_old_lock_without_run_dir_still_reads(tmp_path: Path) -> None:
    import json

    from vlm_pipeline.common import gpu_lock

    path = tmp_path / "GPU_LOCK"
    gpu_lock.acquire(path, who="t", job="j", run_id="r")
    raw = json.loads(path.read_text())
    raw.pop("run_dir")
    path.write_text(json.dumps(raw))
    info = gpu_lock.read_lock(path)
    assert info is not None
    assert info.run_dir is None


def test_a_lock_left_at_the_old_path_keeps_the_gpu_busy(tmp_path: Path) -> None:
    """Moving to .team/lock/GPU_LOCK: while a job started before the move holds .team/GPU_LOCK, nobody else starts."""
    old = gpu_lock.acquire(tmp_path / ".team" / "GPU_LOCK", who="old", job="train", run_id="r-old")
    new = gpu_lock.lock_path_for(tmp_path)
    assert gpu_lock.read_lock(new) == old.info  # readers see the old holder
    with pytest.raises(GpuLockError, match="old lock path"):
        gpu_lock.acquire(new, who="new", job="t", run_id="r-new")
    gpu_lock.release(old)
    assert gpu_lock.acquire(new, who="new", job="t", run_id="r-new").info.who == "new"


def test_a_new_lock_is_mirrored_at_the_old_path_and_released_from_both(tmp_path: Path) -> None:
    """PM 2026-10-13: processes still running pre-move code read .team/GPU_LOCK; they must see the GPU busy too."""
    new, old = gpu_lock.lock_path_for(tmp_path), tmp_path / gpu_lock.LEGACY_REL
    handle = _acquire(new)
    assert old.exists()
    assert old.stat().st_ino == new.stat().st_ino  # one inode: the same lock, never two diverging copies
    assert gpu_lock.read_lock(old) == handle.info
    gpu_lock.release(handle)
    assert not new.exists()
    assert not old.exists()


def test_an_old_code_holder_taking_the_old_path_between_our_links_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new, old = gpu_lock.lock_path_for(tmp_path), tmp_path / gpu_lock.LEGACY_REL
    old.parent.mkdir(parents=True, exist_ok=True)
    other = gpu_lock.acquire(tmp_path / "elsewhere" / "GPU_LOCK", who="old-app", job="train", run_id="r0")
    old.write_text((tmp_path / "elsewhere" / "GPU_LOCK").read_text())  # old code wrote its lock at the old path
    monkeypatch.setattr(gpu_lock, "_legacy_of", lambda _p: None)  # ...right after our "old path is free" check
    with pytest.raises(GpuLockError, match="old lock path") as err:
        _acquire(new)
    assert err.value.params["who"] == "old-app"
    assert not new.exists()  # ours rolled back: never two jobs on the GPU
    assert gpu_lock.read_lock(old) == other.info


def test_release_never_removes_an_old_path_lock_that_is_not_ours(tmp_path: Path) -> None:
    new, old = gpu_lock.lock_path_for(tmp_path), tmp_path / gpu_lock.LEGACY_REL
    handle = _acquire(new)
    old.unlink()
    foreign = gpu_lock.acquire(tmp_path / "x" / "GPU_LOCK", who="old-app", job="train", run_id="r0")
    old.write_text((tmp_path / "x" / "GPU_LOCK").read_text())
    gpu_lock.release(handle)
    assert not new.exists()
    assert gpu_lock.read_lock(old) == foreign.info


def test_a_lock_outside_a_lock_folder_has_no_mirror(tmp_path: Path) -> None:
    """compose: /run/vlm-lock/GPU_LOCK never had an old path."""
    lock = tmp_path / "vlm-lock" / "GPU_LOCK"
    handle = _acquire(lock)
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["GPU_LOCK", "vlm-lock"]
    gpu_lock.release(handle)
