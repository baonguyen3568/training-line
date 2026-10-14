"""LocalDockerLauncher (ADR-015 JobLauncher): the container runs the image id the gate approved, not the tag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common import launcher as launcher_mod
from vlm_pipeline.common.jobrun import ContainerJob


class _Proc:
    def wait(self) -> int:
        return 0


def test_run_uses_the_gated_image_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kw: Any) -> _Proc:
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    job = ContainerJob(name="r1", image="vlm-automodel:26.08-6", argv=("true",), uid=1000, gid=1000)
    pinned = "sha256:" + "ab" * 32
    run = launcher_mod.LocalDockerLauncher().run
    rc = run(job, tmp_path, lock_token=None, log_path=tmp_path / "l", image_id=pinned)
    assert rc == 0
    argv = seen["argv"]
    assert pinned in argv
    assert "vlm-automodel:26.08-6" not in argv  # the tag is never re-resolved between the gate and docker run
    assert argv[argv.index(pinned) + 1] == "true"


def _launcher_with(replies: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    from vlm_pipeline.common.launcher import RunnerLauncher

    launcher = RunnerLauncher(tmp_path / "runner.sock")
    monkeypatch.setattr(RunnerLauncher, "POLL_S", 0.0)
    calls = iter(replies)
    monkeypatch.setattr(launcher, "_call", lambda body: next(calls))
    return launcher


def _narrow_job(tmp_path: Path) -> object:
    from vlm_pipeline.common.jobrun import ContainerJob

    run = tmp_path / "outputs" / "runs" / "r"
    return ContainerJob(name="r", image="vlm-x:1", argv=("python", "-m", "vlm_pipeline.download.hub", "--run-dir", "x"),
                        narrow=True, rw_dirs=(run,))  # fmt: skip


def test_a_container_not_visible_yet_is_waited_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM 2026-10-14: the runner answers submit before docker has created the container; the first poll said
    not_found and the job's real exit (1) became SYS-000."""
    launcher = _launcher_with([
        {"ok": True, "image_id": "sha256:x"},
        {"ok": False, "error": "not_found"},
        {"ok": True, "running": True},
        {"ok": True, "running": False, "exit_code": 1},
    ], monkeypatch, tmp_path)  # fmt: skip
    assert launcher.run(_narrow_job(tmp_path), tmp_path, lock_token=None, log_path=tmp_path / "l", image_id=None) == 1  # type: ignore[attr-defined]


def test_a_container_that_vanishes_after_it_was_seen_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_pipeline.common.launcher import RunnerError

    launcher = _launcher_with([
        {"ok": True, "image_id": "sha256:x"},
        {"ok": True, "running": True},
        {"ok": False, "error": "not_found"},
    ], monkeypatch, tmp_path)  # fmt: skip
    with pytest.raises(RunnerError, match="not_found"):
        launcher.run(_narrow_job(tmp_path), tmp_path, lock_token=None, log_path=tmp_path / "l", image_id=None)  # type: ignore[attr-defined]
