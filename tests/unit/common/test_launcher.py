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


def test_the_runner_never_drops_extra_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from vlm_pipeline.common.launcher import RunnerError

    launcher = _launcher_with([], monkeypatch, tmp_path)
    job = dataclasses.replace(_narrow_job(tmp_path), image="vlm-nemo-rl:0.7.0-1", group_add=(0,))  # type: ignore[type-var]
    with pytest.raises(RunnerError, match="group_add_not_supported"):
        launcher.run(job, tmp_path, lock_token=None, log_path=tmp_path / "l", image_id=None)  # type: ignore[attr-defined]


def test_one_split_feeding_train_and_val_sends_one_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         caplog: pytest.LogCaptureFixture) -> None:  # fmt: skip
    """Slot #2 2026-10-14: train and val data from one split named the same dataset twice; vlm-runner refused it
    (inputs_nested) and the job failed SYS-000 without a word."""
    from vlm_pipeline.common.jobrun import ContainerJob
    from vlm_pipeline.common.launcher import RunnerError, RunnerLauncher

    data = tmp_path / ".cache" / "hf" / "hub" / "datasets--o--d"
    model = tmp_path / ".cache" / "hf" / "hub" / "models--o--m"
    run = tmp_path / "outputs" / "runs" / "r"
    job = ContainerJob(name="r", image="vlm-x:1", argv=("automodel", "c.yaml"), narrow=True, rw_dirs=(run,),
                       ro_inputs=(model, data, data, data / "snapshots" / "x"))  # fmt: skip
    sent: list[dict[str, Any]] = []
    launcher = RunnerLauncher(tmp_path / "runner.sock")

    def call(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(body)
        return {"ok": False, "error": "inputs_nested"}

    monkeypatch.setattr(launcher, "_call", call)
    with caplog.at_level("ERROR", logger="vlm_pipeline.common.launcher"), pytest.raises(RunnerError):
        launcher.run(job, tmp_path, lock_token="t", log_path=tmp_path / "l", image_id=None)  # noqa: S106 - dummy
    assert sent[0]["inputs"] == ["/app/.cache/hf/hub/datasets--o--d", "/app/.cache/hf/hub/models--o--m"]
    assert "vlm-runner refused job r: inputs_nested" in caplog.text  # a refusal is never silent


def test_outermost_inputs_drop_duplicates_and_nested_paths(tmp_path: Path) -> None:
    from vlm_pipeline.common.jobrun import outermost_inputs

    a, b = tmp_path / "a", tmp_path / "b"
    assert outermost_inputs([b, a / "x", a, a, tmp_path / "ab"]) == (a, tmp_path / "ab", b)


def test_what_the_launcher_sends_passes_the_runners_own_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """bao-tester 2026-10-14: the unit tests never met the runner's validation; this sends a split-fed train job's
    real submit body through vlm_pipeline.runner.protocol.parse_submit (the code the runner runs)."""
    import json

    from vlm_pipeline.common.jobrun import ContainerJob
    from vlm_pipeline.common.launcher import RunnerLauncher
    from vlm_pipeline.runner.protocol import parse_request

    data = tmp_path / ".cache" / "hf" / "hub" / "datasets--o--d" / "snapshots" / "x"
    model = tmp_path / ".cache" / "hf" / "hub" / "models--o--m"
    run = tmp_path / "outputs" / "graph" / "runs" / "n4-0123456789abcdef"
    job = ContainerJob(name=run.name, image="vlm-x:1", argv=("automodel", "c.yaml"), narrow=True, rw_dirs=(run,),
                       ro_inputs=(model, data, data))  # fmt: skip
    parsed: list[Any] = []
    launcher = RunnerLauncher(tmp_path / "runner.sock")

    def call(body: dict[str, Any]) -> dict[str, Any]:
        parsed.append(parse_request(json.dumps(body).encode()))  # raises ProtocolError like the runner would
        return {"ok": False, "error": "stop_here"}

    monkeypatch.setattr(launcher, "_call", call)
    token = "A" * 43  # a well-formed dummy (the protocol checks its shape)
    with pytest.raises(Exception, match="stop_here"):
        launcher.run(job, tmp_path, lock_token=token, log_path=tmp_path / "l", image_id=None)
    assert len(parsed[0].inputs) == 2
