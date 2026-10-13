"""vlm-runner server (A2): typed requests only, image gate, GPU token, one job per run id, our containers only."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.runner.server import Runner, RunnerConfig, RunnerServer

RUN = "n4-0123456789abcdef"
IMAGE_ID = "sha256:" + "a" * 64


class FakeDocker:
    def __init__(self, labels: dict[str, str] | None = None, image_id: str = IMAGE_ID) -> None:
        self.calls: list[list[str]] = []
        self.labels = labels if labels is not None else {"project": "vlm-training", "vlm.job": RUN}
        self.image_id = image_id

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if argv[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, self.image_id + "\n", "")
        if argv[0] == "inspect":
            state = {"Running": False, "ExitCode": 0, "OOMKilled": False}
            return subprocess.CompletedProcess(argv, 0, f"{json.dumps(self.labels)}|{json.dumps(state)}", "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _project(tmp_path: Path, verdict: str = "bao-security APPROVED 2026-10-13") -> RunnerConfig:
    code, data = tmp_path / "code", tmp_path / "data"
    for d in ("src", "configs", "docker", "scripts/qa"):
        (code / d).mkdir(parents=True)
    (code / "scripts/qa/eval_cord_worker.py").write_text("")
    row = f"vlm-x | vlm-x:1 (local) | - | base | {IMAGE_ID} | 1GB | {verdict} | 2026-10-13 | t\n"
    (code / "docker" / "IMAGES.lock").write_text("# header\n" + row)
    (data / "outputs" / "graph" / "runs" / RUN).mkdir(parents=True)
    (data / "data" / "mine").mkdir(parents=True)
    return RunnerConfig(socket_path=tmp_path / "r.sock", host_project=code, data_root=data,
                        state_dir=tmp_path / "state", lock_path=tmp_path / "GPU_LOCK")  # fmt: skip


def test_jobs_never_run_as_root(tmp_path: Path) -> None:
    """security R1: the runner may be root for the docker socket; its jobs never are."""
    cfg = _project(tmp_path)
    cfg.job_uid = 0
    with pytest.raises(ValueError, match="root"):
        Runner(cfg, docker=FakeDocker())
    cfg.job_uid = None  # default: the data root's owner (this test user), not the runner's uid
    runner, started = _runner(cfg, FakeDocker())
    runner.handle(_submit())
    import os

    assert started[0][started[0].index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


@pytest.mark.parametrize("plant", ["logs_symlink", "fifo"])
def test_a_planted_log_path_is_never_followed_or_waited_on(tmp_path: Path, plant: str) -> None:
    """security R2: logs/ swapped for a link, or stdout.log for a FIFO: refused at once, never followed or blocked."""
    import os

    cfg = _project(tmp_path)
    run_dir = cfg.data_root / "outputs" / "graph" / "runs" / RUN
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if plant == "logs_symlink":
        (run_dir / "logs").symlink_to(elsewhere)
    else:
        (run_dir / "logs").mkdir()
        os.mkfifo(run_dir / "logs" / "stdout.log")
    runner, started = _runner(cfg, FakeDocker())
    assert runner.handle(_submit()) == {"ok": False, "error": "mount_refused"}
    assert started == []
    assert list(elsewhere.iterdir()) == []


def _submit(**changes: Any) -> bytes:
    body = {
        "command": "submit", "run_id": RUN, "job_kind": "train", "image_tag": "vlm-x:1",
        "argv": ["automodel", f"/workspace/outputs/graph/runs/{RUN}/config.yaml"], "gpu": False, "memory_gb": 8,
        "run_dir": f"/app/outputs/graph/runs/{RUN}", "inputs": ["/app/data/mine"], **changes,
    }  # fmt: skip
    return json.dumps(body).encode()


def _runner(cfg: RunnerConfig, docker: FakeDocker) -> tuple[Runner, list[list[str]]]:
    started: list[list[str]] = []
    return Runner(cfg, docker=docker, popen=lambda argv, **kw: started.append(list(argv))), started


def test_a_submit_runs_a_narrowed_container_by_image_id_once(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    runner, started = _runner(cfg, FakeDocker())
    assert runner.handle(_submit()) == {"ok": True, "container": f"vlm-{RUN}", "image_id": IMAGE_ID}
    argv = started[0]
    assert IMAGE_ID in argv  # runs the id the gate checked, not the tag
    sources = [argv[i + 1].split(":")[0] for i, a in enumerate(argv) if a == "-v"]
    assert str(cfg.data_root) not in sources  # never the whole data root, never the project
    assert str(cfg.data_root / "data" / "mine") in sources
    assert f"vlm.job={RUN}" in argv
    assert runner.handle(_submit()) == {"ok": False, "error": "run_already_started"}  # one job per run id
    assert len(started) == 1


@pytest.mark.parametrize(
    ("verdict", "changes", "error"),
    [
        ("PROBE, not approved", {}, "image_not_approved"),
        ("bao-security APPROVED", {"gpu": True, "lock_token": "x" * 32}, "gpu_lock_not_held"),
        ("bao-security APPROVED", {"inputs": ["/app/.cache/hf"]}, "input_outside_roots"),  # protocol refuses it
        ("bao-security APPROVED", {"run_dir": "/app/outputs/graph/runs/other-0123456789abcdef"}, "bad_run_dir"),
        ("bao-security APPROVED", {"mounts": ["/:/h"]}, "unknown_field"),
    ],
)
def test_submit_refusals(tmp_path: Path, verdict: str, changes: dict[str, Any], error: str) -> None:
    runner, started = _runner(_project(tmp_path, verdict), FakeDocker())
    assert runner.handle(_submit(**changes)) == {"ok": False, "error": error}
    assert started == []


def test_a_gpu_job_needs_the_current_lock_token(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    handle = gpu_lock.acquire(cfg.lock_path, who="app", job="t", run_id=RUN)
    runner, started = _runner(cfg, FakeDocker())
    assert runner.handle(_submit(gpu=True, lock_token=handle.info.token))["ok"] is True
    assert len(started) == 1


def test_only_our_containers_are_touched(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    foreign = FakeDocker(labels={"project": "kafka"})
    runner, _ = _runner(cfg, foreign)
    for command in ("stop", "kill", "status", "remove"):
        assert runner.handle(json.dumps({"command": command, "run_id": RUN}).encode()) == {
            "ok": False, "error": "not_found"}  # fmt: skip
    assert not any(c[0] in ("stop", "kill", "rm") for c in foreign.calls)
    ours = FakeDocker()
    runner, _ = _runner(cfg, ours)
    assert runner.handle(json.dumps({"command": "status", "run_id": RUN}).encode()) == {
        "ok": True, "running": False, "exit_code": 0, "oom_killed": False}  # fmt: skip


def test_every_request_is_audited_without_argv(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    runner, _ = _runner(cfg, FakeDocker())
    runner.handle(_submit())
    runner.handle(b"not json")
    lines = [json.loads(line) for line in (cfg.state_dir / "runner.jsonl").read_text().splitlines()]
    assert [(r["command"], r["ok"]) for r in lines] == [("submit", True), (None, False)]
    assert "automodel" not in (cfg.state_dir / "runner.jsonl").read_text()


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[Path, list[list[str]]]]:
    cfg = _project(tmp_path)
    runner, started = _runner(cfg, FakeDocker())
    server = RunnerServer(runner, cfg.socket_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield cfg.socket_path, started
    server.shutdown()
    server.server_close()


def test_the_unix_socket_speaks_one_json_request(served: tuple[Path, list[list[str]]]) -> None:
    path, started = served
    assert oct(path.stat().st_mode & 0o777) == "0o660"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(str(path))
        s.sendall(_submit())
        s.shutdown(socket.SHUT_WR)
        reply = json.loads(s.makefile("rb").read())
    assert reply["ok"] is True
    assert len(started) == 1


def test_a_mount_the_host_refuses_is_named(tmp_path: Path) -> None:
    """Passes the protocol, refused by the host checks (a credential-looking file in the input)."""
    cfg = _project(tmp_path)
    (cfg.data_root / "data" / "mine" / "token").write_text("hf_x")
    runner, started = _runner(cfg, FakeDocker())
    assert runner.handle(_submit()) == {"ok": False, "error": "mount_refused"}
    assert started == []


def test_run_job_through_the_runner_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compose path: the app (no docker CLI) -> RunnerLauncher -> socket -> runner (image gate, mounts) -> docker."""
    from types import SimpleNamespace

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.jobrun import ContainerJob, run_job
    from vlm_pipeline.common.launcher import RunnerLauncher
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus

    cfg = _project(tmp_path)  # the app's project root here is the data root (dev layout); paths go out as /app/...
    runner, started = _runner(cfg, FakeDocker())
    server = RunnerServer(runner, cfg.socket_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(jobrun, "image_digest", lambda _i: pytest.fail("the app must not ask docker"))
    monkeypatch.setattr(
        "vlm_pipeline.common.preflight.preflight", lambda *a, **k: SimpleNamespace(ok=True, findings=())
    )
    monkeypatch.setattr(jobrun, "_wait_for_memory", lambda *a, **k: None)
    monkeypatch.setattr(jobrun, "git_state", lambda _r: ("c0ffee", False))
    monkeypatch.setattr(jobrun, "run_guarded", lambda launch, *a, **k: (launch(), SimpleNamespace(
        should_stop=threading.Event(), min_available_gb=30.0, fast_fall_killed=False,
        thresholds=SimpleNamespace(abort_gb=8.0))))  # fmt: skip
    monkeypatch.setattr(RunnerLauncher, "POLL_S", 0.01)
    run_dir = cfg.data_root / "outputs" / "graph" / "runs" / RUN
    try:
        final = run_job(
            ContainerJob(name=RUN, image="vlm-x:1", argv=("automodel", "c.yaml"), gpu=False, narrow=True,
                         rw_dirs=(run_dir,), ro_inputs=(cfg.data_root / "data" / "mine",)),
            RunManifest(run_id=RUN, name="t", stage="train", task="sft"), run_dir, project_root=cfg.data_root,
            who="vlm-app", est_peak_gb=1, lock_path=tmp_path / "GPU_LOCK", launcher=RunnerLauncher(cfg.socket_path),
        )  # fmt: skip
    finally:
        server.shutdown()
        server.server_close()
    assert final.status == RunStatus.COMPLETED
    assert final.container_digest == IMAGE_ID  # the id the runner gated and ran
    assert len(started) == 1


def test_inside_compose_without_a_runner_socket_nothing_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """security 7.2 (1): the app container never tries a docker CLI or socket."""
    from vlm_pipeline.common.launcher import LocalDockerLauncher, RunnerError, RunnerLauncher, default_launcher

    monkeypatch.delenv("VLM_RUNNER_SOCKET", raising=False)
    monkeypatch.setenv("VLM_IN_COMPOSE", "1")
    with pytest.raises(RunnerError, match="no_runner_socket"):
        default_launcher()
    monkeypatch.setenv("VLM_RUNNER_SOCKET", "/run/vlm-runner/runner.sock")
    assert isinstance(default_launcher(), RunnerLauncher)
    monkeypatch.delenv("VLM_RUNNER_SOCKET")
    monkeypatch.delenv("VLM_IN_COMPOSE")
    assert isinstance(default_launcher(), LocalDockerLauncher)  # dev on the host


def test_a_stranger_cannot_use_the_socket(tmp_path: Path) -> None:
    """security N2: SO_PEERCRED, not only the socket's group."""
    cfg = _project(tmp_path)
    cfg.allowed_peer_uids = frozenset({999_999})
    runner, started = _runner(cfg, FakeDocker())
    server = RunnerServer(runner, cfg.socket_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(cfg.socket_path))
            s.sendall(_submit())
            s.shutdown(socket.SHUT_WR)
            assert json.loads(s.makefile("rb").read()) == {"ok": False, "error": "peer_not_allowed"}
    finally:
        server.shutdown()
        server.server_close()
    assert started == []
