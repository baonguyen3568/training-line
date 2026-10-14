"""How a job container is started, stopped and inspected (ADR-015): one interface, two implementations.

``LocalDockerLauncher`` (dev on the host, today) calls the docker CLI. ``RunnerLauncher`` (compose, M3) will talk to
the ``vlm-runner`` sidecar over a unix socket: the app then has no docker socket, no docker CLI and no token. Both
build the container from the same ``jobrun.docker_args`` (one source for mounts, secrets masking and env); the runner
receives the typed job and builds the argv itself, never an argv from the app.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vlm_pipeline.common.jobrun import ContainerJob, ExitInfo


class JobLauncher(Protocol):
    def image_id(self, image: str) -> str | None: ...

    def run(
        self, job: ContainerJob, project_root: Path, *, lock_token: str | None, log_path: Path, image_id: str | None
    ) -> int:
        """Start the container, block until it exits (output appended to ``log_path``); returns the exit code.

        ``image_id``: the id the image gate checked; the container runs THAT id, not whatever the tag points at now.
        """
        ...

    def stop(self, name: str, grace_s: int) -> None:
        """Graceful stop, NON-blocking (the memory watchdog keeps polling during the grace period)."""
        ...

    def kill(self, name: str) -> None: ...

    def state(self, name: str) -> ExitInfo: ...

    def remove(self, name: str) -> None: ...


class LocalDockerLauncher:
    """The docker CLI on the host (dev mode; the behaviour run_job always had)."""

    checks_images = True  # run_job resolves and gates the image itself before the lock

    def image_id(self, image: str) -> str | None:
        try:
            return subprocess.run(  # noqa: S603
                ["docker", "image", "inspect", "-f", "{{.Id}}", image],  # noqa: S607
                capture_output=True, text=True, check=True,
            ).stdout.strip() or None  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            return None

    def run(
        self, job: ContainerJob, project_root: Path, *, lock_token: str | None, log_path: Path, image_id: str | None
    ) -> int:
        import dataclasses

        from vlm_pipeline.common.jobrun import docker_args

        if job.download_budget is not None or job.hf_token is not None:  # no egress proxy outside the compose stack
            raise RunnerError("downloads_need_runner")
        pinned = dataclasses.replace(job, image=image_id) if image_id else job  # run by id, never re-resolve the tag
        argv = docker_args(pinned, project_root, lock_token=lock_token, image_id=image_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 - argv from docker_args
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                return proc.wait()

    def stop(self, name: str, grace_s: int) -> None:
        subprocess.Popen(  # noqa: S603
            ["docker", "stop", "-t", str(grace_s), name],  # noqa: S607
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip

    def kill(self, name: str) -> None:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)  # noqa: S603,S607

    def state(self, name: str) -> ExitInfo:
        from vlm_pipeline.common.jobrun import ExitInfo

        out = subprocess.run(  # noqa: S603
            ["docker", "inspect", "-f", "{{json .State}}", name],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        st = json.loads(out)
        return ExitInfo(exit_code=int(st.get("ExitCode", -1)), oom_killed=bool(st.get("OOMKilled", False)))

    def remove(self, name: str) -> None:
        subprocess.run(["docker", "rm", name], capture_output=True, check=False)  # noqa: S603,S607


class RunnerError(RuntimeError):
    """vlm-runner refused or failed a request; ``reason`` is its rule name (never a traceback)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"runner: {reason}")
        self.reason = reason


class RunnerLauncher:
    """The compose path (ADR-015): typed requests to ``vlm-runner`` over a unix socket; no docker CLI in the app.

    The runner checks the image (IMAGES.lock + APPROVED, runs it by id), the GPU token, the mounts and "one job per
    run id" itself, at the trust boundary; ``checks_images`` tells ``run_job`` not to try (no docker here).
    """

    checks_images = False
    POLL_S = 2.0
    START_GRACE_S = 120.0  # a container the runner just launched may not be visible yet (image of ~30 GB)

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.last_image_id: str | None = None

    def _call(self, body: dict[str, object]) -> dict[str, object]:
        import socket

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(str(self.socket_path))
            conn.sendall(json.dumps(body).encode("utf-8"))
            conn.shutdown(socket.SHUT_WR)
            reply = json.loads(conn.makefile("rb").read() or b"{}")
        if not isinstance(reply, dict):
            raise RunnerError("bad_reply")
        return reply

    def image_id(self, image: str) -> str | None:
        """The runner's local id of ``image`` (cache identity); the gate itself stays in the runner's submit."""
        try:
            reply = self._call({"command": "image_id", "image_tag": image})
        except OSError:
            return None
        value = reply.get("image_id")
        return value if reply.get("ok") is True and isinstance(value, str) else None

    def run(
        self, job: ContainerJob, project_root: Path, *, lock_token: str | None, log_path: Path, image_id: str | None
    ) -> int:
        from vlm_pipeline.common.job_kinds import job_kind

        if not job.narrow or len(job.rw_dirs) != 1:
            raise RunnerError("not_a_narrowed_job")
        # The runner picks the entrypoint from the job kind; a job naming its own (vLLM: python3) sends it as argv[0].
        argv = (job.entrypoint, *job.argv) if job.entrypoint else tuple(job.argv)
        kind = job_kind(argv)
        if kind is None:
            raise RunnerError("argv_not_allowed")
        extra: dict[str, object] = {}
        if job.download_budget is not None:  # a catalog download (P018): its budget, and the token if the user set one
            extra["max_bytes"] = job.download_budget
            if job.hf_token:
                extra["hf_token"] = job.hf_token
        reply = self._call({
            **extra,
            "command": "submit", "run_id": job.name, "job_kind": kind, "image_tag": job.image, "argv": list(argv),
            "gpu": job.gpu, "memory_gb": job.memory_gb, "lock_token": lock_token,
            "run_dir": _app_path(job.rw_dirs[0], project_root),
            "inputs": [_app_path(p, project_root) for p in job.ro_inputs],
        })  # fmt: skip
        if reply.get("ok") is not True:
            raise RunnerError(str(reply.get("error", "refused")))
        self.last_image_id = str(reply.get("image_id") or "") or None
        submitted, seen = time.monotonic(), False
        while True:  # the runner owns the container; we only watch it
            try:
                running, info = self._status(f"vlm-{job.name}")
            except RunnerError as exc:
                # The runner starts `docker run` and answers at once: the container may not exist for a moment yet
                # (PM 2026-10-14: the first poll said not_found -> SYS-000, hiding the job's real exit). Only until
                # it has been seen once: a container that disappears afterwards is an error at once.
                if exc.reason == "not_found" and not seen and time.monotonic() - submitted < self.START_GRACE_S:
                    time.sleep(self.POLL_S)
                    continue
                raise
            seen = True
            if not running:
                return info.exit_code
            time.sleep(self.POLL_S)

    def _ref(self, command: str, name: str) -> dict[str, object]:
        return self._call({"command": command, "run_id": name.removeprefix("vlm-")})

    def stop(self, name: str, grace_s: int) -> None:
        self._ref("stop", name)  # the runner applies its own grace period

    def kill(self, name: str) -> None:
        self._ref("kill", name)

    def _status(self, name: str) -> tuple[bool, ExitInfo]:
        from vlm_pipeline.common.jobrun import ExitInfo

        reply = self._ref("status", name)
        if reply.get("ok") is not True:
            raise RunnerError(str(reply.get("error", "status")))
        code = reply.get("exit_code", -1)
        info = ExitInfo(exit_code=code if isinstance(code, int) else -1, oom_killed=bool(reply.get("oom_killed")))
        return bool(reply.get("running")), info

    def state(self, name: str) -> ExitInfo:
        return self._status(name)[1]

    def remove(self, name: str) -> None:
        self._ref("remove", name)


def _app_path(path: Path, project_root: Path) -> str:
    """``path`` in the runner protocol's namespace (``/app/<relative to the app's project root>``)."""
    rel = Path(os.path.normpath(path.absolute())).relative_to(Path(os.path.normpath(project_root.absolute())))
    return f"/app/{rel.as_posix()}"


def default_launcher() -> JobLauncher:
    """The runner when ``VLM_RUNNER_SOCKET`` is set (compose), else the local docker CLI (dev on the host).

    An app inside the compose stack (``VLM_IN_COMPOSE=1``) without a socket fails here: it must never fall back to a
    docker CLI or socket (bao-security 7.2 condition 1).
    """
    socket_path = os.environ.get("VLM_RUNNER_SOCKET")
    if socket_path:
        return RunnerLauncher(Path(socket_path))
    if os.environ.get("VLM_IN_COMPOSE") == "1":
        raise RunnerError("no_runner_socket")
    return LocalDockerLauncher()


__all__ = ["JobLauncher", "LocalDockerLauncher", "RunnerError", "RunnerLauncher", "default_launcher"]
