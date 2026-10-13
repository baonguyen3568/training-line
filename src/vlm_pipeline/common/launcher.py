"""How a job container is started, stopped and inspected (ADR-015): one interface, two implementations.

``LocalDockerLauncher`` (dev on the host, today) calls the docker CLI. ``RunnerLauncher`` (compose, M3) will talk to
the ``vlm-runner`` sidecar over a unix socket: the app then has no docker socket, no docker CLI and no token. Both
build the container from the same ``jobrun.docker_args`` (one source for mounts, secrets masking and env); the runner
receives the typed job and builds the argv itself, never an argv from the app.
"""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vlm_pipeline.common.jobrun import ContainerJob, ExitInfo


class JobLauncher(Protocol):
    def image_id(self, image: str) -> str | None: ...

    def run(self, job: ContainerJob, project_root: Path, *, lock_token: str | None, log_path: Path) -> int:
        """Start the container, block until it exits (output appended to ``log_path``); returns the exit code."""
        ...

    def stop(self, name: str, grace_s: int) -> None:
        """Graceful stop, NON-blocking (the memory watchdog keeps polling during the grace period)."""
        ...

    def kill(self, name: str) -> None: ...

    def state(self, name: str) -> ExitInfo: ...

    def remove(self, name: str) -> None: ...


class LocalDockerLauncher:
    """The docker CLI on the host (dev mode; the behaviour run_job always had)."""

    def image_id(self, image: str) -> str | None:
        try:
            return subprocess.run(  # noqa: S603
                ["docker", "image", "inspect", "-f", "{{.Id}}", image],  # noqa: S607
                capture_output=True, text=True, check=True,
            ).stdout.strip() or None  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            return None

    def run(self, job: ContainerJob, project_root: Path, *, lock_token: str | None, log_path: Path) -> int:
        from vlm_pipeline.common.jobrun import docker_args

        argv = docker_args(job, project_root, lock_token=lock_token, image_id=self.image_id(job.image))
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


__all__ = ["JobLauncher", "LocalDockerLauncher"]
