"""``vlm-runner``: the one process with the docker socket (ADR-015, P015 §3). Stdlib only, unix socket, JSON.

One request per connection (the client shuts its write side), at most 64 KiB, parsed by ``protocol.parse_request``
(closed language: no docker flag, mount, env or free container name ever comes from the app). For ``submit``:

1. app paths (``/app/...``) -> host paths (``HostPaths``: code under the host project, data under the data root);
2. the image tag must be pinned and APPROVED in ``docker/IMAGES.lock`` and run BY ID (``check_pinned_image``);
3. a GPU job's ``lock_token`` must be the current GPU lock's (``gpu_lock.verify_token``);
4. one job per run id, enforced HERE (a marker in the runner's own state dir, which no app or job can write):
   the app's ``run_claims`` keeps its ordering role, the trust boundary does not rely on it;
5. ``docker_args`` with the narrowed mounts (P017 step 6: data root = input root, N3), label ``vlm.job=<run_id>``;
6. the container is started and its output appended to ``<run dir>/logs/stdout.log``.

``stop`` / ``kill`` / ``status`` / ``remove`` / ``logs`` act only on a container labelled
``project=vlm-training`` AND ``vlm.job=<that run id>``. Every request is audited (``runner.jsonl``, no env).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import socketserver
import stat
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.host_paths import HostPaths
from vlm_pipeline.common.images import ImageMismatchError, check_pinned_image
from vlm_pipeline.common.jobrun import STOP_GRACE_S, ContainerJob, docker_args
from vlm_pipeline.runner.protocol import MAX_REQUEST_BYTES, ProtocolError, RunRef, Submit, parse_request

logger = logging.getLogger(__name__)
LABEL = "project=vlm-training"
EVAL_WORKER = "scripts/qa/eval_cord_worker.py"  # bound read-only for eval jobs (security step 6 Q3)
LOG_TAIL_MAX_BYTES = 1 << 20

Docker = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _docker(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv built here from typed fields, never from the request
        ["docker", *argv],  # noqa: S607 - the docker CLI from PATH inside the runner image
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@dataclass
class RunnerConfig:
    socket_path: Path
    host_project: Path  # code (src/, configs/, docker/IMAGES.lock), mounted read-only into the runner at this path
    data_root: Path  # VLM_DATA_ROOT on the host, mounted into the runner at this same path
    state_dir: Path  # runner-private (started markers, audit): never mounted into the app or a job
    lock_path: Path
    app_root: PurePosixPath = field(default_factory=lambda: PurePosixPath("/app"))
    # Jobs run as this uid/gid (security R1): never the runner's own (often root, for the docker socket).
    job_uid: int | None = None  # default: the owner of the data root
    job_gid: int | None = None
    allowed_peer_uids: frozenset[int] = frozenset()  # who may talk to the socket (N2); empty: this process's uid


def _open_run_log(run_dir: Path, *, write: bool) -> int:
    """The fd of ``<run_dir>/logs/stdout.log``, reached through dir fds, never through a link (security R2).

    The run dir belongs to the job (and the app writes there too): ``logs`` may have been swapped for a symlink and
    ``stdout.log`` for a FIFO. Every step is ``O_NOFOLLOW``, the file is opened ``O_NONBLOCK`` (a FIFO never blocks the
    runner) and must be a regular file; anything else raises ``ValueError`` (submit refused, logs empty).
    """
    try:
        run_fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if write:
                with contextlib.suppress(FileExistsError):
                    os.mkdir("logs", 0o755, dir_fd=run_fd)
            logs_fd = os.open("logs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=run_fd)
        finally:
            os.close(run_fd)
        try:
            flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT) if write else os.O_RDONLY
            fd = os.open("stdout.log", flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o644, dir_fd=logs_fd)
        finally:
            os.close(logs_fd)
    except OSError as exc:  # ELOOP / ENOTDIR (a link), ENXIO (a FIFO with no reader)...
        raise ValueError(f"run log unusable ({exc.strerror})") from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("stdout.log is not a regular file")
    return fd


class Runner:
    def __init__(self, cfg: RunnerConfig, *, docker: Docker = _docker, popen: Callable[..., Any] = subprocess.Popen):
        st = cfg.data_root.stat()
        self.job_uid = cfg.job_uid if cfg.job_uid is not None else st.st_uid
        self.job_gid = cfg.job_gid if cfg.job_gid is not None else st.st_gid
        if self.job_uid == 0 or self.job_gid == 0:
            raise ValueError("jobs must not run as root: set VLM_JOB_UID/VLM_JOB_GID or own the data root")
        self.cfg = cfg
        self.paths = HostPaths(cfg.app_root, PurePosixPath(cfg.host_project), PurePosixPath(cfg.data_root))
        self._docker = docker
        self._popen = popen
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ requests
    def handle(self, raw: bytes) -> dict[str, Any]:
        try:
            request = parse_request(raw)
            result = self._submit(request) if isinstance(request, Submit) else self._run_ref(request)
        except ProtocolError as exc:
            result = {"ok": False, "error": exc.reason}
        except ImageMismatchError:
            result = {"ok": False, "error": "image_not_approved"}
        except gpu_lock.GpuLockError:
            result = {"ok": False, "error": "gpu_lock_not_held"}
        except ValueError:  # a mount source the host checks refused (allowlist, symlink, credential...): logged
            logger.warning("runner refused a mount", exc_info=True)
            result = {"ok": False, "error": "mount_refused"}
        except Exception:  # never a traceback to the app
            logger.exception("runner request failed")
            result = {"ok": False, "error": "internal"}
        self._audit(raw, result)
        return result

    def _host(self, path: PurePosixPath) -> Path:
        return Path(str(self.paths.to_host(path)))

    def _submit(self, sub: Submit) -> dict[str, Any]:
        run_dir, inputs = self._host(sub.run_dir), tuple(self._host(p) for p in sub.inputs)
        image_id = self._image_id(sub.image_tag)
        check_pinned_image(self.cfg.host_project, sub.image_tag, image_id, require_approved=True)
        if sub.gpu:
            gpu_lock.verify_token(self.cfg.lock_path, sub.lock_token or "")
        job = ContainerJob(
            name=sub.run_id, image=image_id or sub.image_tag, argv=sub.argv, gpu=sub.gpu, memory_gb=sub.memory_gb,
            narrow=True, rw_dirs=(run_dir,), ro_inputs=inputs, code_root=self.cfg.host_project,
            code_files=(EVAL_WORKER,) if sub.job_kind == "eval" else (), uid=self.job_uid, gid=self.job_gid,
        )  # fmt: skip
        argv = docker_args(job, self.cfg.data_root, lock_token=sub.lock_token)  # validates every mount source
        fd = _open_run_log(run_dir, write=True)  # before the marker: a planted logs/ link refuses the job
        self._mark_started(sub.run_id, run_dir)  # one job per run id, at the trust boundary
        with os.fdopen(fd, "ab") as out:
            self._popen(argv, stdout=out, stderr=subprocess.STDOUT)
        return {"ok": True, "container": f"vlm-{sub.run_id}", "image_id": image_id}

    def _run_ref(self, ref: RunRef) -> dict[str, Any]:
        name = f"vlm-{ref.run_id}"
        state = self._ours(name, ref.run_id)
        if state is None:
            return {"ok": False, "error": "not_found"}
        if ref.command == "status":
            return {"ok": True, **state}
        if ref.command == "stop":
            self._docker(["stop", "-t", str(STOP_GRACE_S), name])
        elif ref.command == "kill":
            self._docker(["kill", name])
        elif ref.command == "remove":
            if state["running"]:
                return {"ok": False, "error": "still_running"}
            self._docker(["rm", name])
        elif ref.command == "logs":
            return {"ok": True, "text": self._tail(ref.run_id, ref.tail)}
        return {"ok": True}

    # ------------------------------------------------------------------------------------------ helpers
    def _image_id(self, tag: str) -> str | None:
        out = self._docker(["image", "inspect", "-f", "{{.Id}}", tag])
        return out.stdout.strip() or None if out.returncode == 0 else None

    def _ours(self, name: str, run_id: str) -> dict[str, Any] | None:
        """The container's state if it carries BOTH our labels, else None (never touch another service)."""
        out = self._docker(["inspect", "-f", "{{json .Config.Labels}}|{{json .State}}", name])
        if out.returncode != 0 or "|" not in out.stdout:
            return None
        labels_raw, state_raw = out.stdout.strip().split("|", 1)
        try:
            labels, state = json.loads(labels_raw) or {}, json.loads(state_raw)
        except ValueError:
            return None
        if labels.get("project") != "vlm-training" or labels.get("vlm.job") != run_id:
            return None
        return {
            "running": bool(state.get("Running")),
            "exit_code": int(state.get("ExitCode", -1)),
            "oom_killed": bool(state.get("OOMKilled", False)),
        }

    def _mark_started(self, run_id: str, run_dir: Path) -> None:
        started = self.cfg.state_dir / "started"
        started.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(started / run_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except FileExistsError:
            raise ProtocolError("run_already_started") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_dir": str(run_dir), "ts": time.time()}) + "\n")

    def _tail(self, run_id: str, lines: int) -> str:
        try:
            marker = json.loads((self.cfg.state_dir / "started" / run_id).read_text(encoding="utf-8"))
            fd = _open_run_log(Path(marker["run_dir"]), write=False)
        except (OSError, ValueError, KeyError):
            return ""
        with os.fdopen(fd, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, size - LOG_TAIL_MAX_BYTES))
            text = fh.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def _audit(self, raw: bytes, result: dict[str, Any]) -> None:
        try:
            body = json.loads(raw) if len(raw) <= MAX_REQUEST_BYTES else {}
        except ValueError:
            body = {}
        line = {
            "ts": time.time(),
            "command": body.get("command") if isinstance(body, dict) else None,
            "run_id": body.get("run_id") if isinstance(body, dict) else None,
            "ok": result.get("ok"),
            "error": result.get("error"),
            "image_id": result.get("image_id"),
        }  # never argv or env: the request is typed, its shape is enough
        with self._lock:
            path = self.cfg.state_dir / "runner.jsonl"
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")


MAX_CONNECTIONS = 16
READ_TIMEOUT_S = 10.0


def _peer_uid(conn: socket.socket) -> int:
    pid_uid_gid = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return int(struct.unpack("3i", pid_uid_gid)[1])


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: RunnerServer = self.server  # type: ignore[assignment]
        if not server.slots.acquire(blocking=False):  # N1: a slow or stuck client cannot hold every thread
            self._reply({"ok": False, "error": "busy"})
            return
        try:
            self.request.settimeout(READ_TIMEOUT_S)
            try:
                raw = self.rfile.read(MAX_REQUEST_BYTES + 1)  # always drained first: a refusal is a clean reply
            except TimeoutError:
                raw = b""
            if _peer_uid(self.request) not in server.allowed_uids:  # N2: not only the socket's group
                self._reply({"ok": False, "error": "peer_not_allowed"})
                return
            self._reply(server.runner.handle(raw))
        finally:
            server.slots.release()

    def _reply(self, body: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(body).encode("utf-8"))


class RunnerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, runner: Runner, socket_path: Path) -> None:
        socket_path.unlink(missing_ok=True)
        super().__init__(str(socket_path), _Handler)
        socket_path.chmod(0o660)  # the app's group only (compose: group_add on the app service)
        self.runner = runner
        self.allowed_uids = runner.cfg.allowed_peer_uids or frozenset({os.getuid()})
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)


def config_from_env(environ: dict[str, str] | None = None) -> RunnerConfig:
    env = os.environ if environ is None else environ
    return RunnerConfig(
        socket_path=Path(env.get("VLM_RUNNER_SOCKET", "/run/vlm-runner/runner.sock")),
        host_project=Path(env["VLM_HOST_PROJECT_ROOT"]),
        data_root=Path(env["VLM_DATA_ROOT"]),
        state_dir=Path(env.get("VLM_RUNNER_STATE", "/var/lib/vlm-runner")),
        lock_path=Path(env.get("VLM_GPU_LOCK_PATH", "/run/vlm-lock/GPU_LOCK")),
        job_uid=int(env["VLM_JOB_UID"]) if env.get("VLM_JOB_UID") else None,
        job_gid=int(env["VLM_JOB_GID"]) if env.get("VLM_JOB_GID") else None,
        allowed_peer_uids=frozenset(int(u) for u in env.get("VLM_RUNNER_ALLOWED_UIDS", "").split(",") if u.strip()),
    )


def main() -> None:  # pragma: no cover - the container entrypoint
    logging.basicConfig(level=logging.INFO)
    cfg = config_from_env()
    with RunnerServer(Runner(cfg), cfg.socket_path) as server:
        server.serve_forever()


__all__ = ["LABEL", "Runner", "RunnerConfig", "RunnerServer", "config_from_env"]
