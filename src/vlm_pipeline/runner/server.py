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

import json
import logging
import os
import secrets
import socket
import socketserver
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
from vlm_pipeline.common.job_kinds import NGC_ENTRYPOINT, job_entrypoint
from vlm_pipeline.common.jobrun import STOP_GRACE_S, ContainerJob, docker_args
from vlm_pipeline.common.run_log import open_run_log
from vlm_pipeline.runner.protocol import (
    MAX_REQUEST_BYTES,
    ImageQuery,
    ProtocolError,
    RunRef,
    Submit,
    parse_request,
)

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
    # Download tokens (security download-egress §2): a host path the runner sees at the same path and the app never
    # mounts, so the docker daemon can bind ONE file into the download job. None: no token can be handed over.
    secrets_dir: Path | None = None
    egress_control: Path | None = None  # the egress proxy's control socket (download budgets); None: no downloads


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
        self._sweep_secrets()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ requests
    def handle(self, raw: bytes) -> dict[str, Any]:
        try:
            request = parse_request(raw)
            if isinstance(request, ImageQuery):
                result = {"ok": True, "image_id": self._image_id(request.image_tag)}
            elif isinstance(request, Submit):
                result = self._submit(request)
            else:
                result = self._run_ref(request)
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
        image_id, image_entrypoint = self._image_facts(sub.image_tag)
        check_pinned_image(self.cfg.host_project, sub.image_tag, image_id, require_approved=True)
        # The entrypoint comes from the job kind's closed set, never from the image or the request (security, ADR-015).
        entrypoint, command = job_entrypoint(sub.job_kind, sub.argv)  # the protocol already allowed this argv
        if entrypoint == NGC_ENTRYPOINT and image_entrypoint != [NGC_ENTRYPOINT]:
            return {"ok": False, "error": "entrypoint_mismatch"}  # not an NGC image: its wrapper is not there
        if sub.job_kind == "data" and sub.gpu:  # CPU work on files: never the GPU (nor its lock)
            return {"ok": False, "error": "data_without_gpu"}
        if sub.gpu:
            gpu_lock.verify_token(self.cfg.lock_path, sub.lock_token or "")
        download = sub.job_kind == "download"  # the ONLY kind with a network (HF-only proxy) and a token
        if download and sub.gpu:
            return {"ok": False, "error": "download_without_gpu"}
        egress_auth = self._register_egress(sub.run_id, sub.max_bytes or 0) if download else None
        secret = self._write_secret(sub.run_id, sub.hf_token) if sub.hf_token is not None else None
        try:
            job = ContainerJob(
                name=sub.run_id, image=image_id or sub.image_tag, argv=command, entrypoint=entrypoint, gpu=sub.gpu,
                memory_gb=sub.memory_gb,
                narrow=True, rw_dirs=(run_dir,), ro_inputs=inputs, code_root=self.cfg.host_project,
                code_files=(EVAL_WORKER,) if sub.job_kind == "eval" else (), uid=self.job_uid, gid=self.job_gid,
                egress=download, secret_file=secret, egress_auth=egress_auth,
            )  # fmt: skip
            argv = docker_args(job, self.cfg.data_root, lock_token=sub.lock_token)  # validates every mount source
            fd = open_run_log(run_dir, write=True)  # before the marker: a planted logs/ link refuses the job
            self._mark_started(sub.run_id, run_dir)  # one job per run id, at the trust boundary
        except BaseException:
            self._drop_secret(sub.run_id)
            if download:
                self._egress_call({"command": "unregister", "job": sub.run_id})
            raise
        with os.fdopen(fd, "ab") as out:
            proc = self._popen(argv, stdout=out, stderr=subprocess.STDOUT)
        if download and callable(getattr(proc, "wait", None)):  # security S2: token and budget live as long as
            threading.Thread(target=self._drop_after, args=(proc, sub.run_id), daemon=True).start()  # the job
        return {"ok": True, "container": f"vlm-{sub.run_id}", "image_id": image_id}

    def _drop_after(self, proc: Any, run_id: str) -> None:
        try:
            proc.wait()  # the attached `docker run` returns when the container exits
        finally:
            self._drop_secret(run_id)
            self._egress_call({"command": "unregister", "job": run_id})

    def _register_egress(self, run_id: str, max_bytes: int) -> str:
        """A fresh proxy credential for this job with its byte budget; the job's HTTPS_PROXY carries it."""
        key = secrets.token_hex(16)
        reply = self._egress_call({"command": "register", "job": run_id, "key": key, "max_bytes": max_bytes})
        if reply.get("ok") is not True:
            raise ProtocolError("egress_unavailable")
        return f"{run_id}:{key}"

    def _egress_call(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.cfg.egress_control is None:
            return {"ok": False}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(5)
                conn.connect(str(self.cfg.egress_control))
                conn.sendall(json.dumps(body).encode())
                conn.shutdown(socket.SHUT_WR)
                reply = json.loads(conn.recv(4096) or b"{}")
        except (OSError, ValueError):
            return {"ok": False}
        return reply if isinstance(reply, dict) else {"ok": False}

    def _write_secret(self, run_id: str, token: str) -> Path:
        """The token as ONE file for ONE job: 0400, created exclusively, never through a link; removed with the job."""
        if self.cfg.secrets_dir is None:
            raise ProtocolError("no_secret_store")
        root = self.cfg.secrets_dir
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / run_id
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o400)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(token)
        return path

    def _drop_secret(self, run_id: str) -> None:
        if self.cfg.secrets_dir is not None:
            (self.cfg.secrets_dir / run_id).unlink(missing_ok=True)

    def _sweep_secrets(self) -> None:
        """At start: a token file is only ever for a running job; leftovers of a crashed runner go."""
        root = self.cfg.secrets_dir
        if root is None or not root.is_dir() or root.is_symlink():
            return
        for leftover in root.iterdir():
            if leftover.is_file() or leftover.is_symlink():
                leftover.unlink(missing_ok=True)

    def _run_ref(self, ref: RunRef) -> dict[str, Any]:
        name = f"vlm-{ref.run_id}"
        state = self._ours(name, ref.run_id)
        if state is None:
            return {"ok": False, "error": "not_found"}
        if ref.command == "status":
            if not state.get("running"):
                self._drop_secret(ref.run_id)  # exited: the token file must not outlive the job
            return {"ok": True, **state}
        if ref.command == "stop":
            self._docker(["stop", "-t", str(STOP_GRACE_S), name])
        elif ref.command == "kill":
            self._docker(["kill", name])
        elif ref.command == "remove":
            if state["running"]:
                return {"ok": False, "error": "still_running"}
            self._docker(["rm", name])
            self._drop_secret(ref.run_id)  # the job is gone: so is its token
        elif ref.command == "logs":
            return {"ok": True, "text": self._tail(ref.run_id, ref.tail)}
        return {"ok": True}

    # ------------------------------------------------------------------------------------------ helpers
    def _image_id(self, tag: str) -> str | None:
        return self._image_facts(tag)[0]

    def _image_facts(self, tag: str) -> tuple[str | None, list[str] | None]:
        """(local image id, its Config.Entrypoint) in one inspect; (None, None) when the image is not here."""
        out = self._docker(["image", "inspect", "-f", "{{.Id}}|{{json .Config.Entrypoint}}", tag])
        if out.returncode != 0 or "|" not in out.stdout:
            return None, None
        image_id, raw = out.stdout.strip().split("|", 1)
        try:
            entrypoint = json.loads(raw)
        except ValueError:
            entrypoint = None
        valid = isinstance(entrypoint, list) and all(isinstance(a, str) for a in entrypoint)
        return image_id or None, entrypoint if valid else None

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
            fd = open_run_log(Path(marker["run_dir"]))
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
        secrets_dir=Path(env["VLM_RUNNER_SECRETS"]) if env.get("VLM_RUNNER_SECRETS") else None,
        egress_control=Path(env["VLM_EGRESS_CONTROL"]) if env.get("VLM_EGRESS_CONTROL") else None,
    )


def main() -> None:  # pragma: no cover - the container entrypoint
    logging.basicConfig(level=logging.INFO)
    cfg = config_from_env()
    with RunnerServer(Runner(cfg), cfg.socket_path) as server:
        server.serve_forever()


__all__ = ["LABEL", "Runner", "RunnerConfig", "RunnerServer", "config_from_env"]


if __name__ == "__main__":  # pragma: no cover - `python -m vlm_pipeline.runner.server` (runner image CMD)
    main()
