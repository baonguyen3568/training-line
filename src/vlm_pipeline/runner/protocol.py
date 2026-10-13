"""The closed request language between the app and ``vlm-runner`` (ADR-015, P015 §3.1). Stdlib only.

The runner holds the docker socket, so it is root-equivalent: it accepts typed requests, never a docker flag, a mount,
an env var, a container name or a free image name. Everything here is a pure check on the decoded JSON; the server
(socket, docker) builds on it. Refusals name the rule (``ProtocolError.reason``), never echo the input.

``submit`` fields:
* ``run_id``: ``^[0-9]{8}-[0-9]{4}-[a-z0-9-]{1,48}$`` (container ``vlm-job-<run_id>``);
* ``job_kind``: train | eval | quantize | probe | download | serve;
* ``image_tag``: a tag; the server checks IMAGES.lock + APPROVED + image id (SEC-004), not this module;
* ``argv``: must start with one of the kind's allowed prefixes (ALLOWED_ARGV); the rest are plain strings;
* ``gpu``, ``memory_gb`` (1..MEMORY_CAP_GB), ``lock_token`` (required with gpu);
* ``hf_token`` (optional, ``download`` jobs ONLY): handed to the job as a mounted file, never env, never logged;
* ``run_dir``: the job's writable dir, an app path ``/app/outputs/runs/<run_id>`` or ``/app/outputs/graph/runs/<dir>``;
* ``inputs``: app paths mounted READ-ONLY (earlier nodes' outputs, uploads, datasets), each under INPUT_ROOTS.
The paths are translated to host paths by ``common.host_paths`` and checked there (no symlink, owner, kind).

What the runner protects is the HOST, not the job: a job container is already treated as running untrusted code
(network none, read-only mounts, cap_drop ALL, non-root). E.g. the ``automodel`` prefix takes a YAML config whose
``_target_`` entries can call any importable callable INSIDE the job; that is accepted and not this module's concern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from vlm_pipeline.common.job_kinds import ALLOWED_ARGV, argv_allowed
from vlm_pipeline.common.run_claims import RUN_ID  # one definition: the claim and the request agree on run ids

MAX_REQUEST_BYTES = 64 * 1024
MAX_ARGV = 128
MAX_ARG_CHARS = 4096
MAX_INPUTS = 32
MEMORY_CAP_GB = 96
APP_ROOT = PurePosixPath("/app")
UPLOAD_NAME = re.compile(r"^[0-9a-f]{32}\.png$")
IMAGE_TAG = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}:[A-Za-z0-9._-]{1,64}$")
LOCK_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
HF_TOKEN = re.compile(r"^hf_[A-Za-z0-9]{20,100}$")  # the shape of a Hugging Face token
COMMANDS = frozenset({"submit", "stop", "kill", "status", "logs", "remove", "image_id"})
JOB_KINDS = frozenset({"train", "eval", "quantize", "probe", "download", "serve"})
RUN_DIR_PARENTS = (APP_ROOT / "outputs" / "runs", APP_ROOT / "outputs" / "graph" / "runs")
INPUT_ROOTS = (
    APP_ROOT / "outputs" / "runs",
    APP_ROOT / "outputs" / "graph" / "runs",
    APP_ROOT / "data",
    APP_ROOT / ".cache" / "hf" / "hub",
    APP_ROOT / ".cache" / "prepared",
    APP_ROOT / "uploads",  # one file each, named <32 hex>.png (checked below)
)


class ProtocolError(ValueError):
    """A refused request. ``reason`` is a rule name (safe to log and return); the message never holds input."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Submit:
    run_id: str
    job_kind: str
    image_tag: str
    argv: tuple[str, ...]
    gpu: bool
    memory_gb: int
    lock_token: str | None
    run_dir: PurePosixPath
    inputs: tuple[PurePosixPath, ...]
    hf_token: str | None = field(default=None, repr=False)  # never in a repr, a log or an audit line


@dataclass(frozen=True)
class RunRef:
    """stop / kill / status / remove (and logs with ``since``/``tail``): only ever a run id, never a name."""

    command: str
    run_id: str
    tail: int = 200
    since: float = 0.0


def _closed(body: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    if set(body) - allowed:
        raise ProtocolError("unknown_field")
    if required - set(body):
        raise ProtocolError("missing_field")


def _text(value: Any, reason: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or (pattern is not None and not pattern.fullmatch(value)):
        raise ProtocolError(reason)
    return value


def _app_path(value: Any, reason: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.isprintable() or "\\" in value:
        raise ProtocolError(reason)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "." in value.split("/") or "//" in value:
        raise ProtocolError(reason)
    return path


def _run_dir(value: Any, run_id: str) -> PurePosixPath:
    """The job's own dir: named exactly ``run_id`` (never another job's), directly under a run-dir parent."""
    path = _app_path(value, "bad_run_dir")
    if path.parent in RUN_DIR_PARENTS and path.name == run_id:
        return path
    raise ProtocolError("bad_run_dir")


def _inputs(value: Any, run_dir: PurePosixPath) -> tuple[PurePosixPath, ...]:
    if not isinstance(value, list) or len(value) > MAX_INPUTS:
        raise ProtocolError("bad_inputs")
    paths = tuple(_app_path(item, "bad_inputs") for item in value)
    for path in paths:
        if not any(path.is_relative_to(root) and path != root for root in INPUT_ROOTS):
            raise ProtocolError("input_outside_roots")
        if path.is_relative_to(run_dir) or run_dir.is_relative_to(path):
            raise ProtocolError("input_overlaps_run_dir")  # the rw dir is never also an input, nor inside one
        if path.parent == APP_ROOT / "uploads" and not UPLOAD_NAME.fullmatch(path.name):
            raise ProtocolError("bad_upload")
        if path.is_relative_to(APP_ROOT / "uploads") and path.parent != APP_ROOT / "uploads":
            raise ProtocolError("bad_upload")
    for i, a in enumerate(paths):
        if any(i != j and (a == b or b.is_relative_to(a)) for j, b in enumerate(paths)):
            raise ProtocolError("inputs_nested")  # no mount on top of another mount
    return paths


def _argv(value: Any, kind: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_ARGV:
        raise ProtocolError("bad_argv")
    if not all(isinstance(a, str) and 0 < len(a) <= MAX_ARG_CHARS and a.isprintable() for a in value):
        raise ProtocolError("bad_argv")
    argv = tuple(value)
    if not argv_allowed(kind, argv):
        raise ProtocolError("argv_not_allowed")
    return argv


def parse_submit(body: dict[str, Any]) -> Submit:
    fields = {"run_id", "job_kind", "image_tag", "argv", "gpu", "memory_gb", "lock_token", "run_dir", "inputs"}
    _closed(body, fields | {"command", "hf_token"}, fields - {"lock_token"})
    run_id = _text(body["run_id"], "bad_run_id", RUN_ID)
    kind = _text(body["job_kind"], "bad_job_kind")
    if kind not in JOB_KINDS:
        raise ProtocolError("bad_job_kind")
    gpu, memory = body["gpu"], body["memory_gb"]
    if not isinstance(gpu, bool):
        raise ProtocolError("bad_gpu")
    if isinstance(memory, bool) or not isinstance(memory, int) or not 1 <= memory <= MEMORY_CAP_GB:
        raise ProtocolError("bad_memory_gb")
    token = body.get("lock_token")
    if token is not None:
        token = _text(token, "bad_lock_token", LOCK_TOKEN)
    if gpu and token is None:
        raise ProtocolError("gpu_needs_lock_token")
    hf_token = body.get("hf_token")
    if hf_token is not None:
        if kind != "download":  # security download-egress §2: only the download job ever receives the token
            raise ProtocolError("token_not_allowed")
        if not isinstance(hf_token, str) or not HF_TOKEN.fullmatch(hf_token):
            raise ProtocolError("bad_hf_token")
    run_dir = _run_dir(body["run_dir"], run_id)
    return Submit(
        run_id=run_id,
        job_kind=kind,
        image_tag=_text(body["image_tag"], "bad_image_tag", IMAGE_TAG),
        argv=_argv(body["argv"], kind),
        gpu=gpu,
        memory_gb=memory,
        lock_token=token,
        run_dir=run_dir,
        inputs=_inputs(body["inputs"], run_dir),
        hf_token=hf_token,
    )


@dataclass(frozen=True)
class ImageQuery:
    """Read-only: the local id of a pinned tag (the app has no docker CLI; its cache keys need the id)."""

    image_tag: str


def parse_run_ref(body: dict[str, Any]) -> RunRef:
    command = body["command"]
    if command == "logs":
        _closed(body, {"command", "run_id", "tail", "since"}, {"command", "run_id"})
        tail, since = body.get("tail", 200), body.get("since", 0.0)
        if isinstance(tail, bool) or not isinstance(tail, int) or not 0 < tail <= 10_000:
            raise ProtocolError("bad_tail")
        if isinstance(since, bool) or not isinstance(since, int | float) or since < 0:
            raise ProtocolError("bad_since")
        return RunRef(command, _text(body["run_id"], "bad_run_id", RUN_ID), tail, float(since))
    _closed(body, {"command", "run_id"}, {"command", "run_id"})
    return RunRef(command, _text(body["run_id"], "bad_run_id", RUN_ID))


def parse_request(raw: bytes) -> Submit | RunRef | ImageQuery:
    """One request (JSON object, <= 64 KiB) -> its typed form, or ProtocolError naming the rule."""
    if len(raw) > MAX_REQUEST_BYTES:
        raise ProtocolError("request_too_large")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ProtocolError("not_json") from None
    if not isinstance(body, dict):
        raise ProtocolError("not_json")
    command = body.get("command")
    if command not in COMMANDS:
        raise ProtocolError("unknown_command")
    if command == "image_id":
        _closed(body, {"command", "image_tag"}, {"command", "image_tag"})
        return ImageQuery(_text(body["image_tag"], "bad_image_tag", IMAGE_TAG))
    return parse_submit(body) if command == "submit" else parse_run_ref(body)


__all__ = [
    "ALLOWED_ARGV",
    "INPUT_ROOTS",
    "JOB_KINDS",
    "MAX_REQUEST_BYTES",
    "ImageQuery",
    "ProtocolError",
    "RunRef",
    "Submit",
    "parse_request",
]
