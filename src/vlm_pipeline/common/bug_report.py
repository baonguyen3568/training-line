"""Plain-text bug report of one failed run, for the "Sao chép báo cáo" button (G-3 action ``send_report``).

Built on the SERVER, never by the client, because it filters: every line goes through ``redact`` (credential shapes of
``secret_scan`` + ``key=value`` assignments of token/key/password/secret), host paths become project-relative, and
nothing user-made is copied: no image, no dataset row, no prompt text (only its length). The app never sends it
anywhere: the user copies it and hands it to whoever supports them. Stdlib + our own modules, host-safe.
"""

from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.common.run_log import RunLogError, tail_run_log
from vlm_pipeline.common.runinfo import read_manifest, read_small_json
from vlm_pipeline.common.secret_scan import PATTERNS

LOG_TAIL_LINES = 80
MAX_LINE_CHARS = 400
REDACTED = "[REDACTED]"
GUESSED = "[REDACTED?]"  # a long opaque string that MAY be a key: marked differently so the reader knows it is a guess
LOG_NAMES = ("stdout.log", "job.log", "container.log")
# key=value / key: value whose key names a credential; the value goes, the key stays (useful for debugging).
_ASSIGNMENT = re.compile(
    r"(?i)\b([\w.-]*(?:token|api[_-]?key|password|passwd|secret|auth|credential|bearer|cookie|session)[\w.-]*)"
    r"(\s*[=:]\s*)(\S+)"
)
# Wider than secret_scan.PATTERNS on purpose (bao-security): those gate the repo in CI and must stay precise; here a
# false positive only hides a harmless string from a report someone may paste publicly.
REDACT_EXTRA = {
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    "GITHUB_PAT": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}"),
    "AUTH_HEADER": re.compile(r"(?i)(authorization\s*:\s*(?:bearer|basic))\s+\S+"),
}
_URL_CREDENTIALS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
# 40+ chars of one token with letters AND digits (paths split on "/" so they stay readable); a sha256: digest stays.
_OPAQUE = re.compile(
    r"(?<!sha256:)(?<![A-Za-z0-9+=_-])(?=[A-Za-z0-9+_-]*\d)(?=[A-Za-z0-9+_-]*[A-Za-z])[A-Za-z0-9+_-]{40,}={0,2}"
    r"(?![A-Za-z0-9+=_-])"
)
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY-----")
# Params that may hold user text: only their length is reported.
PRIVATE_PARAMS = frozenset({"prompt", "text", "caption", "question", "answer"})


def redact(line: str, root: Path | None = None, *, guess: bool = True) -> str:
    """``line`` without credentials or absolute host paths of the project (and truncated).

    ``guess``: also hide long opaque strings (base64/hex >= 40, not a ``sha256:`` digest) as ``GUESSED``; off for the
    report header, whose git commit and image digest are facts the reader needs.
    """
    for pattern in PATTERNS.values():
        line = pattern.sub(REDACTED, line)
    for name, pattern in REDACT_EXTRA.items():
        repl = rf"\1 {REDACTED}" if name == "AUTH_HEADER" else REDACTED
        line = pattern.sub(repl, line)
    line = _URL_CREDENTIALS.sub(f"://{REDACTED}@", line)
    line = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", line)
    if guess:
        line = _OPAQUE.sub(GUESSED, line)
    if root is not None:
        line = line.replace(str(root) + os.sep, "").replace(str(root), ".")
    home = str(Path.home())
    line = line.replace(home, "~")
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " …"


def redact_lines(lines: Iterable[str], root: Path | None = None, *, guess: bool = True) -> list[str]:
    """``redact`` per line, and a whole PEM private-key block (BEGIN ... END, multi-line) dropped."""
    out: list[str] = []
    in_key = False
    for line in lines:
        if in_key or _PEM_BEGIN.search(line):
            if not in_key:
                out.append(REDACTED + " (private key block)")
            in_key = not _PEM_END.search(line)
            continue
        out.append(redact(line, root, guess=guess))
    return out


def _log_tail(run_dir: Path) -> tuple[str, list[str]] | None:
    """(name, last lines) of the first usable log, read through ``run_log`` (no link, no FIFO: bao-security B1)."""
    for name in LOG_NAMES:
        try:
            return name, tail_run_log(run_dir, LOG_TAIL_LINES, name=name)
        except RunLogError:
            continue
    return None


def _meminfo() -> dict[str, float]:
    try:
        from vlm_pipeline.common.memguard import read_meminfo

        snap = read_meminfo()
    except (OSError, ValueError):
        return {}
    return {"mem_total_gb": round(snap.mem_total_gb, 1), "mem_available_gb": round(snap.mem_available_gb, 1)}


def _params(params: Mapping[str, object]) -> Iterable[str]:
    for key in sorted(params):
        value = params[key]
        if key in PRIVATE_PARAMS and isinstance(value, str):
            yield f"  {key}: <{len(value)} chars, not copied>"
        else:
            yield f"  {key}: {value}"


def _error_line(code: str) -> str:
    try:
        from vlm_pipeline.common.error_catalog import load_catalog
        from vlm_pipeline.i18n import load

        entry = load_catalog().get(code)
        title = load("vi").get(f"error.{code}.title") or ""
        title_en = load("en").get(f"error.{code}.title") or ""
    except Exception:  # noqa: BLE001 - a report must never fail because of the text layer
        return f"Error: {code}"
    return f"Error: {code} {entry.slug if entry else ''} — {title} / {title_en}"


@dataclass(frozen=True)
class ReportContext:
    """What the route knows besides the run dir: the error shown on the card and the node it belongs to."""

    error_code: str | None = None
    error_params: Mapping[str, object] | None = None
    node_type: str | None = None
    node_id: str | None = None


def build_report(root: Path, run_dir: Path, ctx: ReportContext = ReportContext()) -> str:  # noqa: B008 - frozen
    """The report text. ``run_dir`` must be inside the project (the route resolves it from a run id)."""
    root, run_dir = root.resolve(), run_dir.resolve()
    if not run_dir.is_relative_to(root):
        raise ValueError("run dir outside the project")
    out: list[str] = ["# Báo cáo lỗi / Bug report", ""]
    code = ctx.error_code
    try:
        manifest = read_manifest(run_dir)
    except (OSError, ValueError):
        manifest = None
    if manifest is not None:
        code = code or manifest.error_code
    if code:
        out.append(_error_line(code))
    if ctx.error_params:
        out += ["Params:", *_params(ctx.error_params)]
    if ctx.node_type or ctx.node_id:
        out.append(f"Node: {ctx.node_type or '?'} ({ctx.node_id or '?'})")
    out.append(f"Run: {run_dir.relative_to(root).as_posix()}")
    if manifest is not None:
        out += [
            f"Status: {manifest.status} exit_code={manifest.exit_code} oom_killed={manifest.oom_killed}",
            f"Stage/task: {manifest.stage}/{manifest.task}",
            f"Code: git {manifest.git_commit} dirty={manifest.git_dirty}",
            f"Image: {manifest.container_image} {manifest.container_digest}",
            f"Memory: peak_gb={manifest.peak_mem_gb} host_min_available_gb={manifest.host_min_available_gb}"
            f" memguard_abort_gb={manifest.memguard_abort_gb}",
        ]
        if manifest.engine_warnings:
            out.append(f"Engine warnings: {', '.join(manifest.engine_warnings)}")
    try:
        engine = read_small_json(run_dir / "engine.json")  # no link, no FIFO, bounded
    except (OSError, ValueError):
        engine = None
    if isinstance(engine, dict):
        keep = ("engine", "vllm", "quantization", "dtype", "max_model_len", "gpu_memory_utilization", "enforce_eager")
        out.append("Engine: " + json.dumps({k: engine.get(k) for k in keep if k in engine}))
    mem = _meminfo()
    out.append(f"Host: {platform.machine()} {platform.release()} " + " ".join(f"{k}={v}" for k, v in mem.items()))
    text = redact_lines(out, root, guess=False)
    tail = _log_tail(run_dir)
    if tail is None:
        text += ["", "--- log: not readable (missing, or not a regular file) ---"]
    else:
        text += ["", f"--- last {LOG_TAIL_LINES} log lines ({tail[0]}) ---", *redact_lines(tail[1], root)]
    return "\n".join(text) + "\n"


__all__ = [
    "GUESSED",
    "LOG_TAIL_LINES",
    "PRIVATE_PARAMS",
    "REDACTED",
    "REDACT_EXTRA",
    "ReportContext",
    "build_report",
    "redact",
    "redact_lines",
]
