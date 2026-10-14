"""Check a user dataset folder and normalize its labels (node ``data.local_folder``, M1b B-3b). Host, CPU only.

Order of checks (cheap first, P4-2 data quality §2.1/2.2): the labels file, each line's shape, each image file
(inside the folder, regular, allowed format, size/pixel limits, longest side), duplicates, then the whole set
(at least one usable sample). Problems are collected, not raised one by one, so the user sees every bad line at
once; the caller decides whether any problem blocks (it does: a dataset with a broken line is not half-used).
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_pipeline.common.image_rules import ImageRuleError, probe_image
from vlm_pipeline.common.local_records import Record, image_name_ok

LABELS_FILE = "labels.jsonl"
MAX_LABELS_BYTES = 50 * 1024 * 1024
MAX_PROMPT_CHARS = 2000
MAX_PROBLEMS = 50  # enough to fix a batch; the count says how many more


@dataclass
class ScanResult:
    records: list[Record] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)  # {line, image?, reason}
    problem_count: int = 0
    # Preview only (H-prompt): the distinct questions of the good lines ("" = none), at most two kept (two = mixed).
    questions: set[str] = field(default_factory=set)

    def add(self, line: int, reason: str, image: str | None = None) -> None:
        self.problem_count += 1
        if len(self.problems) < MAX_PROBLEMS:
            self.problems.append({"line": line, "reason": reason, **({"image": image} if image else {})})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_image_name(name: str) -> str | None:
    """The image name as stored: ``./a.png`` -> ``a.png``, ``a//b.png`` -> ``a/b.png`` (how people write names).

    None if the result still breaks the Record rule (``../x``, ``/etc/x``, ``a/../../x``...).
    """
    if not name or name.startswith(("/", "~")) or "\\" in name:
        return None
    norm = posixpath.normpath(name)
    return norm if image_name_ok(norm) else None


def _image_path(folder: Path, name: str) -> Path | None:
    """The file ``name`` (already normalized) inside ``folder`` (no escape through a symlinked dir either)."""
    if not image_name_ok(name):  # the Record rule: what fails here would fail Record validation later
        return None
    path = folder / name
    return path if path.resolve().is_relative_to(folder) else None


def _target_ok(target: Any) -> bool:
    if isinstance(target, str):
        return bool(target.strip())
    return isinstance(target, dict | list) and bool(target)


def _labels_problem(labels: Path) -> str | None:
    try:
        st = labels.lstat()
    except FileNotFoundError:
        return "labels_missing"
    if not stat.S_ISREG(st.st_mode):
        return "labels_not_a_file"
    return "labels_too_large" if st.st_size > MAX_LABELS_BYTES else None


def _check_line(  # noqa: PLR0911 - flat check table, one early return per rule (cheap first)
    folder: Path, raw: bytes, seen: set[str], max_side: int, *, hashed: bool = True
) -> Record | tuple[str, str | None]:
    """One labels line -> its Record, or (reason, image name if known)."""
    try:
        item = json.loads(raw.decode("utf-8"))  # bytes: a non-UTF-8 line (cp1258/UTF-16) is a line problem
    except ValueError:  # includes UnicodeDecodeError
        return "not_json", None
    if not isinstance(item, dict) or set(item) - {"image", "target", "prompt"}:
        return "bad_keys", None
    raw_name, target, prompt = item.get("image"), item.get("target"), item.get("prompt")
    name = normalize_image_name(raw_name) if isinstance(raw_name, str) else None
    if name is None or (path := _image_path(folder, name)) is None:
        return "bad_image_name", None
    if not _target_ok(target):
        return "empty_target", name
    if prompt is not None and (not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS):
        return "bad_prompt", name
    if name in seen:
        return "duplicate_image", name
    try:
        width, height, _fmt = probe_image(path)
    except ImageRuleError as exc:
        return f"image_{exc.reason}", name
    if max(width, height) > max_side:
        return "image_too_big_side", name
    return Record(image=name, target=target, prompt=prompt, sha256=_sha256(path) if hashed else _NO_HASH)


def scan_folder(folder: Path, *, max_samples: int = 0, max_image_side_px: int = 4096) -> ScanResult:
    """Validate ``folder/labels.jsonl`` and its images; ``max_samples`` keeps the first N good lines (0 = all)."""
    folder = folder.resolve()
    result = ScanResult()
    labels = folder / LABELS_FILE
    if problem := _labels_problem(labels):
        result.add(0, problem)
        return result
    seen: set[str] = set()
    with labels.open("rb") as fh:
        for n, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            if max_samples and len(result.records) >= max_samples:
                break
            checked = _check_line(folder, raw, seen, max_image_side_px)
            if isinstance(checked, Record):
                seen.add(checked.image)
                result.records.append(checked)
            else:
                result.add(n, *checked)
    if not result.records and not result.problem_count:
        result.add(0, "no_samples")
    return result


_NO_HASH = "0" * 64  # preview only: headers are checked, image bytes are not read
PREVIEW_BUDGET_S = 3.0
_OVER_BUDGET = ScanResult()  # cached "no verdict": a big folder is timed once per change, not on every validate
_PREVIEW_CACHE: OrderedDict[tuple[object, ...], ScanResult] = OrderedDict()
_PREVIEW_CACHE_SIZE = 32
_PREVIEW_LOCK = threading.Lock()  # validate runs in the API's thread pool; never held while scanning


def preview_folder(
    folder: Path, *, max_image_side_px: int = 4096, budget_s: float = PREVIEW_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
) -> ScanResult | None:  # fmt: skip
    """The scan's verdict on ``folder`` from the image HEADERS only (no hashing), for the canvas (DATA-V).

    None when the folder takes longer than ``budget_s`` (no verdict: the run's full scan decides). Cached until
    labels.jsonl or the folder itself changes (an image replaced in place is only seen by the run's scan). Only the
    problems are kept: a preview never yields records (they are unhashed and must never become a dataset).
    """
    folder = folder.resolve()
    labels = folder / LABELS_FILE
    try:
        lst, fst = labels.stat(), folder.stat()
        key: tuple[object, ...] = (str(folder), lst.st_mtime_ns, lst.st_size, fst.st_mtime_ns, max_image_side_px)
    except OSError:
        key = (str(folder), None, None, None, max_image_side_px)
    with _PREVIEW_LOCK:
        cached = _PREVIEW_CACHE.get(key)
        if cached is not None:
            _PREVIEW_CACHE.move_to_end(key)
            return None if cached is _OVER_BUDGET else cached
    result = _preview_scan(folder, labels, max_image_side_px, clock() + budget_s, clock)
    with _PREVIEW_LOCK:
        _PREVIEW_CACHE[key] = _OVER_BUDGET if result is None else result
        while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_SIZE:
            _PREVIEW_CACHE.popitem(last=False)
    return result


def _preview_scan(
    folder: Path, labels: Path, max_side: int, deadline: float, clock: Callable[[], float]
) -> ScanResult | None:
    result = ScanResult()
    if problem := _labels_problem(labels):
        result.add(0, problem)
        return result
    seen: set[str] = set()
    with labels.open("rb") as fh:
        for n, raw in enumerate(fh, start=1):
            if clock() > deadline:
                return None
            if not raw.strip():
                continue
            checked = _check_line(folder, raw, seen, max_side, hashed=False)
            if isinstance(checked, Record):
                seen.add(checked.image)
                if len(result.questions) < 2:
                    result.questions.add(checked.prompt or "")
            else:
                result.add(n, *checked)
    if not seen and not result.problem_count:
        result.add(0, "no_samples")
    return result


__all__ = ["LABELS_FILE", "PREVIEW_BUDGET_S", "ScanResult", "preview_folder", "scan_folder"]
