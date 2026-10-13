"""Images for the trial node ("Chạy thử", B-7): upload, fetch, and the node's result (bao-security + frontend contract).

``POST /api/uploads``: RAW body (no multipart parser). Content-Type is only a hint from an allowlist; the real format
is decided by Pillow with a format allowlist at open (``common.image_rules.open_image``: the same rules everywhere).
The image is decoded, EXIF-oriented, converted to RGB and RE-ENCODED as PNG (metadata dropped) into
``<root>/uploads/<uuid4 hex>.png`` (dir 0700, file 0600, temp files O_EXCL + O_NOFOLLOW, then ``os.replace``).
The client's file name is only a display label (cleaned, never a path, never logged). Decoding runs in a thread pool,
at most ``MAX_CONCURRENT_DECODES`` at once (40 MP x 4 B ~ 160 MB each on shared unified memory).
"""

from __future__ import annotations

import asyncio
import os
import re
import stat
import unicodedata
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.image_rules import MAX_IMAGE_BYTES, ImageRuleError, open_image
from vlm_pipeline.common.runinfo import read_small_json

UPLOADS_DIR = "uploads"
CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
LABEL_MAX = 120
MAX_CONCURRENT_DECODES = 2
MAX_UPLOAD_FILES = 500  # quota (bao-security): oldest are not evicted automatically, the user deletes
MAX_UPLOAD_TOTAL_BYTES = 5 * 1024**3
_BIDI = {chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))}
_DECODES = asyncio.Semaphore(MAX_CONCURRENT_DECODES)


class UploadRejectedError(VlmPipelineError):
    """The uploaded image cannot be used (reason: too_large, too_many_pixels, unreadable, unsupported_type, quota)."""

    code: ClassVar[str] = "DATA-005"

    def __init__(self, reason: str, status: int = 400) -> None:
        super().__init__(reason, params={"reason": reason})
        self.status = status


def upload_path(root: Path, upload_id: str) -> Path:
    """Where ``upload_id`` is stored (callers validate the id first)."""
    return root / UPLOADS_DIR / f"{upload_id}.png"


def clean_label(raw: str | None) -> str:
    """Display-only name: URL-decoded, control and bidi characters removed, at most ``LABEL_MAX`` characters."""
    text = unquote(raw or "", errors="replace")
    text = "".join(ch for ch in text if ch not in _BIDI and unicodedata.category(ch) not in {"Cc", "Cf"})
    return text.strip()[:LABEL_MAX]


def _uploads_dir(root: Path) -> Path:
    d = root / UPLOADS_DIR
    if d.is_symlink():
        raise UploadRejectedError("unreadable", status=500)
    d.mkdir(mode=0o700, exist_ok=True)
    d.chmod(0o700)
    return d


def _check_quota(d: Path) -> None:
    files = [p for p in d.glob("*.png") if p.is_file()]
    if len(files) >= MAX_UPLOAD_FILES or sum(p.stat().st_size for p in files) >= MAX_UPLOAD_TOTAL_BYTES:
        raise UploadRejectedError("quota", status=507)


def _open_excl(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


async def _receive(chunks: AsyncIterator[bytes], raw_tmp: Path) -> int:
    """Stream the body to ``raw_tmp``, stopping as soon as it exceeds ``MAX_IMAGE_BYTES`` (nothing buffered)."""
    size = 0
    fd = _open_excl(raw_tmp)
    try:
        async for chunk in chunks:
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise UploadRejectedError("too_large", status=413)
            os.write(fd, chunk)
    finally:
        os.close(fd)
    return size


def _reencode(raw_tmp: Path, png_tmp: Path, final: Path) -> tuple[int, int, int]:
    try:
        img = open_image(raw_tmp)
    except ImageRuleError as exc:
        raise UploadRejectedError(exc.reason) from None
    with os.fdopen(_open_excl(png_tmp), "wb") as fh:
        img.save(fh, format="PNG")  # a fresh PNG: no EXIF, no text chunks, no ICC
    png_tmp.replace(final)
    return img.width, img.height, final.stat().st_size


async def save_upload(root: Path, chunks: AsyncIterator[bytes], *, content_type: str, label: str | None,
                      content_length: int | None) -> dict[str, Any]:  # fmt: skip
    """Store one image; returns ``{upload_id, label, width, height, size_bytes}``."""
    if content_type.split(";", 1)[0].strip().lower() not in CONTENT_TYPES:
        raise UploadRejectedError("unsupported_type", status=415)
    if content_length is not None and content_length > MAX_IMAGE_BYTES:
        raise UploadRejectedError("too_large", status=413)
    d = _uploads_dir(root)
    _check_quota(d)
    upload_id = uuid.uuid4().hex
    raw_tmp, png_tmp = d / f".{upload_id}.raw", d / f".{upload_id}.png.tmp"
    try:
        await _receive(chunks, raw_tmp)
        async with _DECODES:
            width, height, size = await run_in_threadpool(_reencode, raw_tmp, png_tmp, upload_path(root, upload_id))
    finally:
        for tmp in (raw_tmp, png_tmp):
            tmp.unlink(missing_ok=True)
    return {"upload_id": upload_id, "label": clean_label(label), "width": width, "height": height,
            "size_bytes": size}  # fmt: skip


RESULT_FIELDS = ("engine_used", "engine_requested", "text", "n_tokens", "truncated", "decoding", "overrides",
                 "model_class")  # fmt: skip
RESULT_TEXT_MAX = 20_000
RESULT_FILE_MAX = 1024 * 1024  # never load more of a container-written file into the app (shared UMA)
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


def node_result(state_dir: Path, graph_run_id: str, node_id: str) -> dict[str, Any] | None:
    """The trial node's ``result.json`` for that graph run (survives an app restart: read from the node run dirs).

    Only the documented fields; ``text`` is capped (``truncated`` set). None = no such run/node or no result yet.
    The file is untrusted (written by the container): a symlink or non-regular file is ignored, and one over
    ``RESULT_FILE_MAX`` is never read (``{"error": "result_too_large"}``).
    """
    import json

    if not (_RUN_ID.match(graph_run_id) and _RUN_ID.match(node_id)):
        return None
    runs = (state_dir / "runs").resolve()
    for run_dir in sorted(runs.glob(f"{node_id}-*")):
        try:
            manifest = read_small_json(run_dir / "run.json")  # job-written: bounded, no symlink/fifo
            tags = manifest.get("tags") if isinstance(manifest, dict) else None
            if not isinstance(tags, list) or f"graph:{graph_run_id}" not in tags:
                continue
            result = run_dir / "result.json"
            st = result.lstat()  # written by the container (remote code on a user image): untrusted input
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or not result.resolve().is_relative_to(runs):
                return None
            if st.st_size > RESULT_FILE_MAX:
                return {"error": "result_too_large"}
            data = json.loads(result.read_bytes()[:RESULT_FILE_MAX])
        except (OSError, ValueError):
            continue
        out = {k: data[k] for k in RESULT_FIELDS if k in data}
        if isinstance(out.get("text"), str) and len(out["text"]) > RESULT_TEXT_MAX:
            out["text"], out["truncated"] = out["text"][:RESULT_TEXT_MAX], True
        return out
    return None


def register_upload_routes(
    app: FastAPI, root: Path, state_dir: Path, render_error: Callable[[str, dict[str, Any], str], dict[str, Any]]
) -> None:
    """``render_error(code, params, locale)`` -> ``{code, params, title, what, ...}`` (same body as validate)."""

    def _error(status: int, code: str, params: dict[str, Any], locale: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": render_error(code, params, locale)})

    @app.post("/api/uploads")
    async def upload(request: Request, locale: str = "vi") -> JSONResponse:
        length = request.headers.get("content-length")
        try:
            body = await save_upload(
                root, request.stream(), content_type=request.headers.get("content-type", ""),
                label=request.headers.get("x-upload-label"), content_length=int(length) if length else None,
            )  # fmt: skip
        except UploadRejectedError as exc:
            return _error(exc.status, exc.code, dict(exc.params), locale)
        except ValueError:  # malformed Content-Length
            return _error(400, UploadRejectedError.code, {"reason": "unreadable"}, locale)
        return JSONResponse(status_code=201, content=body)

    @app.get("/api/uploads/{upload_id}", response_model=None)
    def get_upload(upload_id: str) -> FileResponse | JSONResponse:
        if not UPLOAD_ID.match(upload_id):  # before the id touches a path
            return JSONResponse(status_code=404, content={"error": {"code": "UPLOAD_NOT_FOUND"}})
        path = upload_path(root, upload_id)
        if path.is_symlink() or not path.is_file():
            return JSONResponse(status_code=404, content={"error": {"code": "UPLOAD_NOT_FOUND"}})
        return FileResponse(path, media_type="image/png",
                            headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})  # fmt: skip

    @app.get("/api/runs/{graph_run_id}/nodes/{node_id}/result")
    def get_result(graph_run_id: str, node_id: str) -> JSONResponse:
        result = node_result(state_dir, graph_run_id, node_id)
        if result is None:
            return JSONResponse(status_code=404, content={"error": {"code": "RESULT_NOT_READY"}})
        if result.get("error") == "result_too_large":
            return JSONResponse(status_code=502, content={"error": {"code": "RESULT_TOO_LARGE"}})
        return JSONResponse(content=result)


__all__ = [
    "CONTENT_TYPES",
    "UPLOADS_DIR",
    "UPLOAD_ID",
    "UploadRejectedError",
    "clean_label",
    "node_result",
    "register_upload_routes",
    "save_upload",
    "upload_path",
]
