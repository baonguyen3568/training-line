"""LAN access info for the owner's Settings page (ADR-019, A5): the LAN links and the local CA to install.

``GET /api/lan`` and ``GET /api/lan/ca.crt`` answer a LOCAL session only (Host 127.0.0.1; a LAN session gets 403
``local_only``): the owner, sitting at this machine, connects a phone or laptop. Only the CA's public certificate is
ever read: install.sh copies it next to the rendered nginx.conf (``~/.config/ntc-training/lan/``, mounted read-only
into the app); ``ca.key`` and the server key are never in that folder.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

CA_FILE_ENV = "VLM_LAN_CA_FILE"  # compose: /run/vlm-lan/ca.crt (read-only mount); dev: the install's own copy
CA_DOWNLOAD = "/api/lan/ca.crt"
CA_MAX_BYTES = 16 * 1024
_PEM = re.compile(rb"-----BEGIN CERTIFICATE-----\r?\n([A-Za-z0-9+/=\r\n]+?)-----END CERTIFICATE-----\r?\n?")


def ca_path() -> Path:
    raw = os.environ.get(CA_FILE_ENV, "").strip()
    return Path(raw) if raw else Path.home() / ".config" / "ntc-training" / "lan" / "ca.crt"


def read_ca(path: Path) -> tuple[bytes, str] | None:
    """(PEM bytes, SHA-256 fingerprint of the DER, ``AB:CD:...``) of a single-certificate PEM file, else None."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode) or path.stat().st_size > CA_MAX_BYTES:
            return None
        pem = path.read_bytes()
    except OSError:
        return None
    match = _PEM.fullmatch(pem.strip() + b"\n")  # exactly one certificate and nothing else (never a key)
    if match is None:
        return None
    try:
        der = base64.b64decode(b"".join(match.group(1).split()), validate=True)
    except ValueError:
        return None
    digest = hashlib.sha256(der).hexdigest().upper()
    return pem, ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))


def _local_only(request: Request) -> JSONResponse | None:
    if getattr(request.state, "vlm_origin_class", None) != "local":  # no security state (tests): refuse as well
        return JSONResponse(status_code=403, content={"code": "local_only"})
    return None


def register_lan_routes(app: FastAPI) -> None:
    def origins() -> tuple[str, ...]:
        guard = getattr(app.state, "vlm_security", None)
        return tuple(guard.config.public_origins) if guard is not None else ()

    @app.get("/api/lan", response_model=None)
    def lan(request: Request) -> dict[str, Any] | JSONResponse:
        if (refused := _local_only(request)) is not None:
            return refused
        urls = list(origins())
        ca = read_ca(ca_path()) if urls else None
        return {
            "enabled": bool(urls),
            "urls": [f"{u}/" for u in urls],
            "ca": {"fingerprint_sha256": ca[1], "download": CA_DOWNLOAD} if ca else None,
        }

    @app.get(CA_DOWNLOAD, response_model=None)
    def ca_certificate(request: Request) -> Response:
        if (refused := _local_only(request)) is not None:
            return refused
        ca = read_ca(ca_path()) if origins() else None
        if ca is None:
            return JSONResponse(status_code=404, content={"code": "lan_off"})
        headers = {"Content-Disposition": 'attachment; filename="ntc-training-ca.crt"'}
        return Response(ca[0], media_type="application/x-x509-ca-cert", headers=headers)


__all__ = ["CA_DOWNLOAD", "CA_FILE_ENV", "read_ca", "register_lan_routes"]
