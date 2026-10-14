"""``vlm-app [--port 4003] [--dev] [--worker fake|container]``: start the app server (127.0.0.1; LAN only via nginx)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from vlm_pipeline.app.services import DefaultServices


def cmd_app(args: argparse.Namespace) -> int:
    import uvicorn

    from vlm_pipeline.app.main import create_app
    from vlm_pipeline.app.services import DefaultServices
    from vlm_pipeline.common.paths import project_root
    from vlm_pipeline.nodes import REGISTRY

    lan = lan_settings()
    if isinstance(lan, str):
        print(f"vlm app: {lan}", file=sys.stderr, flush=True)
        return 2
    origins, proxy_peer = lan
    services = DefaultServices(registry=REGISTRY, root=project_root(), worker=args.worker)
    port = public_port(args.port)  # what the browser uses: the Host allowlist and the printed link (D13)
    app = create_app(services, port=port, dev=args.dev, secure=True, public_origins=origins, proxy_peer=proxy_peer)
    _start_retention_scheduler(app, services)
    guard = getattr(app.state, "vlm_security", None)
    if guard is not None:
        guard.announce(guard.token)  # the sign-in links (#t=, never sent to a server); shown, never logged (D13)
    _log_to_stderr()
    uvicorn.run(app, host=bind_host(), port=args.port, log_level="info", access_log=False)
    return 0


def _log_to_stderr() -> None:
    """The app's own loggers to stderr (docker logs).

    Without a handler only the root's lastResort existed, and an executor error never reached any log (slot #2,
    2026-10-14). Messages never carry secrets (tokens are not logged).
    """
    import logging

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_log = logging.getLogger("vlm_pipeline")
    app_log.addHandler(handler)
    app_log.setLevel(logging.INFO)
    app_log.propagate = False


PUBLIC_PORT_ENV = "VLM_PUBLIC_PORT"


def lan_settings() -> tuple[tuple[str, ...], str | None] | str:
    """ADR-019: ``(VLM_PUBLIC_ORIGINS, VLM_PROXY_PEER)`` validated, or the reason the app must not start.

    A bad value stops the app, never widens access. In compose, LAN origins without nginx's fixed address would
    silently turn the peer check off (security): refused.
    """
    from vlm_pipeline.app.security import PROXY_PEER_ENV, PUBLIC_ORIGINS_ENV, parse_proxy_peer, parse_public_origins

    try:
        origins = parse_public_origins(os.environ.get(PUBLIC_ORIGINS_ENV, ""))
        proxy_peer = parse_proxy_peer(os.environ.get(PROXY_PEER_ENV, ""))
    except ValueError as exc:
        return str(exc)
    if origins and proxy_peer is None and os.environ.get("VLM_IN_COMPOSE") == "1":
        return f"{PUBLIC_ORIGINS_ENV} needs {PROXY_PEER_ENV} (nginx's fixed address) in compose"
    return origins, proxy_peer


def public_port(listen_port: int) -> int:
    """The port the browser reaches: ``listen_port``, or in compose the published ``VLM_PUBLIC_PORT``.

    In the compose stack the container always listens on 7870 and compose publishes ``$VLM_PORT`` on the host. Only
    a plain 1024..65535 number is taken; anything else keeps the listen port (the Host allowlist refuses, never widens).
    """
    import os

    raw = os.environ.get(PUBLIC_PORT_ENV, "")
    if (
        os.environ.get("VLM_IN_COMPOSE") != "1"
        or not (raw.isascii() and raw.isdigit())
        or not 1024 <= int(raw) <= 65535
    ):
        return listen_port
    return int(raw)


def bind_host() -> str:
    """127.0.0.1, except inside the compose stack (``VLM_IN_COMPOSE=1``).

    There the container must listen on all of ITS interfaces so the port mapping reaches it; compose publishes it on
    the host's 127.0.0.1 only (ADR-015).
    """
    import os

    return "0.0.0.0" if os.environ.get("VLM_IN_COMPOSE") == "1" else "127.0.0.1"  # noqa: S104 - container-only


def _start_retention_scheduler(app: FastAPI, services: DefaultServices) -> None:
    """Scheduled cleanup pass (P013 G-2), started only by the real server, never by create_app (unit tests).

    Same settings and index as the storage panel; ``VLM_RETENTION_SCHEDULE`` / ``VLM_PROFILE`` pick the mode.
    """
    from vlm_pipeline.app.storage_scheduler import RetentionScheduler, mode_from_env
    from vlm_pipeline.common.gpu_lock import default_lock_path

    settings = getattr(app.state, "storage_settings", None)
    if settings is None:
        return  # storage panel not registered (no project root)
    mode = mode_from_env()
    RetentionScheduler(services.root, services.index, settings, lock_path=default_lock_path(), mode=mode).start()
    print(f"vlm app: scheduled cleanup pass: {mode}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vlm-app")
    parser.add_argument("--port", type=int, default=4003)  # D-8: 4003 dev app, 4001 the compose app's host port
    parser.add_argument("--dev", action="store_true", help="allow the Vite dev origin :4010, enable /api/docs")
    parser.add_argument("--worker", choices=["fake", "container"], default="fake")
    return cmd_app(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
