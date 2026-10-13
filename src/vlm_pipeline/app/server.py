"""``vlm-app [--port 7870] [--dev] [--worker fake|container]``: start the local app server (127.0.0.1 only)."""

from __future__ import annotations

import argparse
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

    services = DefaultServices(registry=REGISTRY, root=project_root(), worker=args.worker)
    port = public_port(args.port)  # what the browser uses: the Host allowlist and the printed link (D13)
    app = create_app(services, port=port, dev=args.dev, secure=True)
    _start_retention_scheduler(app, services)
    token = getattr(app.state, "startup_token", None)
    url = f"http://127.0.0.1:{port}/" + (f"?token={token}" if token else "")
    print(f"vlm app: open {url}", file=sys.stderr, flush=True)  # shown once, never logged (D13)
    uvicorn.run(app, host=bind_host(), port=args.port, log_level="info", access_log=False)
    return 0


PUBLIC_PORT_ENV = "VLM_PUBLIC_PORT"


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
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--dev", action="store_true", help="allow the Vite dev origin :7875, enable /api/docs")
    parser.add_argument("--worker", choices=["fake", "container"], default="fake")
    return cmd_app(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
