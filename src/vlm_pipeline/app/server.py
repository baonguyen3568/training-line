"""``vlm-app [--port 7870] [--dev] [--worker fake|container]``: start the local app server (127.0.0.1 only)."""

from __future__ import annotations

import argparse
import sys


def cmd_app(args: argparse.Namespace) -> int:
    import uvicorn

    from vlm_pipeline.app.main import create_app
    from vlm_pipeline.app.services import DefaultServices
    from vlm_pipeline.common.paths import project_root
    from vlm_pipeline.nodes import REGISTRY

    services = DefaultServices(registry=REGISTRY, root=project_root(), worker=args.worker)
    app = create_app(services, port=args.port, dev=args.dev, secure=True)
    token = getattr(app.state, "startup_token", None)
    url = f"http://127.0.0.1:{args.port}/" + (f"?token={token}" if token else "")
    print(f"vlm app: open {url}", file=sys.stderr, flush=True)  # shown once, never logged (D13)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info", access_log=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vlm-app")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--dev", action="store_true", help="allow the Vite dev origin :7875, enable /api/docs")
    parser.add_argument("--worker", choices=["fake", "container"], default="fake")
    return cmd_app(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
