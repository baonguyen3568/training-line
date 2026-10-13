"""HTTP/WS API of the node-graph app (P009 §6, M1-3). Bind 127.0.0.1 only; auth/CSRF/Origin via ``app.security``.

Text for the user is rendered here (the edge) from i18n keys; domain objects only carry codes + params.
"""

from __future__ import annotations

import logging
import secrets
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from vlm_pipeline import i18n
from vlm_pipeline.app.events import EventBus
from vlm_pipeline.app.services import BadRequestError, NotAvailableError, Services
from vlm_pipeline.common.error_catalog import Labels, display_params, variant_key

Locale = Annotated[str, Query(pattern="^(vi|en)$")]
Workflow = Annotated[dict[str, Any], Body()]
_log = logging.getLogger(__name__)
MAX_LIMIT = 10_000  # = artifacts.runs_index.MAX_LIMIT


def _labels(workflow: Mapping[str, Any], registry: Any, locale: str) -> Labels:
    """Resolve node/port/param/port-type ids of the workflow being shown to localized labels (never crashes)."""
    raw_nodes = workflow.get("nodes")
    nodes: list[Any] = raw_nodes if isinstance(raw_nodes, list) else []
    type_of = {str(n.get("id")): str(n.get("type")) for n in nodes if isinstance(n, Mapping)}
    port_keys = {pt.id: pt.label_key for pt in registry.port_types}

    def tr(key: str) -> str | None:
        text = i18n.t(key, locale)
        return None if text == key else text

    return Labels(
        node=lambda nid: tr(f"node.{type_of[nid]}.title") if nid in type_of else None,
        port=lambda nid, p: tr(f"node.{type_of[nid]}.port.{p}") if nid in type_of else None,
        param=lambda nid, p: tr(f"node.{type_of[nid]}.param.{p}.label") if nid in type_of else None,
        port_type=lambda pt: tr(port_keys[pt]) if pt in port_keys else None,
    )


def _render_issue(issue: Mapping[str, Any], locale: str, labels: Labels | None = None) -> dict[str, Any]:
    code = str(issue.get("code", "SYS-000"))
    raw = dict(issue.get("params") or {})
    node_id = issue.get("node_id")
    params = display_params(code, raw, labels, node_id=node_id if isinstance(node_id, str) else None) if labels else raw
    out = dict(issue)  # node_id / port / param / params stay raw for the UI
    texts = i18n.load(locale)
    variant = variant_key(code, raw)  # chosen on raw ids, never on display labels
    for part in ("title", "what", "why"):
        key = f"error.{code}.{part}"
        if part == "what" and variant:
            key = f"{key}.{variant}"
        if part == "why" and key not in texts:  # "why" is optional in the catalog
            out[part] = None
            continue
        out[part] = i18n.t(key, locale, **params)
    out["title_key"] = f"error.{code}.title"
    return out


def _node_types(services: Services, locale: str) -> dict[str, Any]:
    body = services.registry.to_api()
    for node in body["node_types"]:
        tid = node["type_id"]
        node["title_key"] = f"node.{tid}.title"
        node["desc_key"] = f"node.{tid}.desc"
        node["title"] = i18n.t(node["title_key"], locale)
        node["desc"] = i18n.t(node["desc_key"], locale)
    for port in body["port_types"]:
        port["label"] = i18n.t(port["label_key"], locale)
    return body


def _start_graph_run(services: Services, bus: EventBus, workflow: dict[str, Any]) -> str:
    graph_run_id = f"g-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    bus.open(graph_run_id)

    def emit(event: dict[str, Any]) -> None:
        bus.publish(graph_run_id, {"graph_run_id": graph_run_id, "ts": time.time(), **event})

    def worker() -> None:
        try:
            services.run_graph(workflow, graph_run_id, emit)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a run_status event
            _log.exception("graph run %s failed", graph_run_id)
            code = getattr(exc, "code", "SYS-000")
            emit(
                {
                    "type": "run_status",
                    "status": "failed",
                    "error": {"code": code, "params": getattr(exc, "params", {})},
                }
            )
        finally:
            bus.close(graph_run_id)

    threading.Thread(target=worker, name=f"graph-{graph_run_id}", daemon=True).start()
    return graph_run_id


def _register_graph_routes(app: FastAPI, services: Services, bus: EventBus) -> None:
    @app.get("/api/node-types")
    @app.get("/api/nodes")
    def node_types(locale: Locale = i18n.DEFAULT_LOCALE) -> dict[str, Any]:
        return _node_types(services, locale)

    @app.post("/api/graphs/validate")
    def validate(workflow: Workflow, locale: Locale = i18n.DEFAULT_LOCALE) -> dict[str, Any]:
        labels = _labels(workflow, services.registry, locale)
        return {"errors": [_render_issue(i, locale, labels) for i in services.validate(workflow)]}

    @app.post("/api/runs", status_code=202)
    def start_run(workflow: Workflow) -> dict[str, Any]:
        blocking = [i for i in services.validate(workflow) if i.get("severity", "error") == "error"]
        if blocking:
            raise HTTPException(status_code=422, detail={"errors": blocking})
        return {"graph_run_id": _start_graph_run(services, bus, workflow)}

    @app.post("/api/runs/{graph_run_id}/cancel", status_code=202)
    def cancel(graph_run_id: str) -> dict[str, Any]:
        if not services.cancel(graph_run_id):
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return {"graph_run_id": graph_run_id}


def _register_template_routes(app: FastAPI, services: Services) -> None:
    root = getattr(services, "root", None)
    if root is None:
        return
    from vlm_pipeline.app.templates import load_templates, template_meta

    @app.get("/api/templates")
    def list_templates() -> list[dict[str, Any]]:
        return [template_meta(t) for t in load_templates(root)]

    @app.get("/api/templates/{template_id}")
    def get_template(template_id: str) -> dict[str, Any]:
        found = next((t for t in load_templates(root) if t["id"] == template_id), None)
        if found is None:
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return found


def _register_run_routes(app: FastAPI, services: Services) -> None:
    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return services.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = services.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return run

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(
        run_id: str,
        after: str | None = None,
        limit: Annotated[int, Query(ge=1)] = 5000,
        downsample_to: Annotated[int | None, Query(ge=2)] = None,
    ) -> dict[str, Any]:
        return services.metrics_page(run_id, after, min(limit, MAX_LIMIT), downsample_to)

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return services.system()


def _register_ws(app: FastAPI, bus: EventBus, ws_auth: Callable[[WebSocket], Awaitable[None]] | None) -> None:
    @app.websocket("/api/runs/{graph_run_id}/events")
    async def events(websocket: WebSocket, graph_run_id: str) -> None:
        if ws_auth is not None:
            try:
                await ws_auth(websocket)  # accepts, then requires the auth message; closes 4401 on failure
            except PermissionError:
                return
        else:
            await websocket.accept()
        if not bus.known(graph_run_id):
            await websocket.send_json({"type": "run_not_found", "graph_run_id": graph_run_id})
            await websocket.close(code=4404)
            return
        try:
            async for event in bus.follow(graph_run_id):
                await websocket.send_json(event)
            await websocket.close(code=1000)
        except WebSocketDisconnect:
            return


def _register_storage(app: FastAPI, services: Services) -> None:
    """Storage panel (bao-cleanup's app/storage.py) on the same index instance the executor uses."""
    root = getattr(services, "root", None)
    if root is None or not hasattr(services, "index"):
        return  # test doubles without a project root
    from vlm_pipeline.app import storage

    app.state.storage_settings = settings = storage.StorageSettings.from_env(root)  # shared with the scheduler
    api = storage.StorageApi(root, services.index, settings)
    storage.register_storage_routes(app, api)


def _register_uploads(app: FastAPI, services: Services) -> None:
    """Trial-node images + result (app/uploads.py); errors render like validate issues."""
    root, state_dir = getattr(services, "root", None), getattr(services, "_state_dir", None)
    if root is None or state_dir is None:
        return  # test doubles without a project root
    from vlm_pipeline.app.uploads import register_upload_routes

    def render(code: str, params: dict[str, Any], locale: str) -> dict[str, Any]:
        return _render_issue({"code": code, "params": params}, locale if locale in ("vi", "en") else "vi")

    register_upload_routes(app, root, state_dir, render)


def _register_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built frontend on the same origin (D13: no CORS); unknown paths fall back to index.html."""
    root = static_dir.resolve()
    index = root / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-store"})


def create_app(
    services: Services | None = None,
    *,
    token: str | None = None,
    port: int = 7870,
    dev: bool = False,
    secret: bytes | None = None,
    ws_auth_timeout_s: float = 5.0,
    secure: bool = True,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the app (D13 security on by default). ``secure=False`` only for unit tests of the API itself."""
    if not secure and "pytest" not in sys.modules:
        raise RuntimeError("create_app(secure=False) is only allowed under pytest")
    if services is None:
        from vlm_pipeline.app.services import DefaultServices
        from vlm_pipeline.common.paths import project_root
        from vlm_pipeline.nodes import REGISTRY

        services = DefaultServices(registry=REGISTRY, root=project_root())
    app = FastAPI(
        title="vlm-pipeline", docs_url="/api/docs" if dev else None, redoc_url=None, openapi_url="/api/openapi.json"
    )
    bus = EventBus()
    app.state.bus = bus
    app.state.services = services
    ws_auth = None
    if secure:
        from vlm_pipeline.app import security  # bao-security (D13)

        app.state.startup_token = security.install_security(
            app, port=port, dev=dev, token=token, secret=secret, ws_auth_timeout_s=ws_auth_timeout_s
        )
        ws_auth = security.ws_authenticate

    @app.exception_handler(NotAvailableError)
    async def _not_available(_request: Any, exc: NotAvailableError) -> JSONResponse:
        return JSONResponse(status_code=501, content={"code": "SYS-000", "detail": str(exc)})

    @app.exception_handler(BadRequestError)
    async def _bad_request(_request: Any, exc: BadRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"code": exc.code, "params": exc.params})

    _register_storage(app, services)
    _register_uploads(app, services)
    _register_graph_routes(app, services, bus)
    _register_run_routes(app, services)
    _register_template_routes(app, services)
    _register_ws(app, bus, ws_auth)
    if static_dir is None and getattr(services, "root", None) is not None:
        static_dir = services.root / "frontend" / "dist"  # type: ignore[attr-defined]
    if static_dir is not None and (static_dir / "index.html").is_file():
        _register_spa(app, static_dir)  # last: API routes win
    return app
