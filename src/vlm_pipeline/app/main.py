"""HTTP/WS API of the node-graph app (P009 §6, M1-3). Bind 127.0.0.1 only; auth/CSRF/Origin via ``app.security``.

Text for the user is rendered here (the edge) from i18n keys; domain objects only carry codes + params.
"""

from __future__ import annotations

import logging
import re
import secrets
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from vlm_pipeline import i18n
from vlm_pipeline.app.events import EventBus
from vlm_pipeline.app.services import BadRequestError, NotAvailableError, Services
from vlm_pipeline.common.error_catalog import Labels, display_params, variant_key
from vlm_pipeline.nodes.standards import MESSAGE_PATTERN, STANDARD_CODE

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


_STD_MESSAGE = re.compile(MESSAGE_PATTERN)  # only standards.* keys may replace error.STD-001.what


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
        if part == "what" and code == STANDARD_CODE and _STD_MESSAGE.match(str(raw.get("message", ""))):
            key = str(raw["message"])  # the standard's own sentence (configs/standards/*.yaml -> i18n standards.*)
        if part == "why" and key not in texts:  # "why" is optional in the catalog
            out[part] = None
            continue
        out[part] = i18n.t(key, locale, **params)
    out["title_key"] = f"error.{code}.title"
    return out


def _issue_actions(issue: Mapping[str, Any], workflow: Mapping[str, Any], registry: Any) -> dict[str, Any]:
    """G-3 quick fixes of one validate issue, from the node's params as validated (defaults filled in).

    A group's inner step or a node whose params do not validate gets no patch (never a guessed value).
    """
    from vlm_pipeline.graph.actions import issue_actions

    node_id = issue.get("node_id")
    raw = next((n for n in workflow.get("nodes", []) if isinstance(n, Mapping) and n.get("id") == node_id), None)
    spec = registry.get(raw.get("type")) if raw is not None and not issue.get("inner_path") else None
    params = None
    if spec is not None and raw is not None:
        try:
            params = spec.validate_params(raw.get("params") or {}).model_dump(mode="json")
        except ValueError:  # the issue may be about these very params (CFG-005): no context, no patch
            spec = None
    error = SimpleNamespace(code=issue.get("code", "SYS-000"), params=issue.get("params") or {})
    return issue_actions(error, node_id=node_id if isinstance(node_id, str) else None, node_params=params,
                         params_model=spec.params_model if spec is not None else None)  # fmt: skip


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
    bus.open(graph_run_id, context=workflow)  # error cards render step names from it (WS ?locale=)

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


def _train_image(services: Services) -> str | None:
    """The image real train jobs run in; None for the demo (fake) worker: its runs say nothing about real speed."""
    worker = getattr(getattr(services, "executor", None), "worker", None)
    image = getattr(worker, "image", None)
    return image if isinstance(image, str) and getattr(services, "worker", "container") != "fake" else None


def _workflow_estimate(services: Services, workflow: dict[str, Any]) -> dict[str, Any]:
    """RAM/disk from the catalog (as the gallery), duration from measured runs; unknown stays null, never a guess."""
    from vlm_pipeline.app.estimates import workflow_duration
    from vlm_pipeline.app.templates import catalog_estimate

    base = catalog_estimate(workflow)
    root, image = getattr(services, "root", None), _train_image(services)
    duration = workflow_duration(root, workflow, image) if root is not None and image is not None else None
    return {
        "ram_gb_peak": base["ram_gb"],
        "disk_gb": base["disk_gb"],
        "duration_min": duration["duration_min"] if duration else None,
        "gpu_exclusive": base["gpu_exclusive"],
        "source": base["source"],
        "duration": duration,  # per train node {steps, minutes, based_on} + heavy nodes without a measurement
    }


def _register_graph_routes(app: FastAPI, services: Services, bus: EventBus) -> None:
    @app.get("/api/node-types")
    @app.get("/api/nodes")
    def node_types(locale: Locale = i18n.DEFAULT_LOCALE) -> dict[str, Any]:
        return _node_types(services, locale)

    @app.post("/api/graphs/validate")
    def validate(workflow: Workflow, locale: Locale = i18n.DEFAULT_LOCALE) -> dict[str, Any]:
        labels = _labels(workflow, services.registry, locale)
        errors = [
            {**_render_issue(i, locale, labels), **_issue_actions(i, workflow, services.registry)}
            for i in services.validate(workflow)
        ]
        return {"errors": errors, "estimate": _workflow_estimate(services, workflow)}

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
        return [template_meta(t) for t in load_templates(root, image=_train_image(services))]

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


def _register_info_routes(app: FastAPI, services: Services, bus: EventBus) -> None:
    """/api/health (the code this process runs) and the public bug-report card (no log, no path: security)."""
    from vlm_pipeline.app import build_info
    from vlm_pipeline.common.paths import project_root

    root = getattr(services, "root", None) or project_root()
    info = build_info.read_build_info(root)  # once: what is RUNNING
    _log.info("vlm-app %s (commit %s, dirty %s)", info.app_version, info.commit, info.dirty)
    app.state.build_info = info

    def _locale(value: str | None) -> str:
        return value if value in i18n.LOCALES else i18n.DEFAULT_LOCALE

    def _report(code: str, locale: str | None, run_id: str | None = None) -> dict[str, str]:
        try:
            key = build_info.ref_key(root) if run_id is not None else None
            return build_info.bug_report(code, _locale(locale), info, run_id=run_id, key=key)
        except KeyError:  # only catalog codes: never echo a string the client made up
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}}) from None

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **info.to_api()}

    @app.get("/api/bug-report")
    def bug_report(code: str, locale: str | None = None) -> dict[str, str]:
        return _report(code, locale)

    @app.get("/api/runs/{run_id}/bug-report")
    def run_bug_report(run_id: str, code: str, locale: str | None = None) -> dict[str, str]:
        if services.get_run(run_id) is None and not bus.known(run_id):
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return _report(code, locale, run_id)


def _register_advanced_routes(app: FastAPI, services: Services) -> None:
    """P016 §3: the engine options a train node's ``advanced`` param accepts (fetched when the panel opens)."""

    @app.get("/api/node-types/{type_id}/advanced")
    def advanced_options(type_id: str, model: str | None = None) -> dict[str, Any]:
        from vlm_pipeline.train.advanced import engine_options

        spec = services.registry.get(type_id)
        if spec is None or "advanced" not in spec.params_model.model_fields:  # unknown, or no Advanced panel
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        options = [
            {
                "key": key,
                "group": key.rsplit(".", 1)[0],  # P016 §3 prefix: model.backend.attn -> model.backend
                **{f: entry.get(f) for f in ("type", "default", "choices", "min", "max")},
                "templates": list(entry.get("templates", ())),
            }
            for key, entry in sorted(engine_options().items())
            if model is None or model in entry.get("templates", ())
        ]
        return {"options": options}


def _rendered_event(event: dict[str, Any], locale: str, labels: Labels | None) -> dict[str, Any]:
    """``event`` with its ``error`` (if any) carrying title / what / why in ``locale``, as validate issues do."""
    error = event.get("error")
    if not isinstance(error, dict) or "code" not in error:
        return event
    issue = {"code": error["code"], "params": error.get("params") or {}, "node_id": event.get("node_id")}
    text = _render_issue(issue, locale, labels)
    return {**event, "error": {**error, **{k: text.get(k) for k in ("title", "what", "why", "title_key")}}}


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["model", "dataset"]
    catalog_id: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9][a-z0-9.\-]*$")
    confirm: bool = False
    check: bool = False  # verify a snapshot already in the cache against the hub (plan only, nothing re-downloaded)


def _register_download_routes(
    app: FastAPI, services: Services, bus: EventBus, ws_auth: Callable[[WebSocket], Awaitable[None]] | None
) -> None:
    """G-5 (P018): start a catalog download, list them; progress over WS ``/api/downloads/{id}/events``."""
    root = getattr(services, "root", None)
    if root is None:
        return
    from vlm_pipeline.app.downloads import DownloadService

    def emit(download_id: str, event: dict[str, Any]) -> None:
        bus.publish(download_id, {"download_id": download_id, "ts": time.time(), **event})
        if event.get("type") == "status" and event.get("status") in ("completed", "failed"):
            bus.close(download_id)

    def token() -> str | None:
        store = getattr(app.state, "secret_store", None)
        return store.reveal("HF_TOKEN") if store is not None else None

    service = DownloadService(root, emit=emit, token=token)
    app.state.downloads = service

    @app.post("/api/downloads", status_code=202)
    def start_download(body: DownloadRequest) -> dict[str, Any]:
        from vlm_pipeline.common.errors import VlmPipelineError

        try:
            result = service.start(body.kind, body.catalog_id, confirm=body.confirm, check=body.check)
        except VlmPipelineError as exc:  # quota, disk, unknown, busy: the card's code and params, never a trace
            status = 404 if exc.code == "CFG-004" else 409
            raise HTTPException(status_code=status, detail={"code": exc.code, "params": exc.params}) from None
        if result.get("download_id"):
            bus.open(result["download_id"])
        return result

    @app.get("/api/downloads")
    def list_downloads() -> list[dict[str, Any]]:
        return [d.to_api() for d in service.downloads.values()]

    @app.websocket("/api/downloads/{download_id}/events")
    async def download_events(websocket: WebSocket, download_id: str) -> None:
        if ws_auth is not None:
            try:
                await ws_auth(websocket)  # same auth as the run events (D13)
            except PermissionError:
                return
        else:
            await websocket.accept()
        if not bus.known(download_id):
            await websocket.send_json({"type": "download_not_found", "download_id": download_id})
            await websocket.close(code=4404)
            return
        try:
            async for event in bus.follow(download_id):
                await websocket.send_json(event)
            await websocket.close(code=1000)
        except WebSocketDisconnect:
            return


def _register_ws(
    app: FastAPI,
    bus: EventBus,
    ws_auth: Callable[[WebSocket], Awaitable[None]] | None,
    registry: Any = None,
) -> None:
    @app.websocket("/api/runs/{graph_run_id}/events")
    async def events(websocket: WebSocket, graph_run_id: str, locale: str | None = None) -> None:
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
        loc = locale if locale in i18n.LOCALES else None  # no locale: raw events, as before
        workflow = bus.context(graph_run_id)
        labels = _labels(workflow, registry, loc) if loc and isinstance(workflow, Mapping) and registry else None
        try:
            async for event in bus.follow(graph_run_id):
                await websocket.send_json(_rendered_event(event, loc, labels) if loc else event)
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


def _register_settings(app: FastAPI, services: Services, key_file: Path | None) -> None:
    """Settings page (D-7): write-only secrets in outputs/system/settings.db, admin policy read-only."""
    root = getattr(services, "root", None)
    if root is None:
        return  # test doubles without a project root
    from vlm_pipeline.app.secret_store import SecretStore
    from vlm_pipeline.app.settings_api import register_settings_routes

    app.state.secret_store = store = SecretStore(root, key_file=key_file)
    register_settings_routes(app, store)


def _reconcile_stale_runs(services: Services) -> None:
    """Runs left "running" by a process that is gone become interrupted (PM): once at start-up, in the background.

    The storage scheduler repeats it every pass. Demo (fake worker) runs are not real jobs: skipped.
    """
    root = getattr(services, "root", None)
    if root is None or getattr(services, "worker", "fake") == "fake":
        return
    from vlm_pipeline.artifacts.stale_runs import reconcile

    def run() -> None:
        try:
            reconcile(root)
        except Exception:  # noqa: BLE001 - a start-up chore must never take the app down (logged)
            _log.exception("stale-run check failed")

    threading.Thread(target=run, name="stale-runs", daemon=True).start()


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
    secret_key_file: Path | None = None,
) -> FastAPI:
    """Build the app (D13 security on by default). ``secure=False`` only for unit tests of the API itself.

    ``secret_key_file``: tests only; the app uses ``~/.config/ntc-training/secrets.key`` (no env override, D-7).
    """
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
    _register_settings(app, services, secret_key_file)
    _reconcile_stale_runs(services)
    _register_graph_routes(app, services, bus)
    _register_run_routes(app, services)
    _register_info_routes(app, services, bus)
    _register_advanced_routes(app, services)
    _register_download_routes(app, services, bus, ws_auth)
    _register_template_routes(app, services)
    _register_ws(app, bus, ws_auth, services.registry)
    if static_dir is None and getattr(services, "root", None) is not None:
        static_dir = services.root / "frontend" / "dist"  # type: ignore[attr-defined]
    if static_dir is not None and (static_dir / "index.html").is_file():
        _register_spa(app, static_dir)  # last: API routes win
    return app
