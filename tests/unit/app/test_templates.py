from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.app.templates import load_templates
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY

ROOT = project_root()


def test_every_template_is_a_valid_workflow() -> None:
    templates = load_templates(ROOT)
    assert {t["id"] for t in templates} >= {"invoice_to_json", "image_video_understanding", "quantize_bench"}
    for t in templates:
        issues = validate(t["workflow"], REGISTRY, workspace=ROOT)
        assert [i for i in issues if i.severity == "error"] == [], t["id"]


def test_default_template_estimate_comes_from_the_catalog() -> None:
    t: dict[str, Any] = next(t for t in load_templates(ROOT) if t["id"] == "invoice_to_json")
    assert t["estimate"]["ram_gb"]["lo"] > 0
    assert t["estimate"]["source"] in {"measured", "formula"}


def test_templates_api(tmp_path: Path) -> None:
    services = DefaultServices(registry=REGISTRY, root=ROOT)
    c = TestClient(create_app(services, secure=False, static_dir=tmp_path))
    metas = c.get("/api/templates").json()
    assert all("workflow" not in m for m in metas)
    full = c.get("/api/templates/invoice_to_json").json()
    assert full["workflow"]["nodes"][0]["type"] == "model.hf_pick"
    assert c.get("/api/templates/nope").status_code == 404
