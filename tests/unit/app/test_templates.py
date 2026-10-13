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
    templates = load_templates(ROOT)  # what the gallery offers: every one must be valid
    assert {t["id"] for t in load_templates(ROOT, include_hidden=True)} >= {
        "invoice_to_json", "image_video_understanding", "quantize_bench",
    }  # fmt: skip
    assert templates[0]["id"] == "invoice_to_json"  # default template first (order)
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


def test_hidden_templates_are_never_served(tmp_path: Path) -> None:
    import json as _json

    d = tmp_path / "configs" / "templates"
    d.mkdir(parents=True)
    base = {"order": 1, "title_key": "t", "description_key": "d", "difficulty": "easy", "tags": [],
            "sample_data": False, "workflow": {}}  # fmt: skip
    (d / "a.json").write_text(_json.dumps({**base, "id": "shown"}))
    (d / "b.json").write_text(_json.dumps({**base, "id": "kept", "hidden": True}))
    assert [t["id"] for t in load_templates(tmp_path)] == ["shown"]
    assert {t["id"] for t in load_templates(tmp_path, include_hidden=True)} == {"shown", "kept"}
