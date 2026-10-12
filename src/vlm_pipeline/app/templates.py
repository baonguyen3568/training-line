"""Template gallery (P010): ``configs/templates/<id>.json`` = gallery metadata + a ready-to-open workflow.

Estimates are derived from the model catalog (``ram_gb_train`` + ``ram_source``) of the template's base model;
nothing is invented (unknown -> nulls, source "unknown").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vlm_pipeline.common.catalog import load_catalog

TEMPLATE_DIR = Path("configs") / "templates"


class TemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title_key: str
    description_key: str
    difficulty: str
    tags: list[str]
    sample_data: bool
    requires_hf_account: bool = False
    workflow: dict[str, Any]


def _estimate(workflow: dict[str, Any]) -> dict[str, Any]:
    catalog = load_catalog()
    for node in workflow.get("nodes", []):
        if node.get("type") == "model.hf_pick":
            entry = catalog.get(str(node.get("params", {}).get("model")))
            if entry is not None:
                ram = entry.ram_gb_train
                return {
                    "ram_gb": {"lo": ram, "hi": ram},
                    "disk_gb": {"lo": entry.disk_gb, "hi": entry.disk_gb},
                    "minutes": None,
                    "gpu_exclusive": True,
                    "source": "measured" if entry.ram_source == "measured" else "formula",
                }
    return {"ram_gb": None, "disk_gb": None, "minutes": None, "gpu_exclusive": True, "source": "unknown"}


def load_templates(root: Path) -> list[dict[str, Any]]:
    """All templates (full, with workflow), sorted by id. Invalid files raise (caught by tests, not at runtime)."""
    out = []
    for path in sorted((root / TEMPLATE_DIR).glob("*.json")):
        tpl = TemplateFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        out.append({**tpl.model_dump(), "estimate": _estimate(tpl.workflow)})
    return out


def template_meta(template: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in template.items() if k != "workflow"}
