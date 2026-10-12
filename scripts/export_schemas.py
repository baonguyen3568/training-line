"""Write JSON Schemas of the backend<->dashboard contract into ``schemas/`` (committed; check.sh diffs them).

Usage: ``uv run python scripts/export_schemas.py`` (``--check`` exits 1 if a file would change).
"""

from __future__ import annotations

import json
import sys

from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import MetricsPage, RunManifest, RunStatus


def render() -> dict[str, str]:
    schemas: dict[str, object] = {
        "run_status.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "RunStatus",
            "type": "string",
            "enum": [s.value for s in RunStatus],
        },
        "run_manifest.schema.json": RunManifest.model_json_schema(mode="serialization"),
        "metrics_page.schema.json": MetricsPage.model_json_schema(mode="serialization"),
    }
    from vlm_pipeline.nodes import REGISTRY

    schemas["node_types.snapshot.json"] = REGISTRY.to_api()  # locale-neutral: keys only, no resolved text
    return {name: json.dumps(body, indent=2, sort_keys=True) + "\n" for name, body in schemas.items()}


def main() -> int:
    out = project_root() / "schemas"
    out.mkdir(exist_ok=True)
    changed = []
    for name, text in render().items():
        path = out / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            changed.append(name)
            if "--check" not in sys.argv:
                path.write_text(text, encoding="utf-8")
    if changed:
        print(("out of date: " if "--check" in sys.argv else "wrote: ") + ", ".join(changed))
    return 1 if changed and "--check" in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(main())
