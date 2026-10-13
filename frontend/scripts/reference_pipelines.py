"""configs/reference_pipelines.yaml (owner: bao-research) -> src/generated/reference-pipelines.json (committed).

The D4 gallery bundles it like templates.json; `npm run schemas:check` fails when the copy drifts. JSON only,
keys and order kept; a missing field the UI needs fails here instead of rendering empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REQUIRED = ("id", "kind", "template_id", "status", "difficulty", "title", "source_url", "license", "steps")


def main() -> None:
    data = yaml.safe_load((ROOT / "configs/reference_pipelines.yaml").read_text(encoding="utf-8"))
    for p in data["pipelines"]:
        missing = [k for k in REQUIRED if k not in p]
        if missing:
            raise SystemExit(f"reference_pipelines: {p.get('id')} lacks {missing}")
    out = HERE.parent / "src/generated/reference-pipelines.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"reference_pipelines: {len(data['pipelines'])} → {out}")


if __name__ == "__main__":
    main()
