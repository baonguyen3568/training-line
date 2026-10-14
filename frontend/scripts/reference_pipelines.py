"""configs/reference_pipelines.yaml (owner: bao-research) -> src/generated/reference-pipelines.json (committed).

The D4 gallery bundles it like templates.json; `npm run schemas:check` fails when the copy drifts. The file is checked
by the strict schema (common/reference_pipelines.py, v2 by method: one pipeline per method, train_arch, compare_with on
the same benchmark/metric…) before it is copied, so a typo fails here instead of rendering an empty card. JSON only,
the YAML's own keys and order kept (no filled-in defaults).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from vlm_pipeline.common.reference_pipelines import REFERENCE_PIPELINES, load_reference_pipelines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main() -> None:
    path = ROOT / REFERENCE_PIPELINES
    try:
        load_reference_pipelines(path)
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise SystemExit(f"reference_pipelines: {exc}") from None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = HERE.parent / "src/generated/reference-pipelines.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"reference_pipelines: {len(data['pipelines'])} → {out}")


if __name__ == "__main__":
    main()
