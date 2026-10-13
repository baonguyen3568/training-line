"""Copy the app's own UI text into the site build, so the landing shows the product's real wording.

Source: src/vlm_pipeline/i18n/vi.yaml (owner: bao-design). Output (generated, git-ignored): src/generated/app-copy.json.
The editor preview on the landing reads node titles, port labels, check-bar and error-card copy from here plus the
real template (configs/templates/*.json), never hand-typed lookalikes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parent.parent
ROOT = SITE.parent
KEYS = ("node", "port", "preflight", "node_state", "template", "canvas", "app")
ERRORS = ("MEM-001", "CFG-003")


def main() -> None:
    vi = yaml.safe_load((ROOT / "src/vlm_pipeline/i18n/vi.yaml").read_text(encoding="utf-8"))
    out = {k: vi[k] for k in KEYS if k in vi}
    out["error"] = {code: vi["error"][code] for code in ERRORS}
    snapshot = json.loads((ROOT / "schemas/node_types.snapshot.json").read_text(encoding="utf-8"))
    out["node_types"] = {
        n["type_id"]: {"category": n["category"], "inputs": n["inputs"], "outputs": n["outputs"]}
        for n in snapshot["node_types"]
    }
    out["port_types"] = snapshot["port_types"]
    out["templates"] = {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "configs/templates").glob("*.json"))
    }
    dest = SITE / "src/generated/app-copy.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"app_copy: {len(out['node'])} node types, {len(out['templates'])} templates", file=sys.stderr)


if __name__ == "__main__":
    main()
