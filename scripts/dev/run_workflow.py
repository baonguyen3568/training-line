"""Run a workflow file through the app's executor without the HTTP layer (dev/E2E of real workers).

``uv run python scripts/dev/run_workflow.py WORKFLOW.json [--worker container|fake]`` prints events as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.nodes import REGISTRY


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("workflow", type=Path)
    p.add_argument("--worker", choices=["container", "fake"], default="container")
    a = p.parse_args()
    workflow = json.loads(a.workflow.read_text(encoding="utf-8"))
    services = DefaultServices(registry=REGISTRY, root=project_root(), worker=a.worker)
    issues = services.validate(workflow)
    if any(i.get("severity", "error") == "error" for i in issues):
        print(json.dumps({"invalid": issues}))
        return 2
    gid = f"g-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    services.run_graph(workflow, gid, lambda e: print(json.dumps(e, default=str), flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
