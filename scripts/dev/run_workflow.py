"""Run a workflow file through the app's executor without the HTTP layer (dev/E2E of real workers).

``uv run python scripts/dev/run_workflow.py WORKFLOW.json [--worker container|fake] [--set NODE.PARAM=VALUE ...]``
prints events as JSON lines. A template file (``configs/templates/*.json``) works too. ``--set`` overrides a node param
for a quick run (VALUE is JSON when it parses, else a string), e.g. ``--set n2.max_samples=100 --set n4.epochs=1``.
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
    p.add_argument("--set", action="append", default=[], metavar="NODE.PARAM=VALUE")
    a = p.parse_args()
    data = json.loads(a.workflow.read_text(encoding="utf-8"))
    workflow = data.get("workflow", data)  # templates wrap the workflow
    nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    for item in a.set:
        target, eq, raw = item.partition("=")
        node_id, dot, param = target.partition(".")
        if not (eq and dot and param) or node_id not in nodes:
            p.error(f"--set {item!r}: expected NODE.PARAM=VALUE with an existing node id")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        nodes[node_id].setdefault("params", {})[param] = value
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
