"""A quick trial run of one training step's chain (MODEL-005, "Chạy thử nhanh"): the workflow cut to that node.

The trial runs the node and everything upstream of it, nothing downstream (no merge, eval or export of a 5-step
model). The workflow the user saved is never changed: the step count is a run option the executor carries
(``Executor.run(trial_max_steps=...)``), so a trial artifact has its own cache key and never stands for a real run.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

TRIAL_STEPS = 5


class TrialNotPossibleError(ValueError):
    """The node is not a step of this workflow that can run a trial."""


def trial_workflow(workflow: Mapping[str, Any], node_id: str, trainable: Collection[str]) -> dict[str, Any]:
    """``workflow`` reduced to ``node_id`` and its ancestors (nodes, edges, group definitions kept as they are).

    ``trainable``: the node types that can run a trial (the app passes ``nodes.train.TRIAL_CAPABLE``).
    """
    nodes = {n.get("id"): n for n in workflow.get("nodes", []) if isinstance(n, Mapping)}
    node = nodes.get(node_id)
    if node is None or node.get("type") not in trainable:
        raise TrialNotPossibleError(f"{node_id!r} is not a training step of this workflow")
    edges = [e for e in workflow.get("edges", []) if isinstance(e, Mapping)]
    keep = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        for edge in edges:
            src, dst = edge.get("from") or [None], edge.get("to") or [None]
            if dst[0] == current and src[0] in nodes and src[0] not in keep:
                keep.add(src[0])
                stack.append(src[0])
    trimmed = dict(workflow)
    trimmed["nodes"] = [n for n in workflow.get("nodes", []) if isinstance(n, Mapping) and n.get("id") in keep]
    trimmed["edges"] = [e for e in edges if (e.get("from") or [None])[0] in keep and (e.get("to") or [None])[0] in keep]
    return trimmed


__all__ = ["TRIAL_STEPS", "TrialNotPossibleError", "trial_workflow"]
