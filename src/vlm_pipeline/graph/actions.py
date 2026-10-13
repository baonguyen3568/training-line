"""The quick-fix actions sent with a validate issue or a node error event (G-3, contract with bao-frontend).

Both lists come from ONE NodeContext (the failing node's current params + its params model), so ``action_ids`` and
the resolved ``actions`` never disagree; the client applies a patch's final ``value`` as computed here (bounds,
scaling) and never runs catalog logic itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from vlm_pipeline.common.error_catalog import NodeContext, resolved_actions


@dataclass
class _Error:
    code: str
    params: Mapping[str, object] = field(default_factory=dict)
    resumable: bool = False


def issue_actions(
    error: object,
    *,
    node_id: str | None = None,
    node_params: Mapping[str, object] | None = None,
    params_model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """``{"action_ids": [...], "actions": [...]}`` for an error-like object (code, params, optional resumable).

    Without the node's params (unknown node, a group's inner step) patches are left out, never guessed.
    """
    err = _Error(
        code=str(getattr(error, "code", "SYS-000")),
        params=dict(getattr(error, "params", None) or {}),
        resumable=bool(getattr(error, "resumable", False)),
    )
    node = (
        NodeContext.from_model(dict(node_params), params_model)
        if node_params is not None and params_model is not None
        else None
    )
    actions = resolved_actions(err, node, node_id)
    return {"action_ids": [a["id"] for a in actions], "actions": actions}  # one list, two shapes: never disagree


__all__ = ["issue_actions"]
