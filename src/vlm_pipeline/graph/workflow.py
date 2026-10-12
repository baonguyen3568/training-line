"""Workflow file shape (``.json`` saved / shared / templates): the single source of truth (P009 §3, P003 D11).

Workflow files are UNTRUSTED data: every object is ``extra="forbid"``. ``graph.validation`` derives its allowed keys
from these models and ``scripts/export_schemas.py`` exports ``schemas/workflow.schema.json`` for the frontend parser,
so backend and frontend cannot drift. Node params stay a free mapping here: they are validated per node type by the
node's own ``params_model``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WORKFLOW_FORMAT = "vlm-workflow"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NodeUi(_Strict):
    """Canvas-only state: never part of a cache key."""

    x: float
    y: float


class WorkflowNode(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    type: str = Field(max_length=129)  # charset enforced by graph.validation (ASCII-only, P003 D16)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    params: dict[str, object] = Field(default_factory=dict)
    ui: NodeUi | None = None
    # Registry lookup key only (P003 D16): never echoed, never used as an install source.
    package: str | None = Field(default=None, max_length=129)


class WorkflowEdge(_Strict):
    from_: tuple[str, str] = Field(alias="from")  # [node_id, output port]
    to: tuple[str, str]  # [node_id, input port]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class WorkflowRequires(_Strict):
    node_types: dict[str, str] = Field(default_factory=dict)  # type_id -> version range, e.g. ">=1.0"


class WorkflowFile(_Strict):
    format: Literal["vlm-workflow"]
    schema_version: Literal[1]
    name: str = Field(max_length=200)
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)
    requires: WorkflowRequires | None = None


def allowed_keys(model: type[BaseModel]) -> frozenset[str]:
    """JSON keys of ``model`` (aliases included) — what graph.validation accepts at that level."""
    return frozenset(f.alias or name for name, f in model.model_fields.items())


__all__ = [
    "WORKFLOW_FORMAT",
    "NodeUi",
    "WorkflowEdge",
    "WorkflowFile",
    "WorkflowNode",
    "WorkflowRequires",
    "allowed_keys",
]
