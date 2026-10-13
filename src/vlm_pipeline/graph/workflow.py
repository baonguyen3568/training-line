"""Workflow file shape (``.json`` saved / shared / templates): the single source of truth (P009 §3, P003 D11).

Workflow files are UNTRUSTED data: every object is ``extra="forbid"``. ``graph.validation`` derives its allowed keys
from these models and ``scripts/export_schemas.py`` exports ``schemas/workflow.schema.json`` for the frontend parser,
so backend and frontend cannot drift. Node params stay a free mapping here: they are validated per node type by the
node's own ``params_model``.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


_NAME = r"^[a-z][a-z0-9_]{0,63}$"  # exposed port / param name of a group
MAX_GROUP_DEFINITIONS = 50  # per workflow file (P016 §2, security)
MAX_GROUP_EDGES = 2000  # per definition (security G1)
MAX_GROUP_PORTS = 100  # exposed inputs / outputs / params per definition
MAX_INPUT_TARGETS = 500  # inner ports all exposed inputs of one definition fan out to


class GroupDefinition(_Strict):
    """A group (subgraph) definition: DATA expanded into its inner nodes before validation (P016 §2).

    First-party groups (``group.*``, ``configs/groups/*.json``) carry an i18n ``title_key``; a user group (``user.*``)
    travels inside the workflow file and is untrusted: plain-text title, never a key.
    """

    id: str = Field(pattern=r"^(user|group)\.[a-z0-9][a-z0-9_-]{0,47}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str | None = Field(default=None, max_length=80)
    title_key: str | None = Field(default=None, pattern=r"^groups\.[a-z0-9_.]+$")
    description: str | None = Field(default=None, max_length=500)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=500)
    edges: list[WorkflowEdge] = Field(default_factory=list, max_length=MAX_GROUP_EDGES)
    inputs: dict[str, list[tuple[str, str]]] = Field(
        default_factory=dict, max_length=MAX_GROUP_PORTS
    )  # name -> [[inner node, input port], ...]
    outputs: dict[str, tuple[str, str]] = Field(
        default_factory=dict, max_length=MAX_GROUP_PORTS
    )  # name -> [node, port]
    params: dict[str, tuple[str, str]] = Field(
        default_factory=dict, max_length=MAX_GROUP_PORTS
    )  # name -> [node, param]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        user = self.id.startswith("user.")
        if user and (self.title_key is not None or not self.title):
            raise ValueError("a user group has a plain-text title and no title_key")
        if not user and self.title_key is None:
            raise ValueError("a first-party group has a title_key")
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids) or any("__" in i for i in ids):
            raise ValueError("inner node ids are unique and never contain '__' (the expansion separator)")
        names = [*self.inputs, *self.outputs, *self.params]
        if bad := sorted(n for n in names if not re.match(_NAME, n)):
            raise ValueError(f"invalid exposed names {bad}")
        targets = [t for ts in self.inputs.values() for t in ts] + [*self.outputs.values(), *self.params.values()]
        targets += [(e.from_[0], "") for e in self.edges] + [(e.to[0], "") for e in self.edges]
        if missing := sorted({node for node, _ in targets} - set(ids)):
            raise ValueError(f"unknown inner nodes {missing}")
        if any(not ts for ts in self.inputs.values()):
            raise ValueError("an exposed input maps to at least one inner port")
        if sum(len(ts) for ts in self.inputs.values()) > MAX_INPUT_TARGETS:
            raise ValueError(f"exposed inputs fan out to more than {MAX_INPUT_TARGETS} inner ports")
        return self


class WorkflowRequires(_Strict):
    node_types: dict[str, str] = Field(default_factory=dict)  # type_id -> version range, e.g. ">=1.0"


class WorkflowFile(_Strict):
    format: Literal["vlm-workflow"]
    schema_version: Literal[1]
    name: str = Field(max_length=200)
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)
    requires: WorkflowRequires | None = None
    groups: list[GroupDefinition] = Field(default_factory=list, max_length=MAX_GROUP_DEFINITIONS)  # user groups


def allowed_keys(model: type[BaseModel]) -> frozenset[str]:
    """JSON keys of ``model`` (aliases included) — what graph.validation accepts at that level."""
    return frozenset(f.alias or name for name, f in model.model_fields.items())


__all__ = [
    "MAX_GROUP_DEFINITIONS",
    "WORKFLOW_FORMAT",
    "GroupDefinition",
    "NodeUi",
    "WorkflowEdge",
    "WorkflowFile",
    "WorkflowNode",
    "WorkflowRequires",
    "allowed_keys",
]
