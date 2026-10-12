"""Node type system: port types, node specs and the registry (P009 §2, ADR-004 A3).

Data only. ``graph/`` depends on this module and on a ``Registry`` passed to it; it never imports concrete nodes.
User-facing text is not stored here: titles, descriptions, port and param labels come from the i18n keys
``node.<type_id>.*`` and ``port.<label>`` (P009 §2.2). Parameters are pydantic models; their JSON Schema, with UI
hints under ``x-ui`` on each field, is what the editor renders (P004 §8.2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vlm_pipeline.common.params import Params, ui

Category = Literal["source", "data", "train", "test", "quantize", "eval", "export", "util"]
Runner = Literal["host", "container"]

TYPE_ID_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PortType(_Frozen):
    """A wire type. A subtype connects wherever its parent is expected (``QuantizedModel`` -> ``Model``)."""

    id: str
    parent: str | None = None
    color_token: str
    label_key: str


def _port(type_id: str, color: str, label: str, parent: str | None = None) -> PortType:
    return PortType(id=type_id, parent=parent, color_token=f"color.port.{color}", label_key=f"port.{label}")


PORT_TYPES: tuple[PortType, ...] = (
    _port("Dataset", "dataset", "dataset"),
    _port("Model", "model", "model"),
    _port("Checkpoint", "model", "checkpoint", parent="Model"),
    _port("QuantizedModel", "model", "quantized_model", parent="Model"),
    _port("Adapter", "model", "adapter"),
    _port("ExportArtifact", "model", "export_artifact"),
    _port("EvalReport", "report", "eval_report"),
    _port("BenchReport", "report", "bench_report"),
    _port("Endpoint", "endpoint", "endpoint"),
)
_PARENT = {p.id: p.parent for p in PORT_TYPES}


def is_compatible(src_type: str, dst_type: str) -> bool:
    """Whether an output of ``src_type`` may feed an input of ``dst_type`` (same type or a subtype of it)."""
    current: str | None = src_type
    while current is not None:
        if current == dst_type:
            return True
        current = _PARENT.get(current)
    return False


class PortSpec(_Frozen):
    type: str
    optional: bool = False
    many: bool = False


class EmptyParams(Params):
    """Params model of a node without parameters."""


class NodeSpec(_Frozen):
    """Declaration of one node type. Serialised by ``to_api()`` for ``GET /api/node-types``."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    type_id: str = Field(pattern=TYPE_ID_PATTERN)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    category: Category
    icon: str
    inputs: Mapping[str, PortSpec] = Field(default_factory=dict)
    outputs: Mapping[str, PortSpec] = Field(default_factory=dict)
    params_model: type[BaseModel] = EmptyParams
    cacheable: bool = True
    runner: Runner
    image: str | None = Field(None, pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")  # always pinned by digest

    @property
    def image_digest(self) -> str | None:
        """``sha256:<hex>`` of a container node's image (part of the cache key), ``None`` for host nodes."""
        if self.image is None:
            return None
        _, sep, digest = self.image.partition("@")
        return digest if sep else None

    def validate_params(self, raw: Mapping[str, object]) -> BaseModel:
        """Validated params with defaults filled in; raises ``pydantic.ValidationError``."""
        return self.params_model.model_validate(dict(raw))

    def params_schema(self) -> dict[str, Any]:
        return self.params_model.model_json_schema()

    def to_api(self) -> dict[str, Any]:
        """Item of ``GET /api/node-types`` (frontend ``NodeSpec``, frontend/src/api/types.ts)."""
        return {
            "type_id": self.type_id,
            "version": self.version,
            "category": self.category,
            "icon": self.icon,
            "inputs": {name: port.model_dump() for name, port in self.inputs.items()},
            "outputs": {name: port.model_dump() for name, port in self.outputs.items()},
            "params_schema": self.params_schema(),
            "cacheable": self.cacheable,
            "runner": self.runner,
            "origin": {"kind": "first_party"},
            "permissions": [],
            "policy_block": None,
        }


class Registry:
    """Immutable set of node specs keyed by ``type_id``.

    Checks at construction: unique ids, known port types, strict params models, and (unless ``allow_unpinned``,
    used until the worker images are wired in M1) a digest-pinned image for every container node.
    """

    def __init__(
        self,
        specs: Iterable[NodeSpec],
        port_types: Iterable[PortType] = PORT_TYPES,
        *,
        allow_unpinned: bool = False,
    ) -> None:
        self.port_types: tuple[PortType, ...] = tuple(port_types)
        known_ports = {p.id for p in self.port_types}
        by_id: dict[str, NodeSpec] = {}
        for spec in specs:
            if spec.type_id in by_id:
                raise ValueError(f"duplicate node type_id {spec.type_id!r}")
            for name, port in {**spec.inputs, **spec.outputs}.items():
                if port.type not in known_ports:
                    raise ValueError(f"{spec.type_id}: port {name!r} has unknown type {port.type!r}")
            if spec.params_model.model_config.get("extra") != "forbid":
                raise ValueError(f"{spec.type_id}: params model must forbid unknown keys (extra='forbid')")
            if spec.runner == "container" and spec.image is None and not allow_unpinned:
                raise ValueError(f"{spec.type_id}: container node needs an image pinned by digest")
            by_id[spec.type_id] = spec
        self._by_id = by_id

    def get(self, type_id: str) -> NodeSpec | None:
        return self._by_id.get(type_id)

    def all(self) -> tuple[NodeSpec, ...]:
        return tuple(self._by_id.values())

    def to_api(self) -> dict[str, Any]:
        """Body of ``GET /api/node-types``."""
        return {
            "node_types": [spec.to_api() for spec in self._by_id.values()],
            "port_types": [p.model_dump() for p in self.port_types],
        }


__all__ = [
    "PORT_TYPES",
    "Category",
    "EmptyParams",
    "NodeSpec",
    "PortSpec",
    "PortType",
    "Registry",
    "Runner",
    "is_compatible",
    "ui",
]
