"""Group (subgraph) expansion (P016 §2): group instances become their inner nodes BEFORE validation.

A group is DATA (``graph.workflow.GroupDefinition``). Validation, cache keys and the executor only ever see the flat
graph: an inner node gets the id ``<group node>__<inner id>`` (recursively, at most 64 chars: the same charset and
length as any node id, because run dirs, run ids and container names are built from it; the UI may show ``/``) and
caches exactly like a hand-placed node.
Issues on inner nodes are re-anchored to the top-level group node with ``inner_path`` = the ids per level.

Limits are enforced WHILE expanding (security): nesting depth, total nodes after expansion, definitions per file.
User groups live in the ``user.*`` namespace and can never redefine a node type or a first-party group.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_pipeline.common.paths import configs_root
from vlm_pipeline.graph.workflow import MAX_GROUP_DEFINITIONS, GroupDefinition
from vlm_pipeline.nodes.spec import Issue, Registry

GROUPS_DIR = Path("configs") / "groups"
GROUP_CODE = "CFG-012"
MAX_DEPTH = 3  # group inside group inside group
MAX_NODES = 500  # expansion budget: group instances (expanded or refused) + flat nodes (security G1: no N^k blow-up)
MAX_EDGES = 4 * MAX_NODES  # flat wires generated from groups (exposed inputs fan out)
MAX_ISSUES = 200  # the first ones are reported; a hostile file cannot make millions
SEPARATOR = "__"  # user-chosen node ids never contain it (checked here and in GroupDefinition)
MAX_ID_LENGTH = 64  # generated ids obey graph.workflow.WorkflowNode.id like any other id

_FIRST_PARTY_ID = re.compile(r"^group\.[a-z0-9][a-z0-9_-]{0,47}$")  # same pattern as GroupDefinition.id
Endpoint = tuple[str, str]  # (flat node id, port)


@dataclass(frozen=True)
class Expansion:
    """The flat workflow, where each generated id comes from, and the problems found while expanding."""

    workflow: Mapping[str, Any]  # same file, group instances replaced by their inner nodes, no ``groups`` key
    origins: Mapping[str, tuple[str, tuple[str, ...]]]  # generated id -> (top-level group node, inner path)
    issues: tuple[Issue, ...]

    def anchor(self, issue: Issue) -> Issue:
        """``issue`` moved onto the top-level group node that contains its node (unchanged otherwise)."""
        origin = self.origins.get(issue.node_id or "")
        return issue if origin is None else replace(issue, node_id=origin[0], inner_path=origin[1])


class _OverLimitError(Exception):
    def __init__(self, reason: str, limit: int) -> None:
        super().__init__(reason)
        self.reason, self.limit = reason, limit


@functools.cache  # read once per process: the files ship with the app (an edit needs a restart)
def first_party_groups(root: Path | None = None) -> Mapping[str, GroupDefinition]:
    """``configs/groups/*.json`` (``group.*`` only); a broken file fails loudly: it ships with the app."""
    found: dict[str, GroupDefinition] = {}
    for path in sorted((configs_root(root) / GROUPS_DIR).glob("*.json")):
        group = GroupDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if not group.id.startswith("group.") or group.id in found:
            raise ValueError(f"{path.name}: first-party group ids are unique and under group.*")
        found[group.id] = group
    return found


def expand(workflow: object, registry: Registry, first_party: Mapping[str, GroupDefinition] | None = None) -> Expansion:
    """Inline every group instance of ``workflow``. Never raises on bad input: problems become issues."""
    if not isinstance(workflow, Mapping):
        return Expansion(workflow={}, origins={}, issues=())
    known = first_party_groups() if first_party is None else first_party
    issues: list[Issue] = []
    groups = {**known, **_user_groups(workflow.get("groups", []), registry, known, issues)}
    raw_nodes, raw_edges = workflow.get("nodes"), workflow.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return Expansion(workflow={k: v for k, v in workflow.items() if k != "groups"}, origins={}, issues=())
    before = len(issues)
    inliner = _Inliner(groups, registry, issues)
    try:
        nodes = inliner.top_level(raw_nodes)
        edges = inliner.inner_edges + [e for raw in raw_edges for e in inliner.edge(raw)]
    except _OverLimitError as over:  # the file is refused as a whole: only this issue, nothing half-expanded
        issue = Issue(GROUP_CODE, params={"group": "", "reason": over.reason, "limit": over.limit})
        return Expansion(workflow={"nodes": [], "edges": []}, origins={}, issues=(*issues[:before], issue))
    flat = {**{k: v for k, v in workflow.items() if k != "groups"}, "nodes": nodes, "edges": edges}
    return Expansion(workflow=flat, origins=inliner.origins, issues=tuple(issues))


def _user_groups(
    raw: object, registry: Registry, first_party: Mapping[str, GroupDefinition], issues: list[Issue]
) -> dict[str, GroupDefinition]:
    if not isinstance(raw, list):
        issues.append(Issue("CFG-007", params={"field": "groups"}))
        return {}
    if len(raw) > MAX_GROUP_DEFINITIONS:
        issues.append(
            Issue(GROUP_CODE, params={"group": "", "reason": "too_many_definitions", "limit": MAX_GROUP_DEFINITIONS})
        )
        return {}
    found: dict[str, GroupDefinition] = {}
    for item in raw:
        raw_id = item.get("id") if isinstance(item, Mapping) else None
        if isinstance(raw_id, str) and _FIRST_PARTY_ID.match(raw_id):
            issues.append(Issue(GROUP_CODE, params={"group": raw_id, "reason": "namespace"}))
            continue
        try:
            group = GroupDefinition.model_validate(item)
        except ValidationError:
            # The id is echoed only once it passed the pattern (never an arbitrary string from the file).
            issues.append(Issue(GROUP_CODE, params={"group": "", "reason": "invalid_definition"}))
            continue
        if group.id in found or group.id in first_party or registry.get(group.id) is not None:
            issues.append(Issue(GROUP_CODE, params={"group": group.id, "reason": "name_clash"}))
            continue
        found[group.id] = group
    return found


class _Inliner:
    """One expansion pass: flat nodes, the edges inside groups, and the exposed-port maps of every instance."""

    def __init__(self, groups: Mapping[str, GroupDefinition], registry: Registry, issues: list[Issue]) -> None:
        self.groups, self.registry, self.issues = groups, registry, issues
        self.origins: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.instances: dict[str, GroupDefinition] = {}  # flat id of an expanded group instance -> its definition
        self.failed: set[str] = set()  # instances not expanded (already reported): their wires are dropped quietly
        self.inner_edges: list[dict[str, Any]] = []
        self.nodes: list[Any] = []
        self.checked: set[str] = set()
        self.spent = 0  # expansion budget used (MAX_NODES)
        self.wires = 0  # wires generated (MAX_EDGES)

    def top_level(self, raw_nodes: list[Any]) -> list[Any]:
        for raw in raw_nodes:
            node_id = raw.get("id") if isinstance(raw, Mapping) else None
            if isinstance(node_id, str) and SEPARATOR in node_id:
                self._add(Issue("CFG-007", params={"field": "nodes.id"}))
                continue
            group = self.groups.get(str(raw.get("type"))) if isinstance(raw, Mapping) else None
            if group is None or not isinstance(node_id, str):
                self.nodes.append(raw)  # plain node (or broken: graph.validation reports it)
                continue
            self._inline(node_id, raw.get("params", {}), group, root=node_id, path=(), stack=())
        return self.nodes

    def _inline(
        self,
        flat_id: str,
        params: object,
        group: GroupDefinition,
        *,
        root: str,
        path: tuple[str, ...],
        stack: tuple[str, ...],
    ) -> None:
        at = (root, path)
        self._spend()  # every instance counts, even one refused below
        if group.id in stack or len(stack) >= MAX_DEPTH:
            if group.id in stack:
                self._report(GROUP_CODE, at, {"group": group.id, "reason": "recursion"})
            else:
                self._report(GROUP_CODE, at, {"group": group.id, "reason": "too_deep", "limit": MAX_DEPTH})
            self.failed.add(flat_id)
            return
        if not isinstance(params, Mapping):
            params = {}
            self._report("CFG-007", at, {"field": "nodes.params"})
        for name in sorted(set(map(str, params)) - set(group.params)):
            rule = {"node": root, "param": name, "rule": "extra_forbidden"}
            self._report("CFG-005", at, rule, param=name)
        self._check_mappings(group, at)
        self.instances[flat_id] = group
        for inner in group.nodes:
            inner_id = f"{flat_id}{SEPARATOR}{inner.id}"
            inner_path = (*path, inner.id)
            self.origins[inner_id] = (root, inner_path)
            inner_params = dict(inner.params)
            for name, (target, param) in group.params.items():
                if target == inner.id and name in params:
                    inner_params[param] = params[name]
            nested = self.groups.get(inner.type)
            if nested is not None:
                self._inline(inner_id, inner_params, nested, root=root, path=inner_path, stack=(*stack, group.id))
                continue
            self._spend()
            if len(inner_id) > MAX_ID_LENGTH:
                self._report(GROUP_CODE, at, {"group": group.id, "reason": "id_too_long"})
                self.failed.add(inner_id)
                continue
            node = {"id": inner_id, "type": inner.type, "version": inner.version, "params": inner_params}
            self.nodes.append(node if inner.package is None else {**node, "package": inner.package})
        for edge in group.edges:
            self.inner_edges += self._wire(
                (f"{flat_id}{SEPARATOR}{edge.from_[0]}", edge.from_[1]),
                (f"{flat_id}{SEPARATOR}{edge.to[0]}", edge.to[1]),
            )

    def _report(
        self, code: str, at: tuple[str, tuple[str, ...]], params: Mapping[str, object], param: str | None = None
    ) -> None:
        self._add(Issue(code, node_id=at[0], inner_path=at[1], param=param, params=params))

    def _add(self, issue: Issue) -> None:
        if len(self.issues) < MAX_ISSUES:
            self.issues.append(issue)

    def _spend(self) -> None:
        self.spent += 1
        if self.spent > MAX_NODES:
            raise _OverLimitError("too_many_nodes", MAX_NODES)

    def _check_mappings(self, group: GroupDefinition, at: tuple[str, tuple[str, ...]]) -> None:
        """Every exposed port/param points at a real port/param of its inner node (checked once per definition)."""
        if group.id in self.checked:
            return
        self.checked.add(group.id)
        types = {n.id: n.type for n in group.nodes}

        def exposes(inner: str, kind: str, name: str) -> bool:
            nested = self.groups.get(types[inner])
            if nested is not None:
                exposed: Mapping[str, object] = getattr(nested, kind)  # its own inputs / outputs / params
                return name in exposed
            spec = self.registry.get(types[inner])
            if spec is None:
                return True  # unknown type: CFG-004 on the inner node says it better
            if kind == "params":
                properties: Mapping[str, Any] = spec.params_schema().get("properties") or {}
                return name in properties
            ports = spec.inputs if kind == "inputs" else spec.outputs
            return name in ports

        pairs = [("inputs", t) for ts in group.inputs.values() for t in ts]
        pairs += [("outputs", t) for t in group.outputs.values()] + [("params", t) for t in group.params.values()]
        if not all(exposes(inner, kind, name) for kind, (inner, name) in pairs):
            self._report(GROUP_CODE, at, {"group": group.id, "reason": "bad_mapping"})

    def edge(self, raw: object) -> list[Any]:
        """A top-level edge with group ends resolved to inner ports; malformed edges pass through for validation."""
        ends = _ends(raw)
        if ends is None or not any(n in self.instances or n in self.failed for n, _ in ends):
            return [raw]  # untouched: graph.validation checks its keys and ends
        return self._wire(*ends)

    def _wire(self, src: Endpoint, dst: Endpoint) -> list[dict[str, Any]]:
        if src[0] in self.failed or dst[0] in self.failed:
            return []
        out = self._resolve_out(src)
        ins = self._resolve_in(dst)
        if out is None or ins is None:
            missing = src[0] if out is None else dst[0]
            self._add(Issue("CFG-006", node_id=missing, params={"node_id": missing}))
            return []
        self.wires += len(ins)
        if self.wires > MAX_EDGES:
            raise _OverLimitError("too_many_edges", MAX_EDGES)
        return [{"from": list(out), "to": list(end)} for end in ins]

    def _resolve_out(self, end: Endpoint) -> Endpoint | None:
        node, port = end
        while node in self.instances:
            mapped = self.instances[node].outputs.get(port)
            if mapped is None:
                return None
            node, port = f"{node}{SEPARATOR}{mapped[0]}", mapped[1]
        return node, port

    def _resolve_in(self, end: Endpoint) -> list[Endpoint] | None:
        node, port = end
        if node not in self.instances:
            return [end]
        targets = self.instances[node].inputs.get(port)
        if targets is None:
            return None
        resolved: list[Endpoint] = []
        for inner, inner_port in targets:
            found = self._resolve_in((f"{node}{SEPARATOR}{inner}", inner_port))
            if found is None:
                return None
            resolved += found
        return resolved


def _ends(raw: object) -> tuple[Endpoint, Endpoint] | None:
    if not isinstance(raw, Mapping):
        return None
    ends = [raw.get("from"), raw.get("to")]
    if not all(isinstance(e, list | tuple) and len(e) == 2 and all(isinstance(x, str) for x in e) for e in ends):
        return None
    (a, b), (c, d) = ends  # type: ignore[misc]
    return (a, b), (c, d)


__all__ = ["GROUP_CODE", "MAX_DEPTH", "MAX_EDGES", "MAX_NODES", "Expansion", "expand", "first_party_groups"]
