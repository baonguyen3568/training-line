"""Pre-run validation of a workflow file (P009 §4, P003 D11/D16). Pure: never raises, never opens the network.

Returns ``Issue``s (catalog code + params only; text comes from i18n at the API edge). Order: file shape (CFG-007),
node ids (CFG-008), node types (CFG-004), params (CFG-005 / SEC-001 / SEC-002), edges (CFG-006 / CFG-001),
cycles (CFG-002), missing inputs (CFG-003), then the registry's cross-node rules (e.g. CFG-010).
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from pydantic import ValidationError

if TYPE_CHECKING:
    from vlm_pipeline.common.errors import VlmPipelineError
    from vlm_pipeline.graph.worker import SupportQuery

from vlm_pipeline.graph.workflow import WORKFLOW_FORMAT, WorkflowEdge, WorkflowFile, WorkflowNode, allowed_keys
from vlm_pipeline.nodes.spec import LOCAL_PATH_FORMAT, GraphView, Issue, NodeSpec, Registry, is_compatible

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
# Allowed keys come from the workflow file models (single source shared with the frontend via JSON schema).
_TOP_KEYS = allowed_keys(WorkflowFile)
_NODE_KEYS = allowed_keys(WorkflowNode)
_EDGE_KEYS = allowed_keys(WorkflowEdge)
# Workflow node types (first-party "a.b" and plugin "publisher/pack.node"): ASCII only (P003 D16) so homoglyph,
# zero-width and bidi tricks are rejected before any lookup or echo.
_TYPE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}(/[a-z0-9][a-z0-9_.-]{0,63})?$")
_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")


# ----------------------------------------------------------------------------------------------- helpers


def _is_secret(value: object) -> bool:
    if isinstance(value, str):
        return bool(_SECRET.search(value))
    if isinstance(value, Mapping):
        return any(_is_secret(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_is_secret(v) for v in value)
    return False


def _decode(path: str) -> str:
    """Percent-decode repeatedly (``%252e`` -> ``%2e`` -> ``.``) and normalise unicode."""
    for _ in range(5):
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    return unicodedata.normalize("NFKC", path).replace("\\", "/")


def path_escapes_workspace(raw: str, workspace: Path | None) -> bool:
    """True if ``raw`` is absolute, climbs out with ``..``, or (with a workspace) resolves outside it via symlinks."""
    path = _decode(raw)
    if not path or path.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", path) or "\x00" in path:
        return True
    if ".." in Path(path).parts:
        return True
    if workspace is not None:
        root = workspace.resolve()
        target = Path(os.path.realpath(root / path))
        if target != root and root not in target.parents:
            return True
    return False


def _local_path_params(spec: NodeSpec) -> set[str]:
    props = spec.params_schema().get("properties") or {}
    return {
        name for name, prop in props.items() if isinstance(prop, Mapping) and prop.get("format") == LOCAL_PATH_FORMAT
    }


def _lookalike(type_id: str, known: Iterable[str]) -> str | None:
    """A registered type id that ``type_id`` imitates (edit distance <= 2 after ASCII confusables), else None."""

    def norm(s: str) -> str:
        s = s.lower().replace("rn", "m").replace("vv", "w")
        table: dict[str, str | int | None] = {"0": "o", "1": "l", "|": "l", "_": "-", ".": "-", "/": "-"}
        return s.translate(str.maketrans(table))

    def dist(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    target = norm(type_id)
    for candidate in known:
        if candidate != type_id and dist(target, norm(candidate)) <= 2:
            return candidate
    return None


# ---------------------------------------------------------------------------------------------- validate


def _check_node(raw: Mapping[str, Any], registry: Registry, workspace: Path | None) -> list[Issue]:
    node_id = str(raw.get("id"))
    issues = [Issue("CFG-007", node_id=node_id, params={"field": f"nodes.{k}"}) for k in sorted(set(raw) - _NODE_KEYS)]
    type_id = raw.get("type")
    if not isinstance(type_id, str) or not _TYPE_ID.match(type_id):
        # Never echo an invalid identifier: only the field name (P003 D16).
        return [
            *issues,
            Issue("CFG-007", node_id=node_id, params={"field": "nodes.type", "reason": "invalid_identifier"}),
        ]
    spec = registry.get(type_id)
    if spec is None:
        # trust_level / suggested_package come from OUR registry only; a package hint in the workflow is never echoed.
        lookalike = _lookalike(type_id, (s.type_id for s in registry.all()))
        params = {
            "node_type": type_id,
            "reason": "not_installed",
            "trust_level": "unverified",
            "suggested_package": None,
            "name_lookalike_of": lookalike,
        }
        return [*issues, Issue("CFG-004", node_id=node_id, params=params)]
    params_raw = raw.get("params", {})
    if not isinstance(params_raw, Mapping):
        return [*issues, Issue("CFG-007", node_id=node_id, params={"field": "nodes.params"})]
    for name, value in params_raw.items():
        if _is_secret(value):
            issues.append(Issue("SEC-002", node_id=node_id, param=str(name), params={"param": str(name)}))
    path_params = _local_path_params(spec)
    escaping = {
        n for n in path_params if isinstance(v := params_raw.get(n), str) and path_escapes_workspace(v, workspace)
    }
    try:
        spec.validate_params(params_raw)
    except ValidationError as exc:
        for e in exc.errors():
            param = str(e["loc"][0]) if e.get("loc") else None
            if param in path_params:
                escaping.add(param)  # LocalPath itself rejects absolute/".." paths: report it as SEC-001
                continue
            # The offending value is not echoed (it may be a secret); the rule is the pydantic error type.
            rule = {"node": node_id, "param": param, "rule": e["type"]}
            issues.append(Issue("CFG-005", node_id=node_id, param=param, params=rule))
    issues.extend(Issue("SEC-001", node_id=node_id, param=n, params={"path": n}) for n in sorted(escaping))
    return issues


def _parse_edge(raw: object) -> tuple[str, str, str, str] | None:
    if not isinstance(raw, Mapping) or set(raw) - _EDGE_KEYS:
        return None
    ends: list[str] = []
    for end in (raw.get("from"), raw.get("to")):
        if not isinstance(end, list | tuple) or len(end) != 2 or not all(isinstance(x, str) for x in end):
            return None
        ends.extend(str(x) for x in end)
    return ends[0], ends[1], ends[2], ends[3]


def _find_cycle(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str] | None:
    """Nodes on or behind a cycle (Kahn's algorithm leftovers), in a stable order; None if the graph is acyclic.

    Iterative on purpose: a recursive DFS hits RecursionError on long chains and would be misreported as CFG-007.
    """
    indeg: dict[str, int] = dict.fromkeys(nodes, 0)
    children: dict[str, list[str]] = {n: [] for n in indeg}
    for a, b in edges:
        if a in indeg and b in indeg:
            children[a].append(b)
            indeg[b] += 1
    ready = [n for n, d in indeg.items() if d == 0]
    while ready:
        for child in children[ready.pop()]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    left = [n for n, d in indeg.items() if d > 0]
    return left or None


CanRun = Callable[[NodeSpec], bool]  # the worker that will run the graph: can it run this node type? (CFG-011)
Support = Callable[["SupportQuery"], "VlmPipelineError | None"]  # graph.worker.worker_support_problem bound to it


def validate(
    workflow: object,
    registry: Registry,
    workspace: Path | None = None,
    *,
    can_run: CanRun | None = None,
    support: Support | None = None,
) -> list[Issue]:
    """All problems of ``workflow`` that block a run. Empty list = runnable (as far as static checks go).

    ``can_run`` (from ``graph.worker.worker_supports``): nodes the actual worker cannot run yet are reported before
    the run (CFG-011) instead of failing halfway through it. ``support`` (``worker_support_problem`` bound to the
    worker) asks the SAME check the worker runs before each node, per node x model x params x input kinds.
    """
    try:
        return _validate(workflow, registry, workspace, can_run, support)
    except Exception:  # noqa: BLE001 - contract: never raise on bad input
        return [Issue("CFG-007", params={"field": "workflow"})]


def _validate(
    workflow: object, registry: Registry, workspace: Path | None, can_run: CanRun | None, support: Support | None
) -> list[Issue]:
    if not isinstance(workflow, Mapping):
        return [Issue("CFG-007", params={"field": "workflow"})]
    issues = [Issue("CFG-007", params={"field": str(k)}) for k in sorted(map(str, set(workflow) - _TOP_KEYS))]
    if workflow.get("format") != WORKFLOW_FORMAT or workflow.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        issues.append(Issue("CFG-007", params={"field": "format"}))
    raw_nodes, raw_edges = workflow.get("nodes", []), workflow.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return [*issues, Issue("CFG-007", params={"field": "nodes" if not isinstance(raw_nodes, list) else "edges"})]
    if not raw_nodes:
        issues.append(Issue("CFG-007", params={"field": "nodes"}))
    seen, specs, raw_params = _check_nodes(raw_nodes, registry, workspace, issues)
    links, connected, sources = _check_edges(raw_edges, seen, specs, issues)
    cycle = _find_cycle(seen, links)
    if cycle:
        issues.append(Issue("CFG-002", node_id=cycle[0], params={"nodes": ", ".join(cycle)}))
    for node_id, spec in specs.items():
        for name, port in spec.inputs.items():
            if not port.optional and (node_id, name) not in connected:
                issues.append(Issue("CFG-003", node_id=node_id, port=name, params={"node": node_id, "port": name}))
    if can_run is not None:
        issues += [
            Issue("CFG-011", node_id=node_id, params={"node": node_id, "node_type": spec.type_id})
            for node_id, spec in specs.items()
            if not can_run(spec)
        ]
    view = GraphView(specs=specs, params=_valid_params(specs, raw_params), sources=sources)
    for rule in registry.rules:
        issues.extend(rule(view))
    if support is not None:
        issues += _support_issues(view, support, already={(i.code, i.node_id) for i in issues})
    return issues


def _support_issues(view: GraphView, support: Support, *, already: set[tuple[str, str | None]]) -> list[Issue]:
    from vlm_pipeline.common.catalog import load_catalog
    from vlm_pipeline.graph.worker import SupportQuery
    from vlm_pipeline.nodes.rules import input_kinds, model_input, origin_model_id

    catalog = load_catalog()
    out: list[Issue] = []
    for node_id, params in view.params.items():
        port = model_input(view, node_id)
        model = catalog.get(origin_model_id(view, node_id, port) or "") if port else None
        query = SupportQuery(node_id, view.specs[node_id], params, model, input_kinds(view, node_id))
        problem = support(query)
        if problem is not None and (problem.code, node_id) not in already:  # e.g. CFG-010 also comes from a rule
            out.append(Issue(problem.code, node_id=node_id, params=dict(problem.params)))
    return out


def _valid_params(specs: Mapping[str, NodeSpec], raw_params: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    valid: dict[str, Any] = {}
    for node_id, spec in specs.items():
        try:
            valid[node_id] = spec.validate_params(raw_params.get(node_id, {}))
        except ValidationError:
            continue  # already reported as CFG-005 / SEC-001
    return valid


def _check_nodes(
    raw_nodes: list[object], registry: Registry, workspace: Path | None, issues: list[Issue]
) -> tuple[set[str], dict[str, NodeSpec], dict[str, Mapping[str, Any]]]:
    """Per-node checks; returns the node ids seen, the specs of nodes valid enough to wire and their raw params."""
    specs: dict[str, NodeSpec] = {}
    raw_params: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, int] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str) or not _NODE_ID.match(raw["id"]):
            issues.append(Issue("CFG-007", params={"field": "nodes.id"}))
            continue
        node_id = raw["id"]
        counts[node_id] = counts.get(node_id, 0) + 1
        if counts[node_id] > 1:
            continue
        node_issues = _check_node(raw, registry, workspace)
        issues.extend(node_issues)
        bad_type = any(i.params.get("field") == "nodes.type" for i in node_issues)
        spec = None if bad_type else registry.get(str(raw.get("type")))
        if spec is not None:
            specs[node_id] = spec
            params = raw.get("params", {})
            raw_params[node_id] = params if isinstance(params, Mapping) else {}
    for node_id, n in counts.items():
        if n > 1:
            issues.append(Issue("CFG-008", node_id=node_id, params={"node_id": node_id, "duplicate_count": n}))
    return set(counts), specs, raw_params


def _check_edges(
    raw_edges: list[object], seen: set[str], specs: Mapping[str, NodeSpec], issues: list[Issue]
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int], dict[tuple[str, str], tuple[str, ...]]]:
    links: list[tuple[str, str]] = []
    connected: dict[tuple[str, str], int] = {}
    sources: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw_edge in raw_edges:
        edge = _parse_edge(raw_edge)
        if edge is None:
            issues.append(Issue("CFG-007", params={"field": "edges"}))
            continue
        src, out, dst, inp = edge
        if src not in seen or dst not in seen:
            missing = src if src not in seen else dst
            issues.append(Issue("CFG-006", node_id=dst if dst in seen else None, params={"node_id": missing}))
            continue
        links.append((src, dst))
        s_spec, d_spec = specs.get(src), specs.get(dst)
        if s_spec is None or d_spec is None:
            continue  # unknown / invalid type already reported
        if out not in s_spec.outputs or inp not in d_spec.inputs:
            bad = src if out not in s_spec.outputs else dst
            issues.append(Issue("CFG-006", node_id=dst, port=inp, params={"node_id": bad}))
            continue
        connected[(dst, inp)] = connected.get((dst, inp), 0) + 1
        s_type, d_type = s_spec.outputs[out].type, d_spec.inputs[inp].type
        too_many = connected[(dst, inp)] > 1 and not d_spec.inputs[inp].many
        if too_many or not is_compatible(s_type, d_type):
            params = {"a": src, "b": dst, "type_a": s_type, "type_b": d_type}
            issues.append(Issue("CFG-001", node_id=dst, port=inp, params=params))
            continue
        sources[(dst, inp)] = (*sources.get((dst, inp), ()), src)  # only wires that pass, for the rules
    return links, connected, sources


__all__ = ["WORKFLOW_FORMAT", "Issue", "path_escapes_workspace", "validate"]
