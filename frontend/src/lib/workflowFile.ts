import type { GroupDefinition, Workflow, WorkflowEdge, WorkflowNode } from "../api/types";

// Workflow files are UNTRUSTED (shared by other people, P003 D11): strict shape, unknown fields rejected
// (no install_url/source, no secrets), size-limited, ids as plain identifiers. The server re-validates.

export const MAX_FILE_BYTES = 2 * 1024 * 1024;
// P016: at most 50 group definitions, checked with the same strict key sets as the top level (no install sources
// hide inside a group). Their text (title, description) stays untrusted: rendered only as plain text (untrustedText)
// once the subgraph UI shows it; the backend re-validates the whole graph.
const MAX_GROUPS = 50;
// A group from a file is the user's: `user.*` only (server pattern). First-party `group.*` never comes from a file,
// and there is no title_key, so a file cannot borrow app copy to look official (security).
const USER_GROUP = /^user\.[a-z0-9][a-z0-9_-]{0,47}$/;
const ID = /^[A-Za-z0-9_-]{1,64}$/;
const TYPE = /^[a-z0-9][a-z0-9_./-]{0,127}$/;

export type ParseResult = { ok: true; workflow: Workflow } | { ok: false; field: string };

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const onlyKeys = (o: Record<string, unknown>, allowed: string[]) =>
  Object.keys(o).find((k) => !allowed.includes(k));
const isEnd = (v: unknown): v is [string, string] =>
  Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === "string" && ID.test(x));

export function parseWorkflowFile(text: string): ParseResult {
  if (text.length > MAX_FILE_BYTES) return { ok: false, field: "size" };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ok: false, field: "json" };
  }
  if (!isObj(raw)) return { ok: false, field: "root" };
  // Same key sets as backend graph/validation.py (_TOP_KEYS/_NODE_KEYS/_EDGE_KEYS) until schemas/ has a workflow schema.
  const extra = onlyKeys(raw, [
    "format",
    "schema_version",
    "name",
    "description",
    "nodes",
    "edges",
    "requires",
    "ui",
    "groups", // P016 subgraphs: shape checked here, contents by the backend (validate on open)
  ]);
  if (extra) return { ok: false, field: extra };
  if (raw.format !== "vlm-workflow") return { ok: false, field: "format" };
  if (raw.schema_version !== 1) return { ok: false, field: "schema_version" };
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) return { ok: false, field: "nodes" };

  const parsed = parseGraph(raw.nodes, raw.edges, "");
  if ("field" in parsed) return { ok: false, field: parsed.field };
  const { nodes, edges } = parsed;
  let groups: GroupDefinition[] | undefined;
  if (raw.groups !== undefined) {
    if (!Array.isArray(raw.groups) || raw.groups.length > MAX_GROUPS)
      return { ok: false, field: "groups" };
    groups = [];
    for (const g of raw.groups) {
      const res = parseGroup(g);
      if ("field" in res) return { ok: false, field: res.field };
      groups.push(res.group);
    }
  }
  const name = typeof raw.name === "string" ? raw.name.slice(0, 120) : "untitled";
  return {
    ok: true,
    workflow: {
      format: "vlm-workflow",
      schema_version: 1,
      name,
      nodes,
      edges,
      ...(groups ? { groups } : {}),
    },
  };
}

type GraphResult = { nodes: WorkflowNode[]; edges: WorkflowEdge[] } | { field: string };

/** Nodes + edges with the strict per-item key sets; `at` prefixes the failing field ("groups.nodes.id"). */
function parseGraph(rawNodes: unknown, rawEdges: unknown, at: string): GraphResult {
  if (!Array.isArray(rawNodes) || !Array.isArray(rawEdges)) return { field: `${at}nodes` };
  const nodes: WorkflowNode[] = [];
  const ids = new Set<string>();
  for (const n of rawNodes) {
    if (!isObj(n)) return { field: `${at}nodes` };
    const bad = onlyKeys(n, ["id", "type", "version", "params", "ui", "package"]); // package: registry lookup key only
    if (bad) return { field: `${at}nodes.${bad}` };
    if (typeof n.id !== "string" || !ID.test(n.id) || ids.has(n.id))
      return { field: `${at}nodes.id` };
    if (typeof n.type !== "string" || !TYPE.test(n.type)) return { field: `${at}nodes.type` };
    if (typeof n.version !== "string" || n.version.length > 32)
      return { field: `${at}nodes.version` };
    if (n.params !== undefined && !isObj(n.params)) return { field: `${at}nodes.params` };
    ids.add(n.id);
    const node: WorkflowNode = {
      id: n.id,
      type: n.type,
      version: n.version,
      params: { ...(n.params ?? {}) },
    };
    const ui = n.ui;
    if (
      isObj(ui) &&
      typeof ui.x === "number" &&
      typeof ui.y === "number" &&
      Number.isFinite(ui.x + ui.y)
    ) {
      node.ui = { x: ui.x, y: ui.y };
    }
    nodes.push(node);
  }
  const edges: WorkflowEdge[] = [];
  for (const e of rawEdges) {
    if (!isObj(e) || onlyKeys(e, ["from", "to", "ui"]) || !isEnd(e.from) || !isEnd(e.to))
      return { field: `${at}edges` };
    edges.push({ from: e.from, to: e.to });
  }
  return { nodes, edges };
}

// Server caps (P016, eb5c017): ≤ 2000 edges per group, ≤ 100 names per inputs/outputs/params, ≤ 500 input targets.
const MAX_GROUP_EDGES = 2000;
const MAX_GROUP_NAMES = 100;
const MAX_INPUT_TARGETS = 500;

const isEndMap = (v: unknown, many: boolean): boolean =>
  isObj(v) &&
  Object.keys(v).length <= MAX_GROUP_NAMES &&
  Object.entries(v).every(
    ([k, x]) =>
      ID.test(k) && (many ? Array.isArray(x) && x.length <= 64 && x.every(isEnd) : isEnd(x)),
  );

/** One P016 group definition: the same strict key sets as the top level, recursively for its nodes/edges. */
function parseGroup(g: unknown): { group: GroupDefinition } | { field: string } {
  if (!isObj(g)) return { field: "groups" };
  const bad = onlyKeys(g, [
    "id",
    "version",
    "title",
    "description",
    "nodes",
    "edges",
    "inputs",
    "outputs",
    "params",
  ]);
  if (bad) return { field: `groups.${bad}` };
  if (typeof g.id !== "string" || !USER_GROUP.test(g.id)) return { field: "groups.id" };
  if (typeof g.version !== "string" || g.version.length > 32) return { field: "groups.version" };
  if (typeof g.title !== "string" || g.title.length > 120) return { field: "groups.title" };
  if (
    g.description !== undefined &&
    (typeof g.description !== "string" || g.description.length > 2000)
  )
    return { field: "groups.description" };
  const graph = parseGraph(g.nodes, g.edges ?? [], "groups."); // edges optional, as on the server
  if ("field" in graph) return graph;
  if (Array.isArray(g.edges) && g.edges.length > MAX_GROUP_EDGES) return { field: "groups.edges" };
  if (
    !isEndMap(g.inputs, true) ||
    Object.values(g.inputs as Record<string, unknown[]>).flat().length > MAX_INPUT_TARGETS
  )
    return { field: "groups.inputs" };
  if (!isEndMap(g.outputs, false)) return { field: "groups.outputs" };
  if (!isEndMap(g.params, false)) return { field: "groups.params" };
  return {
    group: {
      id: g.id,
      version: g.version,
      title: g.title,
      ...(g.description !== undefined ? { description: g.description as string } : {}),
      nodes: graph.nodes,
      edges: graph.edges,
      inputs: g.inputs as GroupDefinition["inputs"],
      outputs: g.outputs as GroupDefinition["outputs"],
      params: g.params as GroupDefinition["params"],
    },
  };
}

/** Serialize for download; `ui` positions included, nothing else beyond the documented shape. */
export function serializeWorkflow(wf: Workflow): string {
  return `${JSON.stringify(
    {
      format: wf.format,
      schema_version: wf.schema_version,
      name: wf.name,
      nodes: wf.nodes.map(({ id, type, version, params, ui }) => ({
        id,
        type,
        version,
        params,
        ...(ui ? { ui } : {}),
      })),
      edges: wf.edges.map(({ from, to }) => ({ from, to })),
      ...(wf.groups?.length ? { groups: wf.groups } : {}),
    },
    null,
    2,
  )}\n`;
}
