import type { Workflow, WorkflowEdge, WorkflowNode } from "../api/types";

// Workflow files are UNTRUSTED (shared by other people, P003 D11): strict shape, unknown fields rejected
// (no install_url/source, no secrets), size-limited, ids as plain identifiers. The server re-validates.

export const MAX_FILE_BYTES = 2 * 1024 * 1024;
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
  ]);
  if (extra) return { ok: false, field: extra };
  if (raw.format !== "vlm-workflow") return { ok: false, field: "format" };
  if (raw.schema_version !== 1) return { ok: false, field: "schema_version" };
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) return { ok: false, field: "nodes" };

  const nodes: WorkflowNode[] = [];
  const ids = new Set<string>();
  for (const n of raw.nodes) {
    if (!isObj(n)) return { ok: false, field: "nodes" };
    const bad = onlyKeys(n, ["id", "type", "version", "params", "ui", "package"]); // package: registry lookup key only
    if (bad) return { ok: false, field: `nodes.${bad}` };
    if (typeof n.id !== "string" || !ID.test(n.id) || ids.has(n.id))
      return { ok: false, field: "nodes.id" };
    if (typeof n.type !== "string" || !TYPE.test(n.type)) return { ok: false, field: "nodes.type" };
    if (typeof n.version !== "string" || n.version.length > 32)
      return { ok: false, field: "nodes.version" };
    if (n.params !== undefined && !isObj(n.params)) return { ok: false, field: "nodes.params" };
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
  for (const e of raw.edges) {
    if (!isObj(e) || onlyKeys(e, ["from", "to", "ui"]) || !isEnd(e.from) || !isEnd(e.to))
      return { ok: false, field: "edges" };
    edges.push({ from: e.from, to: e.to });
  }
  const name = typeof raw.name === "string" ? raw.name.slice(0, 120) : "untitled";
  return { ok: true, workflow: { format: "vlm-workflow", schema_version: 1, name, nodes, edges } };
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
    },
    null,
    2,
  )}\n`;
}
