import type {
  NodeSpec,
  PortSpec,
  ValidationIssue,
  Workflow,
  WorkflowEdge,
  WorkflowNode,
} from "../api/types";
import { SEPARATOR } from "./groups";
import { LANE_WIDTH, LANES, laneLayout, ROW_HEIGHT } from "./lanes";

/** The two fixed nodes inside a group (P4-4 §2); ids with "__" never clash with a step (CFG-007). */
export const BOUNDARY_IN = "__group_in";
export const BOUNDARY_OUT = "__group_out";

/** One level of the workflow as the canvas shows it: the workflow itself, or the inside of an entered group. */
export interface LevelView {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  /** Flat-id prefix of this level's steps ("" at the top, "g__" inside group node g, "g__h__" deeper). */
  prefix: string;
  /** Inside a group: the group step entered last, its spec, and its exposed ports for the two boundary nodes. */
  group?: {
    nodeId: string;
    typeId: string;
    spec: NodeSpec;
    inputs: Record<string, PortSpec>;
    outputs: Record<string, PortSpec>;
  };
  /** Positions: saved ones at the top; inside a group the lane layout (+ the boundary nodes in the first/last lane). */
  positions: Map<string, { x: number; y: number }>;
}

/**
 * The level for `path` (group node ids, outermost first), or null when the path no longer leads to a group (another
 * workflow opened, a node removed): the caller then goes back to the top.
 */
export function resolveLevel(
  workflow: Workflow,
  specs: Map<string, NodeSpec>,
  path: readonly string[],
): LevelView | null {
  let nodes = workflow.nodes;
  let edges = workflow.edges;
  let prefix = "";
  let group: LevelView["group"];
  for (const id of path) {
    const n = nodes.find((x) => x.id === id);
    const spec = n && specs.get(n.type);
    if (!n || !spec?.group) return null;
    nodes = spec.group.body.nodes;
    edges = spec.group.body.edges;
    prefix += `${id}${SEPARATOR}`;
    group = { nodeId: id, typeId: n.type, spec, inputs: spec.inputs, outputs: spec.outputs };
  }
  if (!group) {
    const positions = new Map(nodes.map((n) => [n.id, { x: n.ui?.x ?? 0, y: n.ui?.y ?? 0 }]));
    return { nodes, edges, prefix, positions };
  }
  // inside: the steps in their lanes; "Đầu vào nhóm" first, "Đầu ra nhóm" last, each on its own top row
  const laid = laneLayout(
    nodes.map((n) => ({ id: n.id, category: specs.get(n.type)?.category ?? "data" })),
  );
  const positions = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    const p = laid.get(n.id) ?? { x: 0, y: 0 };
    const edgeLane = p.x < LANE_WIDTH || p.x >= (LANES.length - 1) * LANE_WIDTH;
    positions.set(n.id, n.ui ?? (edgeLane ? { ...p, y: p.y + ROW_HEIGHT } : p));
  }
  positions.set(BOUNDARY_IN, { x: 30, y: 70 });
  positions.set(BOUNDARY_OUT, { x: (LANES.length - 1) * LANE_WIDTH + 30, y: 70 });
  const body = group.spec.group?.body;
  const boundaryEdges: WorkflowEdge[] = [
    ...Object.entries(body?.inputs ?? {}).flatMap(([name, targets]) =>
      targets.map(([node, port]) => ({
        from: [BOUNDARY_IN, name] as [string, string],
        to: [node, port] as [string, string],
      })),
    ),
    ...Object.entries(body?.outputs ?? {}).map(([name, [node, port]]) => ({
      from: [node, port] as [string, string],
      to: [BOUNDARY_OUT, name] as [string, string],
    })),
  ];
  return { nodes, edges: [...edges, ...boundaryEdges], prefix, group, positions };
}

/** The full path of an issue: its top node then the inner ids per level (016b §3). */
const issuePath = (i: ValidationIssue) => (i.node_id ? [i.node_id, ...(i.inner_path ?? [])] : []);

/**
 * The issues of step `nodeId` at the level `path`: its own (the path ends at it) and those inside it when it is a
 * group (longer path, reported with inner_path), so a collapsed group can count them.
 */
export function issuesAt(
  issues: readonly ValidationIssue[],
  path: readonly string[],
  nodeId: string,
): ValidationIssue[] {
  const want = [...path, nodeId];
  return issues
    .filter((i) => {
      const full = issuePath(i);
      return full.length >= want.length && want.every((id, k) => full[k] === id);
    })
    .map((i) => {
      const full = issuePath(i);
      // seen from this level: deeper than the step itself = "inside" it
      const rest = full.slice(want.length);
      return rest.length ? { ...i, inner_path: rest } : { ...i, inner_path: [] };
    });
}

/**
 * Where an issue on a step inside a group lives (P4-4 §3): the groups to enter, the step's flat id, and each level's
 * title for the "Dạy model › Vòng học" trail (titles, never ids). Null for an issue on a top-level step.
 */
export function issueTrail(
  workflow: Workflow,
  specs: Map<string, NodeSpec>,
  issue: ValidationIssue,
  title: (spec: NodeSpec | undefined, typeId: string) => string,
): { groupPath: string[]; stepId: string; labels: string[] } | null {
  const full = issuePath(issue);
  if (full.length < 2) return null;
  const labels: string[] = [];
  let nodes = workflow.nodes;
  for (const [k, id] of full.entries()) {
    const n = nodes.find((x) => x.id === id);
    if (!n) return null;
    const spec = specs.get(n.type);
    labels.push(title(spec, n.type));
    if (k < full.length - 1) nodes = spec?.group?.body.nodes ?? [];
  }
  return { groupPath: full.slice(0, -1), stepId: full.join(SEPARATOR), labels };
}
