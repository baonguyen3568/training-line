import type { GroupBody, GroupDefinition, NodeSpec, ParamSchema, PortSpec } from "../api/types";
import type { NodeRunState } from "../store/workflow";

export type { GroupBody };

/**
 * P016 groups (subgraphs) for the UI (contract .team/proposals/016b): a group is DATA expanded by the server; the UI
 * shows it as one step (its exposed ports and params) and lets people look inside, read-only in 1.0 (design 818e7c3).
 * Inner steps are addressed by flat ids `<group>__<inner>` (recursive), exactly as the server's run events.
 */
export const SEPARATOR = "__";
export const GROUP_ICON = "layers";

/** A first-party group entry of GET /api/node-types `groups[]` (016b §2). */
export interface GroupTypeEntry extends Omit<NodeSpec, "type_id" | "icon"> {
  group_id: string;
  title_key?: string;
  desc_key?: string;
  icon?: string;
  body: GroupBody;
}

export const isGroupType = (typeId: string) => /^(group|user)\./.test(typeId);
export const flatId = (path: readonly string[]) => path.join(SEPARATOR);

/** Server group entry → the spec map's shape (the alternative `node_types[]` entry with `group` is taken as is). */
export function groupEntryToSpec(entry: GroupTypeEntry): NodeSpec {
  const { group_id, body, title_key: _t, desc_key: _d, ...rest } = entry;
  return { ...rest, type_id: group_id, icon: GROUP_ICON, group: { body, readOnly: true } };
}

/** Accept both server shapes (016b §1): `groups[]` beside node types, or group entries inside `node_types[]`. */
export function specsFromCatalog(
  nodeTypes: NodeSpec[],
  groups: GroupTypeEntry[] | undefined,
): Map<string, NodeSpec> {
  const specs = new Map<string, NodeSpec>();
  for (const s of nodeTypes) {
    // the alternative shape: a node_types entry whose `group` is the bare body (+ read_only), not { body, readOnly }
    const raw = (s as unknown as { group?: unknown }).group;
    if (raw && typeof raw === "object" && !("body" in raw)) {
      const { read_only: _r, ...body } = raw as GroupBody & { read_only?: boolean };
      specs.set(s.type_id, { ...s, icon: GROUP_ICON, group: { body, readOnly: true } });
    } else specs.set(s.type_id, s);
  }
  for (const g of groups ?? []) specs.set(g.group_id, groupEntryToSpec(g));
  return specs;
}

const TRUST_ORDER = ["first_party", "third_party"] as const;

/**
 * A `user.*` group from the workflow file, derived with the server's rules (016b §2, checked against cleancode's
 * golden fixtures): ports/params from the exposed inner ones, category = lane of the node behind the first exposed
 * output, permissions = union, lowest trust. Null when an inner type is unknown (the server reports why on validate).
 */
export function deriveGroupSpec(
  def: GroupDefinition,
  specs: Map<string, NodeSpec>,
): NodeSpec | null {
  const inner = new Map(def.nodes.map((n) => [n.id, n]));
  const specOf = (id: string) => {
    const n = inner.get(id);
    return n ? specs.get(n.type) : undefined;
  };
  const innerSpecs = def.nodes.map((n) => specs.get(n.type));
  if (innerSpecs.some((s) => !s)) return null;
  const known = innerSpecs as NodeSpec[];

  const inputs: Record<string, PortSpec> = {};
  for (const [name, targets] of Object.entries(def.inputs)) {
    const ports = targets.map(([id, port]) => specOf(id)?.inputs[port]);
    if (ports.some((p) => !p)) return null;
    const first = ports[0] as PortSpec;
    if (ports.some((p) => p?.type !== first.type)) return null; // the server refuses mixed types too
    inputs[name] = {
      type: first.type,
      ...(ports.every((p) => p?.optional) ? { optional: true } : {}),
      ...(first.many ? { many: true } : {}),
    };
  }
  const outputs: Record<string, PortSpec> = {};
  for (const [name, [id, port]] of Object.entries(def.outputs)) {
    const p = specOf(id)?.outputs[port];
    if (!p) return null;
    outputs[name] = { type: p.type };
  }
  const properties: Record<string, ParamSchema> = {};
  for (const [name, [id, param]] of Object.entries(def.params)) {
    const prop = specOf(id)?.params_schema.properties[param];
    if (!prop) return null;
    const preset = inner.get(id)?.params[param];
    properties[name] = preset === undefined ? prop : { ...prop, default: preset };
  }
  const firstOut = Object.values(def.outputs)[0];
  const lead = (firstOut && specOf(firstOut[0])) ?? known[0];
  const permissions = [
    ...new Map(known.flatMap((s) => s.permissions).map((p) => [p.id, p])).values(),
  ];
  const trust = known
    .map((s) => s.origin.kind)
    .reduce(
      (lo, k) => (TRUST_ORDER.indexOf(k) > TRUST_ORDER.indexOf(lo) ? k : lo),
      "first_party" as const,
    );
  return {
    type_id: def.id,
    title: def.title, // plain text from the file (untrusted): rendered as text, never as a key
    version: def.version,
    category: (lead as NodeSpec).category,
    icon: GROUP_ICON,
    inputs,
    outputs,
    params_schema: { type: "object", properties },
    cacheable: known.every((s) => s.cacheable),
    runner: known.some((s) => s.runner === "container") ? "container" : "host",
    origin: { kind: trust },
    permissions,
    policy_block: known.find((s) => s.policy_block)?.policy_block ?? null,
    group: {
      body: {
        nodes: def.nodes,
        edges: def.edges,
        inputs: def.inputs,
        outputs: def.outputs,
        params: def.params,
      },
      readOnly: true, // 1.0: no editing inside any group (design 818e7c3)
      user: true,
    },
  };
}

/** Base catalog + the open workflow's own `user.*` groups (inner groups may use each other: resolved in order). */
export function withUserGroups(base: Map<string, NodeSpec>, groups: GroupDefinition[] | undefined) {
  if (!groups?.length) return base;
  const specs = new Map(base);
  let pending = groups.filter((g) => g.id.startsWith("user."));
  for (let pass = 0; pass < 4 && pending.length; pass++) {
    const next: GroupDefinition[] = [];
    for (const g of pending) {
      const spec = deriveGroupSpec(g, specs);
      if (spec) specs.set(g.id, spec);
      else next.push(g);
    }
    if (next.length === pending.length) break; // unknown inner types: the server's CFG-012 says why
    pending = next;
  }
  return specs;
}

// Aggregate status order (P4-4 §1): the first that matches wins.
type Folded = NodeRunState["status"] | "waiting_memory";

/**
 * The status a collapsed group shows, folded from its inner steps' flat ids (any depth): failed > interrupted… as in
 * the spec; `done_count/step_count` for the "x/y bước" progress. Undefined until an inner step reported anything.
 */
export function foldGroupState(
  runNodes: Record<string, NodeRunState> | undefined,
  groupId: string,
  stepCount: number,
): { state: NodeRunState; doneCount: number; waiting: boolean } | undefined {
  if (!runNodes) return undefined;
  const prefix = `${groupId}${SEPARATOR}`;
  const inner = Object.entries(runNodes)
    .filter(([id]) => id.startsWith(prefix))
    .map(([, s]) => s);
  if (inner.length === 0) return undefined;
  const has = (st: Folded) =>
    inner.some((s) => (st === "waiting_memory" ? !!s.waiting : s.status === st));
  const doneCount = inner.filter((s) => s.status === "completed" || s.status === "cached").length;
  const all = (pred: (s: NodeRunState) => boolean) =>
    inner.length >= stepCount && inner.every(pred);
  let status: NodeRunState["status"] = "pending";
  if (has("failed")) status = "failed";
  else if (has("waiting_memory") || has("running")) status = "running";
  else if (has("interrupted")) status = "interrupted";
  else if (all((s) => s.status === "skipped")) status = "skipped";
  else if (all((s) => s.status === "completed" || s.status === "cached"))
    status = inner.some((s) => s.status === "completed") ? "completed" : "cached";
  else if (has("completed") || has("cached")) status = "running"; // part done, rest pending
  const failed = inner.find((s) => s.status === "failed");
  const waiting = inner.find((s) => s.waiting);
  return {
    state: {
      status,
      ...(failed?.error ? { error: failed.error, errorCode: failed.error.code } : {}),
      ...(waiting?.waiting && status === "running" ? { waiting: waiting.waiting } : {}),
    },
    doneCount,
    waiting: !!waiting,
  };
}

/** Every step inside a group, any depth (for "x/y bước"): groups count their own inner steps, not themselves. */
export function innerStepCount(
  spec: NodeSpec | undefined,
  specs: Map<string, NodeSpec>,
  depth = 0,
): number {
  if (!spec?.group || depth > 3) return 0;
  return spec.group.body.nodes.reduce((sum, n) => {
    const s = specs.get(n.type);
    return sum + (s?.group ? innerStepCount(s, specs, depth + 1) : 1);
  }, 0);
}

/**
 * The inner step that owns an exposed param or port of a group, followed down nested groups, so its copy
 * (node.<type>.param/port.<name>.*) labels the group's field; an ordinary step owns its own.
 */
export function copyOwner(
  typeId: string,
  name: string,
  kind: "param" | "input" | "output",
  specs: Map<string, NodeSpec>,
): { typeId: string; name: string } {
  let cur = { typeId, name };
  for (let depth = 0; depth < 4; depth++) {
    const body = specs.get(cur.typeId)?.group?.body;
    if (!body) return cur;
    const target =
      kind === "param"
        ? body.params[cur.name]
        : kind === "input"
          ? body.inputs[cur.name]?.[0]
          : body.outputs[cur.name];
    const inner = target && body.nodes.find((n) => n.id === target[0]);
    if (!target || !inner) return cur;
    cur = { typeId: inner.type, name: target[1] };
  }
  return cur;
}
