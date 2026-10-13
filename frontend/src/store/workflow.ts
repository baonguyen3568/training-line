import { create } from "zustand";
import { api } from "../api";
import type {
  EventError,
  NodeRunStatus,
  NodeSpec,
  PortType,
  ValidationResult,
  Workflow,
  WorkflowEdge,
  WorkflowNode,
} from "../api/types";
import { getLocale } from "../i18n";
import { laneLayout } from "../lib/lanes";
import { isCompatible } from "../lib/ports";
import { selfAndDownstream } from "../lib/topo";

export type ConnectResult = { ok: true } | { ok: false; expected: string; actual: string };

export interface NodeRunState {
  status: NodeRunStatus;
  errorCode?: string;
  progress?: number;
  message?: string;
  /** Run `warning` events for this node (e.g. CACHE-001: the saved result is unreadable, the step runs again). */
  warnings?: EventError[];
  /** G-4: run-dir base "<nid>-<key16>" from node_status, what "Đặt lại bước" acts on. */
  nodeBase?: string;
}

interface WorkflowState {
  specs: Map<string, NodeSpec>;
  ports: PortType[];
  workflow: Workflow | null;
  selectedId: string | null;
  validation: ValidationResult | null;
  validating: boolean;
  past: Workflow[];
  future: Workflow[];
  /** Bumped whenever another workflow is opened or closed (node ids like n1 repeat across workflows). */
  opened: number;
  /** Live per-node state of the current run (node_status events). */
  run: {
    id: string;
    nodes: Record<string, NodeRunState>;
    done: boolean;
    status?: string;
    startedAt: number;
    endedAt?: number;
  } | null;
  loadCatalog(): Promise<void>;
  openTemplate(id: string): Promise<void>;
  newWorkflow(): void;
  openWorkflow(wf: Workflow): void;
  closeWorkflow(): void;
  select(id: string | null): void;
  move(id: string, x: number, y: number): void;
  /** Call once when a drag ends so the whole drag is one undo step. */
  commitMove(): void;
  setParam(id: string, name: string, value: unknown): void;
  addNode(typeId: string, at: { x: number; y: number }): string | null;
  removeNodes(ids: string[]): void;
  duplicateNode(id: string): string | null;
  connect(from: [string, string], to: [string, string]): ConnectResult;
  disconnect(to: [string, string]): void;
  undo(): void;
  redo(): void;
  validate(): Promise<void>;
  startRun(): Promise<void>;
  cancelRun(): Promise<void>;
  stopWatching(): void;
}

const HISTORY = 50;

/** Semantic identity of a graph: everything except `ui` (positions), mirroring the backend cache key. */
export function graphKey(wf: Workflow | null): string {
  if (!wf) return "";
  return JSON.stringify([
    wf.nodes.map((n) => [n.id, n.type, n.version, n.params]),
    wf.edges.map((e) => [e.from, e.to]),
  ]);
}

const sameEnd = (a: [string, string], b: [string, string]) => a[0] === b[0] && a[1] === b[1];

function defaultParams(spec: NodeSpec): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [name, s] of Object.entries(spec.params_schema.properties)) {
    if (s.default !== undefined) out[name] = structuredClone(s.default);
  }
  return out;
}

function nextId(wf: Workflow): string {
  const used = new Set(wf.nodes.map((n) => n.id));
  let i = wf.nodes.length + 1;
  while (used.has(`n${i}`)) i++;
  return `n${i}`;
}

export const emptyWorkflow = (): Workflow => ({
  format: "vlm-workflow",
  schema_version: 1,
  name: "untitled",
  nodes: [],
  edges: [],
});

let unsubscribe: (() => void) | null = null;
let dragStart: Workflow | null = null;

export const useWorkflow = create<WorkflowState>((set, get) => {
  /** Apply a graph edit as one undo step. */
  const edit = (next: Workflow) => {
    const cur = get().workflow;
    if (!cur) return;
    set({ workflow: next, past: [...get().past, cur].slice(-HISTORY), future: [] });
  };

  return {
    specs: new Map(),
    ports: [],
    workflow: null,
    selectedId: null,
    validation: null,
    validating: false,
    past: [],
    future: [],
    opened: 0,
    run: null,

    async loadCatalog() {
      const res = await api.nodeTypes(getLocale());
      set({ specs: new Map(res.node_types.map((s) => [s.type_id, s])), ports: res.port_types });
    },

    async openTemplate(id) {
      const tpl = await api.template(id);
      // Opening a template always works on a copy (P010 §8).
      get().openWorkflow(structuredClone(tpl.workflow));
      await get().validate();
    },

    newWorkflow() {
      get().openWorkflow(emptyWorkflow());
    },

    openWorkflow(wf) {
      // Positions come from lanes when absent (templates, hand-written files).
      const { specs } = get();
      const layout = laneLayout(
        wf.nodes.map((n) => ({ id: n.id, category: specs.get(n.type)?.category ?? "util" })),
      );
      for (const n of wf.nodes) n.ui ??= layout.get(n.id) ?? { x: 0, y: 0 };
      get().stopWatching();
      set({
        workflow: wf,
        selectedId: null,
        validation: null,
        past: [],
        future: [],
        run: null,
        opened: get().opened + 1,
      });
    },

    closeWorkflow() {
      get().stopWatching();
      set({
        workflow: null,
        selectedId: null,
        validation: null,
        past: [],
        future: [],
        run: null,
        opened: get().opened + 1,
      });
    },

    select(id) {
      set({ selectedId: id });
    },

    move(id, x, y) {
      const wf = get().workflow;
      if (!wf) return;
      dragStart ??= wf;
      // Position lives in `ui` and is excluded from the backend cache key (P009 §3).
      set({
        workflow: { ...wf, nodes: wf.nodes.map((n) => (n.id === id ? { ...n, ui: { x, y } } : n)) },
      });
    },

    commitMove() {
      if (!dragStart) return;
      set({ past: [...get().past, dragStart].slice(-HISTORY), future: [] });
      dragStart = null;
    },

    setParam(id, name, value) {
      const wf = get().workflow;
      if (!wf) return;
      edit({
        ...wf,
        nodes: wf.nodes.map((n) =>
          n.id === id ? { ...n, params: { ...n.params, [name]: value } } : n,
        ),
      });
    },

    addNode(typeId, at) {
      const wf = get().workflow;
      const spec = get().specs.get(typeId);
      if (!wf || !spec) return null;
      const id = nextId(wf);
      const node: WorkflowNode = {
        id,
        type: typeId,
        version: spec.version,
        params: defaultParams(spec),
        ui: at,
      };
      edit({ ...wf, nodes: [...wf.nodes, node] });
      set({ selectedId: id });
      return id;
    },

    removeNodes(ids) {
      const wf = get().workflow;
      if (!wf || ids.length === 0) return;
      const gone = new Set(ids);
      edit({
        ...wf,
        nodes: wf.nodes.filter((n) => !gone.has(n.id)),
        edges: wf.edges.filter((e) => !gone.has(e.from[0]) && !gone.has(e.to[0])),
      });
      if (get().selectedId && gone.has(get().selectedId ?? "")) set({ selectedId: null });
    },

    duplicateNode(id) {
      const wf = get().workflow;
      const src = wf?.nodes.find((n) => n.id === id);
      if (!wf || !src) return null;
      const copyId = nextId(wf);
      const pos = src.ui ?? { x: 0, y: 0 };
      // Params copied, wires not (a duplicate starts unconnected, like ComfyUI's Ctrl+C/V).
      const copy: WorkflowNode = {
        ...structuredClone(src),
        id: copyId,
        ui: { x: pos.x + 30, y: pos.y + 40 },
      };
      edit({ ...wf, nodes: [...wf.nodes, copy] });
      set({ selectedId: copyId });
      return copyId;
    },

    connect(from, to) {
      const { workflow: wf, specs, ports } = get();
      if (!wf) return { ok: false, expected: "", actual: "" };
      const src = wf.nodes.find((n) => n.id === from[0]);
      const dst = wf.nodes.find((n) => n.id === to[0]);
      const actual = (src && specs.get(src.type)?.outputs[from[1]]?.type) ?? "";
      const expected = (dst && specs.get(dst.type)?.inputs[to[1]]?.type) ?? "";
      // Reject type mismatches and cycles (feeding a node from itself or from its own downstream).
      if (
        !actual ||
        !expected ||
        selfAndDownstream(wf, to[0]).has(from[0]) ||
        !isCompatible(actual, expected, ports)
      ) {
        return { ok: false, expected, actual };
      }
      // One wire per input port (inputs are not `many` in v1): replace any existing wire.
      const edges: WorkflowEdge[] = [...wf.edges.filter((e) => !sameEnd(e.to, to)), { from, to }];
      edit({ ...wf, edges });
      return { ok: true };
    },

    disconnect(to) {
      const wf = get().workflow;
      if (!wf) return;
      edit({ ...wf, edges: wf.edges.filter((e) => !sameEnd(e.to, to)) });
    },

    undo() {
      const { past, workflow, future } = get();
      const prev = past.at(-1);
      if (!prev || !workflow) return;
      set({ workflow: prev, past: past.slice(0, -1), future: [workflow, ...future] });
    },

    redo() {
      const { past, workflow, future } = get();
      const next = future[0];
      if (!next || !workflow) return;
      set({ workflow: next, past: [...past, workflow], future: future.slice(1) });
    },

    async validate() {
      const wf = get().workflow;
      if (!wf) return;
      const key = graphKey(wf);
      set({ validating: true });
      try {
        const validation = await api.validate(wf, getLocale());
        // Ignore stale responses if the graph changed meanwhile (moving nodes does not count).
        if (graphKey(get().workflow) === key) set({ validation });
      } finally {
        set({ validating: false });
      }
    },

    async startRun() {
      const wf = get().workflow;
      if (!wf) return;
      get().stopWatching();
      const { run_id } = await api.startRun(wf);
      const nodes: Record<string, NodeRunState> = {};
      for (const n of wf.nodes) nodes[n.id] = { status: "pending" };
      set({ run: { id: run_id, nodes, done: false, startedAt: Date.now() } });
      unsubscribe = api.watchRun(run_id, (ev) => {
        const run = get().run;
        if (!run || run.id !== run_id) return;
        if (ev.type === "node_status") {
          const st: NodeRunState = { status: ev.status };
          if (ev.progress !== undefined) st.progress = ev.progress;
          if (ev.error) st.errorCode = ev.error.code;
          const prev = run.nodes[ev.node_id];
          if (prev?.warnings) st.warnings = prev.warnings;
          const base = ev.node_base ?? prev?.nodeBase; // later events without a key keep the known base
          if (base) st.nodeBase = base;
          set({ run: { ...run, nodes: { ...run.nodes, [ev.node_id]: st } } });
        } else if (ev.type === "warning" && ev.node_id && ev.error) {
          const prev = run.nodes[ev.node_id] ?? { status: "pending" as const };
          const warnings = [...(prev.warnings ?? []), ev.error];
          set({ run: { ...run, nodes: { ...run.nodes, [ev.node_id]: { ...prev, warnings } } } });
        } else if (ev.type === "run_status") {
          if (ev.status === "running") return;
          // Terminal (completed/failed/cancelled/interrupted/invalid): nodes never started stay as they are.
          set({ run: { ...run, done: true, status: ev.status, endedAt: Date.now() } });
        } else if (ev.type === "run_not_found" || ev.type === "connection_lost") {
          set({ run: { ...run, done: true, status: ev.type, endedAt: Date.now() } });
        }
      });
    },

    async cancelRun() {
      const run = get().run;
      if (!run || run.done) return;
      await api.cancelRun(run.id);
    },

    stopWatching() {
      unsubscribe?.();
      unsubscribe = null;
    },
  };
});
