import { create } from "zustand";
import { api } from "../api";
import type { NodeSpec, PortType, ValidationResult, Workflow, WorkflowEdge } from "../api/types";
import { getLocale } from "../i18n";
import { laneLayout } from "../lib/lanes";
import { isCompatible } from "../lib/ports";

export type ConnectResult = { ok: true } | { ok: false; expected: string; actual: string };

interface WorkflowState {
  specs: Map<string, NodeSpec>;
  ports: PortType[];
  workflow: Workflow | null;
  selectedId: string | null;
  validation: ValidationResult | null;
  validating: boolean;
  loadCatalog(): Promise<void>;
  openTemplate(id: string): Promise<void>;
  closeWorkflow(): void;
  select(id: string | null): void;
  move(id: string, x: number, y: number): void;
  setParam(id: string, name: string, value: unknown): void;
  connect(from: [string, string], to: [string, string]): ConnectResult;
  disconnect(to: [string, string]): void;
  validate(): Promise<void>;
}

/** Semantic identity of a graph: everything except `ui` (positions), mirroring the backend cache key. */
export function graphKey(wf: Workflow | null): string {
  if (!wf) return "";
  return JSON.stringify([
    wf.nodes.map((n) => [n.id, n.type, n.version, n.params]),
    wf.edges.map((e) => [e.from, e.to]),
  ]);
}

const sameEnd = (a: [string, string], b: [string, string]) => a[0] === b[0] && a[1] === b[1];

export const useWorkflow = create<WorkflowState>((set, get) => ({
  specs: new Map(),
  ports: [],
  workflow: null,
  selectedId: null,
  validation: null,
  validating: false,

  async loadCatalog() {
    const res = await api.nodeTypes(getLocale());
    set({ specs: new Map(res.node_types.map((s) => [s.type_id, s])), ports: res.port_types });
  },

  async openTemplate(id) {
    const tpl = await api.template(id);
    // Opening a template always works on a copy (P010 §8); positions come from lanes when absent.
    const wf = structuredClone(tpl.workflow);
    const { specs } = get();
    const layout = laneLayout(
      wf.nodes.map((n) => ({ id: n.id, category: specs.get(n.type)?.category ?? "util" })),
    );
    for (const n of wf.nodes) n.ui ??= layout.get(n.id) ?? { x: 0, y: 0 };
    set({ workflow: wf, selectedId: null, validation: null });
    await get().validate();
  },

  closeWorkflow() {
    set({ workflow: null, selectedId: null, validation: null });
  },

  select(id) {
    set({ selectedId: id });
  },

  move(id, x, y) {
    const wf = get().workflow;
    if (!wf) return;
    // Position lives in `ui` and is excluded from the backend cache key (P009 §3).
    set({ workflow: { ...wf, nodes: wf.nodes.map((n) => (n.id === id ? { ...n, ui: { x, y } } : n)) } });
  },

  setParam(id, name, value) {
    const wf = get().workflow;
    if (!wf) return;
    set({
      workflow: {
        ...wf,
        nodes: wf.nodes.map((n) => (n.id === id ? { ...n, params: { ...n.params, [name]: value } } : n)),
      },
    });
  },

  connect(from, to) {
    const { workflow: wf, specs, ports } = get();
    if (!wf) return { ok: false, expected: "", actual: "" };
    const src = wf.nodes.find((n) => n.id === from[0]);
    const dst = wf.nodes.find((n) => n.id === to[0]);
    const actual = (src && specs.get(src.type)?.outputs[from[1]]?.type) ?? "";
    const expected = (dst && specs.get(dst.type)?.inputs[to[1]]?.type) ?? "";
    if (!actual || !expected || !isCompatible(actual, expected, ports)) {
      return { ok: false, expected, actual };
    }
    // One wire per input port (inputs are not `many` in v1): replace any existing wire.
    const edges: WorkflowEdge[] = [...wf.edges.filter((e) => !sameEnd(e.to, to)), { from, to }];
    set({ workflow: { ...wf, edges } });
    return { ok: true };
  },

  disconnect(to) {
    const wf = get().workflow;
    if (!wf) return;
    set({ workflow: { ...wf, edges: wf.edges.filter((e) => !sameEnd(e.to, to)) } });
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
}));
