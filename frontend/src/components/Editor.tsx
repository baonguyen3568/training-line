import {
  Background,
  type Connection,
  Controls,
  type Edge,
  type NodeChange,
  ReactFlow,
  ReactFlowProvider,
  useViewport,
  ViewportPortal,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { t, tp } from "../i18n";
import { LANE_WIDTH, LANES } from "../lib/lanes";
import { isCompatible, tokenVar } from "../lib/ports";
import { stepNumbers } from "../lib/topo";
import { graphKey, useWorkflow } from "../store/workflow";
import { CheckBar } from "./CheckBar";
import { Inspector } from "./Inspector";
import { NodeShell, type ShellNode } from "./NodeShell";

const nodeTypes = { vlm: NodeShell };
const VALIDATE_DEBOUNCE_MS = 500;

const LANE_TOP = -4000;
const LANE_HEIGHT = 12000;

function Lanes() {
  // Labels stick to the top edge of the visible canvas and stay ≥12px on screen at any zoom (design review #2).
  const { y, zoom } = useViewport();
  const labelTop = -y / zoom + 8 - LANE_TOP;
  return (
    <ViewportPortal>
      {LANES.map((lane, i) => (
        <div
          key={lane}
          className="lane"
          style={{
            transform: `translate(${i * LANE_WIDTH}px, ${LANE_TOP}px)`,
            width: LANE_WIDTH,
            height: LANE_HEIGHT,
          }}
        >
          <span
            className="lane-label"
            style={{ top: labelTop, fontSize: `${12 / Math.min(zoom, 1)}px` }}
          >
            {t(`canvas.lane.${lane}`)}
          </span>
        </div>
      ))}
    </ViewportPortal>
  );
}

function EditorInner() {
  const { workflow, specs, ports, validation, selectedId } = useWorkflow();
  const { move, select, connect, disconnect, validate } = useWorkflow.getState();
  const [toast, setToast] = useState<string | null>(null);
  const key = graphKey(workflow);

  // Re-validate on semantic changes only (not on drag), debounced (P010 §6).
  useEffect(() => {
    if (!key) return;
    const timer = setTimeout(() => void validate(), VALIDATE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [key, validate]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(timer);
  }, [toast]);

  const nodes: ShellNode[] = useMemo(() => {
    if (!workflow) return [];
    const steps = stepNumbers(workflow);
    return workflow.nodes.map((n) => ({
      id: n.id,
      type: "vlm",
      position: n.ui ?? { x: 0, y: 0 },
      selected: n.id === selectedId,
      data: {
        typeId: n.type,
        spec: specs.get(n.type),
        ports,
        issues: validation?.errors.filter((e) => e.node_id === n.id) ?? [],
        step: steps.get(n.id) ?? 0,
      },
    }));
  }, [workflow, specs, ports, validation, selectedId]);

  const edges: Edge[] = useMemo(() => {
    if (!workflow) return [];
    return workflow.edges.map((e) => {
      const src = workflow.nodes.find((n) => n.id === e.from[0]);
      const type = src ? specs.get(src.type)?.outputs[e.from[1]]?.type : undefined;
      const pt = ports.find((p) => p.id === type);
      return {
        id: `${e.from.join(":")}->${e.to.join(":")}`,
        source: e.from[0],
        sourceHandle: e.from[1],
        target: e.to[0],
        targetHandle: e.to[1],
        style: { stroke: pt ? tokenVar(pt.color_token) : undefined, strokeWidth: 2 },
      };
    });
  }, [workflow, specs, ports]);

  const typeOf = useCallback(
    (nodeId: string, handle: string | null | undefined, dir: "in" | "out") => {
      const n = workflow?.nodes.find((x) => x.id === nodeId);
      const spec = n && specs.get(n.type);
      if (!spec || !handle) return undefined;
      return (dir === "in" ? spec.inputs : spec.outputs)[handle]?.type;
    },
    [workflow, specs],
  );

  const isValidConnection = useCallback(
    (c: Connection | Edge) => {
      const out = typeOf(c.source, c.sourceHandle, "out");
      const inp = typeOf(c.target, c.targetHandle, "in");
      return !!out && !!inp && c.source !== c.target && isCompatible(out, inp, ports);
    },
    [typeOf, ports],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.sourceHandle || !c.targetHandle) return;
      const res = connect([c.source, c.sourceHandle], [c.target, c.targetHandle]);
      if (!res.ok) {
        const label = (type: string) => {
          const pt = ports.find((p) => p.id === type);
          return pt ? tp(pt.label_key.split(".")) : type;
        };
        setToast(
          t("canvas.wire_rejected", {
            target: c.targetHandle,
            expected: label(res.expected),
            actual: label(res.actual),
          }),
        );
      }
    },
    [connect, ports],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<ShellNode>[]) => {
      for (const ch of changes) {
        if (ch.type === "position" && ch.position) move(ch.id, ch.position.x, ch.position.y);
        if (ch.type === "select" && ch.selected) select(ch.id);
      }
    },
    [move, select],
  );

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      for (const e of deleted) if (e.targetHandle) disconnect([e.target, e.targetHandle]);
    },
    [disconnect],
  );

  if (!workflow) return null;
  return (
    <div className="editor">
      <section className="canvas" aria-label={workflow.name}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onConnect={onConnect}
          onEdgesDelete={onEdgesDelete}
          onPaneClick={() => select(null)}
          isValidConnection={isValidConnection}
          deleteKeyCode={["Delete", "Backspace"]}
          defaultViewport={{ x: 16, y: 40, zoom: 1 }}
          minZoom={0.3}
          proOptions={{ hideAttribution: false }}
        >
          <Lanes />
          <Background gap={24} color="var(--color-canvas-grid-dot)" />
          <Controls />
        </ReactFlow>
        {toast && (
          <div className="toast" role="status">
            {toast}
          </div>
        )}
      </section>
      <Inspector />
      <CheckBar />
    </div>
  );
}

export function Editor() {
  return (
    <ReactFlowProvider>
      <EditorInner />
    </ReactFlowProvider>
  );
}
