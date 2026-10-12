import {
  Background,
  type Connection,
  Controls,
  type Edge,
  type NodeChange,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useViewport,
  ViewportPortal,
} from "@xyflow/react";
import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { t, tp } from "../i18n";
import { LANE_WIDTH, LANES, laneOf, ROW_HEIGHT } from "../lib/lanes";
import { nodeTitle } from "../lib/nodeText";
import { isCompatible, tokenVar } from "../lib/ports";
import { stepNumbers } from "../lib/topo";
import { graphKey, useWorkflow } from "../store/workflow";
import { CheckBar } from "./CheckBar";
import { Inspector } from "./Inspector";
import { DRAG_MIME, NodeLibrary } from "./NodeLibrary";
import { NodeShell, type ShellNode } from "./NodeShell";
import { Toolbar } from "./Toolbar";

const nodeTypes = { vlm: NodeShell };
const VALIDATE_DEBOUNCE_MS = 500;

const LANE_TOP = -4000;
const LANE_HEIGHT = 12000;

function Lanes() {
  // Labels stick to the top edge of the visible canvas and stay ≥12px on screen at any zoom (design review #2).
  const { y, zoom } = useViewport();
  const labelTop = (-y + 48) / zoom - LANE_TOP; // below the canvas toolbar
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
  const { workflow, specs, ports, validation, selectedId, run } = useWorkflow();
  const {
    move,
    commitMove,
    select,
    connect,
    disconnect,
    validate,
    addNode,
    removeNodes,
    duplicateNode,
    undo,
    redo,
  } = useWorkflow.getState();
  const { screenToFlowPosition } = useReactFlow();
  const canvasRef = useRef<HTMLElement>(null);
  const [toast, setToast] = useState<string | null>(null);
  const key = graphKey(workflow);
  // While a run is live the graph structure is locked (design spec M1-5 §7).
  const locked = !!run && !run.done;

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

  /** Place a new node in its own lane, below the nodes already there (drop position only picks the row). */
  const placeInLane = useCallback(
    (typeId: string, at?: { x: number; y: number }) => {
      const spec = specs.get(typeId);
      const live = useWorkflow.getState().workflow; // not the render-time snapshot: rapid adds must stack
      if (!spec || !live) return;
      const lane = laneOf(spec.category);
      const laneX = LANES.indexOf(lane) * LANE_WIDTH + 10;
      const inLane = live.nodes.filter((n) => {
        const c = specs.get(n.type)?.category;
        return c && laneOf(c) === lane;
      });
      const y = at ? at.y : 70 + inLane.length * ROW_HEIGHT;
      const id = addNode(typeId, { x: laneX, y });
      if (at && Math.abs(at.x - laneX) > LANE_WIDTH) {
        setToast(
          t("library.snapped_to_lane", {
            node: nodeTitle(spec, typeId),
            lane: t(`canvas.lane.${lane}`),
          }),
        );
      }
      return id;
    },
    [specs, addNode],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      const typeId = e.dataTransfer.getData(DRAG_MIME);
      if (!typeId || locked) return;
      e.preventDefault();
      placeInLane(typeId, screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    },
    [placeInLane, screenToFlowPosition, locked],
  );

  // Keyboard: Ctrl/Cmd+Z undo, Ctrl/Cmd+Shift+Z or Ctrl+Y redo, Ctrl/Cmd+D duplicate selection.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (!mod || locked) return;
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) undo();
      else if ((k === "z" && e.shiftKey) || k === "y") redo();
      else if (k === "d" && selectedId) duplicateNode(selectedId);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, duplicateNode, selectedId, locked]);

  const nodes: ShellNode[] = useMemo(() => {
    if (!workflow) return [];
    const steps = stepNumbers(workflow);
    return workflow.nodes.map((n) => ({
      id: n.id,
      type: "vlm",
      position: n.ui ?? { x: 0, y: 0 },
      selected: n.id === selectedId,
      draggable: !locked,
      deletable: !locked,
      data: {
        typeId: n.type,
        spec: specs.get(n.type),
        ports,
        issues: validation?.errors.filter((e) => e.node_id === n.id) ?? [],
        step: steps.get(n.id) ?? 0,
        params: n.params,
        runState: run?.nodes[n.id],
      },
    }));
  }, [workflow, specs, ports, validation, selectedId, run, locked]);

  const edges: Edge[] = useMemo(() => {
    if (!workflow) return [];
    return workflow.edges.map((e) => {
      const src = workflow.nodes.find((n) => n.id === e.from[0]);
      const type = src ? specs.get(src.type)?.outputs[e.from[1]]?.type : undefined;
      const pt = ports.find((p) => p.id === type);
      const active = run?.nodes[e.to[0]]?.status === "running";
      return {
        id: `${e.from.join(":")}->${e.to.join(":")}`,
        source: e.from[0],
        sourceHandle: e.from[1],
        target: e.to[0],
        targetHandle: e.to[1],
        animated: active,
        deletable: !locked,
        style: { stroke: pt ? tokenVar(pt.color_token) : undefined, strokeWidth: 2 },
      };
    });
  }, [workflow, specs, ports, run, locked]);

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
      if (locked) return false;
      const out = typeOf(c.source, c.sourceHandle, "out");
      const inp = typeOf(c.target, c.targetHandle, "in");
      return !!out && !!inp && c.source !== c.target && isCompatible(out, inp, ports);
    },
    [typeOf, ports, locked],
  );

  const label = useCallback(
    (type: string) => {
      const pt = ports.find((p) => p.id === type);
      return pt ? tp(pt.label_key.split(".")) : type;
    },
    [ports],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.sourceHandle || !c.targetHandle) return;
      const res = connect([c.source, c.sourceHandle], [c.target, c.targetHandle]);
      if (!res.ok) {
        setToast(
          t("canvas.wire_rejected", {
            target: c.targetHandle,
            expected: label(res.expected),
            actual: label(res.actual),
          }),
        );
      }
    },
    [connect, label],
  );

  // A rejected drag (isValidConnection=false) never reaches onConnect: explain it on release (no modal).
  const onConnectEnd = useCallback(
    (
      _e: MouseEvent | TouchEvent,
      state: {
        isValid: boolean | null;
        fromHandle: { nodeId: string; id?: string | null; type: string } | null;
        toHandle: { nodeId: string; id?: string | null } | null;
      },
    ) => {
      if (state.isValid !== false || !state.fromHandle || !state.toHandle) return;
      const fromOut = state.fromHandle.type === "source";
      const out = fromOut ? state.fromHandle : state.toHandle;
      const inp = fromOut ? state.toHandle : state.fromHandle;
      const actual = typeOf(out.nodeId, out.id, "out") ?? "";
      const expected = typeOf(inp.nodeId, inp.id, "in") ?? "";
      if (actual && expected) {
        setToast(
          t("canvas.wire_rejected", {
            target: inp.id ?? "",
            expected: label(expected),
            actual: label(actual),
          }),
        );
      }
    },
    [typeOf, label],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<ShellNode>[]) => {
      for (const ch of changes) {
        if (ch.type === "position" && ch.position) move(ch.id, ch.position.x, ch.position.y);
        if (ch.type === "position" && ch.dragging === false) commitMove();
        if (ch.type === "select" && ch.selected) select(ch.id);
      }
    },
    [move, commitMove, select],
  );

  const onNodesDelete = useCallback(
    (deleted: ShellNode[]) => {
      removeNodes(deleted.map((n) => n.id));
      const first = deleted[0];
      if (first)
        setToast(
          t("workflow.deleted_undo", { node: nodeTitle(first.data.spec, first.data.typeId) }),
        );
    },
    [removeNodes],
  );

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      for (const e of deleted) if (e.targetHandle) disconnect([e.target, e.targetHandle]);
    },
    [disconnect],
  );

  /** Double-click / Enter in the library: add at the end of the node's lane. */
  const addFromLibrary = useCallback((typeId: string) => void placeInLane(typeId), [placeInLane]);

  if (!workflow) return null;
  return (
    <div className="editor">
      <NodeLibrary onAdd={addFromLibrary} disabled={locked} />
      <section
        className="canvas"
        aria-label={workflow.name}
        ref={canvasRef}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes(DRAG_MIME)) {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
          }
        }}
        onDrop={onDrop}
      >
        <Toolbar locked={locked} />
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onNodesDelete={onNodesDelete}
          onConnect={onConnect}
          onConnectEnd={onConnectEnd}
          onEdgesDelete={onEdgesDelete}
          onPaneClick={() => select(null)}
          isValidConnection={isValidConnection}
          nodesConnectable={!locked}
          deleteKeyCode={locked ? null : ["Delete", "Backspace"]}
          defaultViewport={{ x: 16, y: 40, zoom: 1 }}
          minZoom={0.3}
          proOptions={{ hideAttribution: false }}
        >
          <Lanes />
          <Background gap={24} color="var(--color-canvas-grid-dot)" />
          <Controls />
        </ReactFlow>
        {workflow.nodes.length === 0 && (
          // Empty canvas: a target in lane ① instead of a blank page (design spec M1-5 §4).
          <div className="empty-canvas">
            <p className="empty-drop">{t("canvas.empty.drop_here")}</p>
            <p className="muted small">{t("canvas.empty.explain")}</p>
          </div>
        )}
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
