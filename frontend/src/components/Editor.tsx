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
import {
  LANE_WIDTH,
  LANES,
  type Lane,
  type LaneBox,
  laneBoxes,
  laneOf,
  occupiedLanes,
  ROW_HEIGHT,
  toDisplayX,
  toModelX,
} from "../lib/lanes";
import { nodeTitle } from "../lib/nodeText";
import { isCompatible, tokenVar } from "../lib/ports";
import { selfAndDownstream, stepNumbers } from "../lib/topo";
import { connectionBetween, type HandleRef, handleUnder } from "../lib/wire";
import { useUi } from "../store/ui";
import { graphKey, useWorkflow } from "../store/workflow";
import { BugReportDialog } from "./BugReportDialog";
import { CheckBar } from "./CheckBar";
import { Icon } from "./Icon";
import { Inspector } from "./Inspector";
import { DRAG_MIME, NodeLibrary } from "./NodeLibrary";
import { NodeShell, type ShellNode } from "./NodeShell";
import { ResetStepDialog } from "./ResetStepDialog";
import { Dock, IconStrip, LibraryFlyout, StatusBar } from "./ShellD4";
import { Toolbar } from "./Toolbar";
import { ToolbarD4 } from "./ToolbarD4";

const nodeTypes = { vlm: NodeShell };
const NARROW = "(max-width: 1279px)";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia?.(NARROW).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW);
    if (!mq) return;
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}
const VALIDATE_DEBOUNCE_MS = 500;

/** A measured size only for the node it was measured on (same type); else undefined, so React Flow measures it. */
export function measuredOf(
  entry: { type: string; width: number; height: number } | undefined,
  type: string,
): { width: number; height: number } | undefined {
  return entry && entry.type === type ? { width: entry.width, height: entry.height } : undefined;
}

const LANE_TOP = -4000;
const LANE_HEIGHT = 12000;

function Lanes({ boxes }: { boxes: readonly LaneBox[] }) {
  // Labels stick to the top edge of the visible canvas and stay ≥12px on screen at any zoom (design review #2).
  const { y, zoom } = useViewport();
  const labelTop = (-y + 48) / zoom - LANE_TOP; // below the canvas toolbar
  return (
    <ViewportPortal>
      {boxes.map((box) => {
        const name = t(`canvas.lane.${box.lane}`);
        return (
          <div
            key={box.lane}
            className={box.compact ? "lane compact" : "lane"}
            data-lane={box.lane}
            style={{
              transform: `translate(${box.x}px, ${LANE_TOP}px)`,
              width: box.width,
              height: LANE_HEIGHT,
            }}
          >
            <span
              className="lane-label"
              title={box.compact ? name : undefined}
              style={{ top: labelTop, fontSize: `${12 / Math.min(zoom, 1)}px` }}
            >
              {/* compact: only the circled number ①…⑦, full name in the tooltip */}
              {box.compact ? Array.from(name)[0] : name}
            </span>
          </div>
        );
      })}
    </ViewportPortal>
  );
}

function EditorInner() {
  const { workflow, specs, ports, validation, selectedId, run, opened } = useWorkflow();
  const skin = useUi((st) => st.skin);
  // D4 compact lanes: geometry from occupied lanes; frozen while a node is dragged (a lane only shrinks after
  // the drop), and the target lane of a library drag opens under the cursor.
  const [dragLane, setDragLane] = useState<Lane | null>(null);
  const [nodeDragging, setNodeDragging] = useState(false);
  const frozen = useRef<LaneBox[] | null>(null);
  const liveBoxes = useMemo(
    () =>
      laneBoxes(
        occupiedLanes((workflow?.nodes ?? []).map((n) => n.ui?.x ?? 0)),
        dragLane,
        true, // both skins: empty lanes compact, so a template fits at 1280 px (tester, classic desktop-min)
      ),
    [workflow, dragLane],
  );
  if (!nodeDragging) frozen.current = null;
  else if (!frozen.current) frozen.current = liveBoxes;
  const boxes = frozen.current ?? liveBoxes;
  useEffect(() => {
    const onDrag = (e: Event) => {
      const typeId = (e as CustomEvent<string | null>).detail;
      const cat = typeId ? specs.get(typeId)?.category : undefined;
      setDragLane(cat ? laneOf(cat) : null);
    };
    window.addEventListener("vlm:library-drag", onDrag);
    return () => window.removeEventListener("vlm:library-drag", onDrag);
  }, [specs]);
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
  const { screenToFlowPosition, fitView, getViewport, getInternalNode } = useReactFlow();
  const [flashId, setFlashId] = useState<string | null>(null);
  const canvasRef = useRef<HTMLElement>(null);
  const [toast, setToast] = useState<string | null>(null);
  // G-4: right-click menu of a node that already ran (its run-dir base is known from node_status)
  const [menu, setMenu] = useState<{ x: number; y: number; base: string } | null>(null);
  const resetNode = useUi((st) => st.resetNode);
  const bugReport = useUi((st) => st.bugReport);
  useEffect(() => {
    const onToast = (e: Event) => setToast((e as CustomEvent<string>).detail);
    window.addEventListener("vlm:toast", onToast);
    return () => window.removeEventListener("vlm:toast", onToast);
  }, []);
  useEffect(() => {
    if (!menu) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent && e.key !== "Escape") return;
      setMenu(null);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [menu]);
  const key = graphKey(workflow);
  // While a run is live the graph structure is locked (design spec M1-5 §7).
  const narrow = useNarrow();
  const runLocked = !!run && !run.done;
  // Narrow screens (< 1280px) are view-only (P010): structure is locked, but Run/inspection still work.
  const locked = runLocked || narrow;

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
      const at = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      placeInLane(typeId, { x: toModelX(at.x, boxes), y: at.y });
      setDragLane(null);
    },
    [placeInLane, screenToFlowPosition, locked, boxes],
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

  // Sizes React Flow measured (its "dimensions" changes). Every rebuild below makes new node objects (validation,
  // selection, flash…); without `measured` React Flow treats each as re-initialised: it drops the handle bounds and
  // hides the node until re-measured, so a wire dropped then finds no port and a press on a port pans the canvas
  // (tester x100 traces, @xyflow/system adoptUserNodes/parseHandles). Carrying `measured` keeps them.
  // An entry belongs to one node of one workflow: dropped when the node is removed (its id may come back as another
  // node), ignored if the type differs, and cleared when another workflow opens (ids n1, n2… repeat).
  const measuredRef = useRef(new Map<string, { type: string; width: number; height: number }>());
  const openedRef = useRef(opened);
  const nodes: ShellNode[] = useMemo(() => {
    if (openedRef.current !== opened) {
      measuredRef.current.clear();
      openedRef.current = opened;
    }
    if (!workflow) return [];
    const steps = stepNumbers(workflow);
    return workflow.nodes.map((n) => ({
      id: n.id,
      type: "vlm",
      measured: measuredOf(measuredRef.current.get(n.id), n.type),
      position: { x: toDisplayX(n.ui?.x ?? 0, boxes), y: n.ui?.y ?? 0 },
      selected: n.id === selectedId,
      className: n.id === flashId ? "flash" : undefined,
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
  }, [workflow, specs, ports, validation, selectedId, run, locked, flashId, boxes, opened]);

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
      const wf = useWorkflow.getState().workflow;
      const cycle = wf ? selfAndDownstream(wf, c.target).has(c.source) : c.source === c.target;
      return !!out && !!inp && !cycle && isCompatible(out, inp, ports);
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

  // Safety net: a drag that ends without onConnectEnd (Esc, alt-tab, window blur) must not leave ports dimmed.
  useEffect(() => {
    const clear = () => {
      if (useUi.getState().wireDrag) useUi.getState().setWireDrag(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") clear();
    };
    window.addEventListener("blur", clear);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("blur", clear);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  const onConnectStart = useCallback(
    (
      _e: MouseEvent | TouchEvent,
      p: { nodeId: string | null; handleId: string | null; handleType: "source" | "target" | null },
    ) => {
      if (!p.nodeId || !p.handleType) return;
      const type = typeOf(p.nodeId, p.handleId, p.handleType === "source" ? "out" : "in");
      if (type) useUi.getState().setWireDrag({ nodeId: p.nodeId, side: p.handleType, type });
    },
    [typeOf],
  );

  // A rejected drag (isValidConnection=false) never reaches onConnect: explain it on release (no modal).
  // A drop React Flow did not match to a port (stale handle bounds right after a node mounts, ~1/100) is matched
  // from the DOM and connected when valid, so a wire dropped on a port always sticks or explains why.
  const onConnectEnd = useCallback(
    (
      e: MouseEvent | TouchEvent,
      state: {
        isValid: boolean | null;
        fromHandle: { nodeId: string; id?: string | null; type: string } | null;
        toHandle: { nodeId: string; id?: string | null; type?: string } | null;
      },
    ) => {
      useUi.getState().setWireDrag(null);
      if (state.isValid === true || !state.fromHandle) return; // onConnect has it
      let to: HandleRef | null = state.toHandle?.type
        ? { ...state.toHandle, type: state.toHandle.type }
        : null;
      if (!to) {
        const p = "changedTouches" in e ? e.changedTouches[0] : e;
        if (!p) return;
        to = handleUnder(document.elementFromPoint(p.clientX, p.clientY));
        if (!to) return; // dropped on empty canvas: nothing to say
      }
      const conn = connectionBetween(state.fromHandle, to);
      if (!conn) {
        // two outputs, two inputs, or the step's own port: say how wires go (design; ports were dimmed meanwhile)
        setToast(t("canvas.wire_same_side"));
        return;
      }
      if (isValidConnection(conn)) {
        // Should no longer fire now that nodes carry `measured` (see measuredRef); kept as a safety net, and the
        // log tells E2E (tester) if React Flow still misses a port.
        console.debug("[wire] connected from the DOM drop target", conn);
        onConnect(conn);
        return;
      }
      const actual = typeOf(conn.source, conn.sourceHandle, "out") ?? "";
      const expected = typeOf(conn.target, conn.targetHandle, "in") ?? "";
      if (actual && expected) {
        setToast(
          t("canvas.wire_rejected", {
            target: conn.targetHandle ?? "",
            expected: label(expected),
            actual: label(actual),
          }),
        );
      }
    },
    [typeOf, label, isValidConnection, onConnect],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<ShellNode>[]) => {
      for (const ch of changes) {
        if (ch.type === "dimensions" && ch.dimensions) {
          const type = useWorkflow.getState().workflow?.nodes.find((n) => n.id === ch.id)?.type;
          if (type) measuredRef.current.set(ch.id, { type, ...ch.dimensions });
        }
        if (ch.type === "remove") measuredRef.current.delete(ch.id);
        if (ch.type === "position" && ch.position)
          move(ch.id, toModelX(ch.position.x, boxes), ch.position.y);
        if (ch.type === "position" && ch.dragging === true) setNodeDragging(true);
        if (ch.type === "position" && ch.dragging === false) {
          commitMove();
          setNodeDragging(false);
        }
        if (ch.type === "select" && ch.selected) select(ch.id);
      }
    },
    [move, commitMove, select, boxes],
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
  const addFromLibrary = useCallback(
    (typeId: string) => {
      const id = placeInLane(typeId);
      const pos = id && useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.ui;
      if (!id || !pos) return;
      // A node added off-screen must not be "lost" (design M1 r1): pan ONLY if it is outside the visible
      // canvas — panning on every add pushed earlier nodes out of view. Always select + flash it.
      const vp = getViewport();
      const rect = canvasRef.current?.getBoundingClientRect();
      // the node's lane opens after the add: measure against the new lane geometry
      const after = laneBoxes(
        occupiedLanes((useWorkflow.getState().workflow?.nodes ?? []).map((n) => n.ui?.x ?? 0)),
        null,
        true,
      );
      const left = toDisplayX(pos.x, after) * vp.zoom + vp.x;
      const top = pos.y * vp.zoom + vp.y;
      const visible =
        !!rect &&
        left >= 0 &&
        top >= 0 &&
        left + 240 * vp.zoom <= rect.width &&
        top + 120 * vp.zoom <= rect.height;
      // Off-screen: fit ALL nodes (zoom ≤ 1, ≥ 0.6) instead of centering the new one, so earlier nodes
      // stay on screen too. Wait until React Flow has measured the new node (ResizeObserver), else it is left
      // out of the bounds. No animation: a 250 ms pan moved the handles under a wire the user had already
      // started dragging (lost-wire flake).
      const fitWhenMeasured = (framesLeft: number) => {
        if (getInternalNode(id)?.measured.width || framesLeft === 0)
          void fitView({ maxZoom: 1, minZoom: 0.6, padding: 0.1 });
        else requestAnimationFrame(() => fitWhenMeasured(framesLeft - 1));
      };
      if (!visible) requestAnimationFrame(() => fitWhenMeasured(10));
      setFlashId(id);
      setTimeout(() => setFlashId((cur) => (cur === id ? null : cur)), 1200);
    },
    [placeInLane, fitView, getViewport, getInternalNode],
  );

  if (!workflow) return null;
  const d4 = skin === "d4";
  const errorCount = validation?.errors.filter((e) => e.severity === "error").length ?? 0;
  return (
    <div className={d4 ? "editor d4" : "editor"}>
      {d4 ? (
        <>
          <ToolbarD4 locked={locked} />
          <IconStrip />
          <LibraryFlyout>
            <NodeLibrary onAdd={addFromLibrary} disabled={locked} />
          </LibraryFlyout>
        </>
      ) : (
        <NodeLibrary onAdd={addFromLibrary} disabled={locked} />
      )}
      <main
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
        {!d4 && <Toolbar locked={locked} running={runLocked} />}
        {narrow && (
          <p className="readonly-note" role="note">
            {t("app.read_only_small_screen")}
          </p>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onNodesDelete={onNodesDelete}
          onNodeContextMenu={(e, n) => {
            const base = run?.nodes[n.id]?.nodeBase;
            if (!base) return; // nothing to reset yet: keep the browser's own menu
            e.preventDefault();
            setMenu({ x: e.clientX, y: e.clientY, base });
          }}
          onConnect={onConnect}
          onConnectEnd={onConnectEnd}
          onConnectStart={onConnectStart}
          onEdgesDelete={onEdgesDelete}
          onPaneClick={() => select(null)}
          isValidConnection={isValidConnection}
          nodesConnectable={!locked}
          deleteKeyCode={locked ? null : ["Delete", "Backspace"]}
          defaultViewport={{ x: 16, y: 40, zoom: 1 }}
          // D4 "one screen": empty lanes are compact, so the whole chain fits at ≥75% (design r1 B3).
          // Both skins: an opened template shows every step (tester: classic left step 5 off-screen at scale 1).
          fitView={workflow.nodes.length > 0}
          fitViewOptions={
            d4
              ? { minZoom: 0.75, maxZoom: 1, padding: 0.06 }
              : { minZoom: 0.6, maxZoom: 1, padding: 0.1 }
          }
          minZoom={0.3}
          proOptions={{ hideAttribution: false }}
        >
          <Lanes boxes={boxes} />
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
        {menu && (
          <div className="node-menu" role="menu" style={{ left: menu.x, top: menu.y }}>
            <button
              type="button"
              role="menuitem"
              title={t("storage.reset_step.menu_hint")}
              // pointerdown would close the menu before the click lands
              onPointerDown={(e) => e.stopPropagation()}
              onClick={() => {
                useUi.getState().setResetNode(menu.base);
                setMenu(null);
              }}
              ref={(el) => el?.focus()}
            >
              <Icon name="rotate-ccw" size={14} /> {t("storage.reset_step.menu")}
            </button>
          </div>
        )}
        {bugReport && (
          <BugReportDialog
            runId={bugReport.runId}
            nodeId={bugReport.nodeId}
            code={bugReport.code}
            onClose={() => useUi.getState().setBugReport(null)}
          />
        )}
        {resetNode && (
          <ResetStepDialog
            node={resetNode}
            onClose={() => useUi.getState().setResetNode(null)}
            onDone={setToast}
          />
        )}
      </main>
      <Inspector />
      {d4 ? (
        <Dock errorCount={errorCount}>
          <CheckBar />
        </Dock>
      ) : null}
      {d4 ? <StatusBar /> : <CheckBar />}
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
