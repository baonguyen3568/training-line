import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";
import { type CSSProperties, memo } from "react";
import type { NodeSpec, PortType, ValidationIssue } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { minutesText } from "../lib/format";
import { nodeTitle } from "../lib/nodeText";
import { customizedAdvanced } from "../lib/params";
import { isCompatible, portShape, tokenVar } from "../lib/ports";
import { runWarningText } from "../lib/runWarning";
import { identifierView, untrustedText } from "../lib/safeText";
import { useUi } from "../store/ui";
import type { NodeRunState } from "../store/workflow";
import { Icon } from "./Icon";

export interface ShellData extends Record<string, unknown> {
  typeId: string;
  spec: NodeSpec | undefined;
  ports: PortType[];
  issues: ValidationIssue[];
  step: number;
  params: Record<string, unknown>;
  runState?: NodeRunState | undefined;
}
export type ShellNode = Node<ShellData, "vlm">;

/** Node-specific port copy (node.<type>.port.<name>) first, then the generic port-type label. */
function portLabel(ports: PortType[], typeId: string, name: string, type: string): string {
  const specific = tpOptional(["node", typeId, "port", name]);
  if (specific) return specific;
  const pt = ports.find((p) => p.id === type);
  return pt ? tp(pt.label_key.split(".")) : type;
}

function portColor(ports: PortType[], type: string): string {
  const pt = ports.find((p) => p.id === type);
  return pt ? tokenVar(pt.color_token) : "var(--color-port-value)";
}

function ThirdPartyBadge({ spec }: { spec: NodeSpec }) {
  // Badge depends ONLY on origin.kind/reviewed from the server, never on the node's name (P003 D16 T8).
  const publisher = identifierView(spec.origin.publisher ?? "?");
  const key = spec.origin.reviewed ? "custom_node.badge.reviewed" : "custom_node.badge.unverified";
  return (
    <span className="badge third-party" title={publisher.suspicious ? "⚠ non-ASCII" : undefined}>
      🧩 {t(key, { publisher: publisher.text })}
      {publisher.suspicious && " ⚠"}
    </span>
  );
}

const RUN_ICON: Record<NodeRunState["status"], string> = {
  pending: "⧗",
  cached: "↺",
  running: "◐",
  completed: "✓",
  failed: "✕",
  skipped: "–",
  interrupted: "⏸",
};
// Executor statuses → existing copy keys (node_state.*).
const RUN_COPY: Record<NodeRunState["status"], string> = {
  pending: "queued",
  cached: "cached",
  running: "running",
  completed: "done",
  failed: "failed",
  skipped: "skipped",
  interrupted: "interrupted",
};

// D4 (design 2026-10-13): Lucide glyph per state; only the glyph carries status ink, the label stays text ink.
const RUN_LUCIDE: Record<NodeRunState["status"], string> = {
  pending: "clock",
  cached: "rotate-ccw-clock",
  running: "loader-circle",
  completed: "circle-check-big",
  failed: "circle-x",
  skipped: "circle-minus",
  interrupted: "circle-pause",
};

function StateGlyph({ icon, tone }: { icon: string; tone: "bad" | "warn" | "muted" }) {
  return (
    <span className={`si si-${tone}`}>
      <Icon name={icon} size={14} />
    </span>
  );
}

/**
 * D5 "Chờ bộ nhớ trống": the job is queued until enough memory is free (or the GPU lock is released). Clock glyph in
 * warning ink, label in text ink; the tooltip names the numbers when the executor sent them (never "{need_gb}").
 */
function WaitingBadge({ waiting }: { waiting: NonNullable<NodeRunState["waiting"]> }) {
  const label = tpOptional(["node_state", waiting.state]) ?? t("node_state.queued"); // waiting_gpu copy may not exist yet
  const why =
    waiting.state === "waiting_gpu"
      ? t("node_state.waiting_gpu_tooltip")
      : waiting.need_gb !== undefined && waiting.avail_gb !== undefined
        ? t("node_state.waiting_memory_tooltip", {
            need_gb: Math.round(waiting.need_gb),
            avail_gb: Math.round(waiting.avail_gb),
          })
        : undefined;
  // "Đã chờ 1 giờ 5 phút." once the executor reports it (a wait can pass an hour: dur_* copy)
  const waited =
    waiting.waited_s !== undefined && waiting.waited_s >= 60
      ? t("node_state.waited_for", { duration: minutesText(waiting.waited_s / 60) })
      : undefined;
  const tip = [why, waited].filter(Boolean).join(" ") || undefined;
  return (
    <span className="state run waiting" title={tip} data-waiting={waiting.state}>
      <span className="si si-warn" aria-hidden="true">
        <Icon name="clock" size={14} />
      </span>{" "}
      {label}
      {tip && <span className="sr-only">. {tip}</span>}
    </span>
  );
}

/** Run state on the node face: icon + text (never color alone) and a 3px progress bar while running. */
function RunBadge({ state }: { state: NodeRunState }) {
  const d4 = useUi((s) => s.skin) === "d4";
  if (state.waiting && state.status === "running") return <WaitingBadge waiting={state.waiting} />;
  const percent = state.progress === undefined ? undefined : Math.round(state.progress * 100);
  const text =
    state.status === "running"
      ? percent === undefined
        ? t("node_state.running", { percent: "…" })
        : t("node_state.running", { percent })
      : state.status === "cached"
        ? t("node_state.cached_no_time") // no saved time here: never "lúc " with nothing after it
        : t(`node_state.${RUN_COPY[state.status]}`);
  return (
    <>
      <span className={`state run ${state.status}`}>
        {d4 ? (
          <span className={`si si-${state.status}`}>
            <Icon name={RUN_LUCIDE[state.status]} size={14} />
          </span>
        ) : (
          RUN_ICON[state.status]
        )}{" "}
        {text}
      </span>
      {state.status === "running" && (
        <span
          className="run-progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          style={{ width: `${percent ?? 100}%` }}
        />
      )}
    </>
  );
}

function NodeShellImpl({ id, data, selected }: NodeProps<ShellNode>) {
  const { spec, ports, issues, typeId, step, params, runState } = data;
  const mode = useUi((s) => s.mode);
  const wireDrag = useUi((s) => s.wireDrag);
  // While a wire is dragged, ports it cannot reach fade (same side, own node, incompatible type): opacity only,
  // never layout, so handle positions stay where React Flow measured them.
  const unreachable = (side: "source" | "target", type: string): boolean => {
    if (!wireDrag) return false;
    if (wireDrag.nodeId === id || wireDrag.side === side) return true;
    return side === "target"
      ? !isCompatible(wireDrag.type, type, ports)
      : !isCompatible(type, wireDrag.type, ports);
  };
  const d4 = useUi((s) => s.skin) === "d4";
  const customized = mode === "simple" ? customizedAdvanced(spec, params) : [];
  // Key choices on the node face (x-ui.summary), e.g. the chosen model/dataset; repo ids shortened to the name.
  const summary = Object.entries(spec?.params_schema.properties ?? {})
    .filter(([, s]) => s["x-ui"]?.summary)
    .map(([name, s]) => {
      const v = String(params[name] ?? s.default ?? "");
      const keys = s["x-ui"]?.option_keys;
      // catalog id → display name; enum value → its option copy ("3" → "3 lượt", design D4 r1 B4)
      return s["x-ui"]?.option_meta?.[v]?.label ?? (keys ? tpOptional([keys, v]) : undefined) ?? v;
    })
    .filter(Boolean)
    .map((v) =>
      spec?.origin.kind === "third_party" ? untrustedText(v, 40) : (v.split("/").pop() ?? v),
    );
  const errors = issues.filter((i) => i.severity === "error");
  const warns = issues.filter((i) => i.severity === "warn");
  const thirdParty = spec?.origin.kind === "third_party";
  const classes = [
    "vlm-node",
    selected ? "selected" : "",
    errors.length ? "has-error" : warns.length ? "has-warn" : "",
    thirdParty ? "third-party" : "",
    spec ? "" : "placeholder",
  ].join(" ");

  if (!spec) {
    // Missing node type (CFG-004): keep node + wires, never auto-install (P003 D11).
    const id = identifierView(typeId);
    return (
      <div className={classes}>
        <div className="node-header">⚠ {id.text}</div>
        <div className="node-body muted">{t("error.CFG-004.title")}</div>
      </div>
    );
  }

  const inputs = Object.entries(spec.inputs);
  const missing = new Set(
    issues.filter((i) => i.code === "CFG-003" && i.port).map((i) => i.port as string),
  );
  const outputs = Object.entries(spec.outputs);
  return (
    <div
      className={`${classes}${runState ? ` run-${runState.status}` : ""}`}
      data-testid={`node-${typeId}`}
      data-status={runState?.status ?? "idle"}
    >
      <div className="node-header">
        <span className="step" aria-hidden="true">
          {step}
        </span>
        <span className="node-title" title={nodeTitle(spec, typeId)}>
          {nodeTitle(spec, typeId)}
        </span>
      </div>
      {thirdParty && <ThirdPartyBadge spec={spec} />}
      {summary.length > 0 && <div className="node-summary">{summary.join(" · ")}</div>}
      {spec.policy_block && (
        <span className="badge blocked">⛔ {t("custom_node.badge.policy_block")}</span>
      )}
      <div className="node-ports">
        <ul className="ports in">
          {inputs.map(([name, ps]) => (
            <li key={name} className={unreachable("target", ps.type) ? "port-dim" : undefined}>
              <Handle
                type="target"
                position={Position.Left}
                id={name}
                className={`port shape-${ps.type} hg-${portShape(ps.type)}${missing.has(name) ? " needs-wire" : ""}`}
                style={
                  {
                    background: portColor(ports, ps.type),
                    "--pc": portColor(ports, ps.type),
                  } as CSSProperties
                }
              />
              <span>{portLabel(ports, typeId, name, ps.type)}</span>
              {missing.has(name) && (
                // Users stumble exactly here (design M1 r1): name what to connect, from CFG-003.
                <span className="port-hint">{t("canvas.hint.connect_port")}</span>
              )}
            </li>
          ))}
        </ul>
        <ul className="ports out">
          {outputs.map(([name, ps]) => (
            <li key={name} className={unreachable("source", ps.type) ? "port-dim" : undefined}>
              <span>{portLabel(ports, typeId, name, ps.type)}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={name}
                className={`port shape-${ps.type} hg-${portShape(ps.type)}`}
                style={
                  {
                    background: portColor(ports, ps.type),
                    "--pc": portColor(ports, ps.type),
                  } as CSSProperties
                }
              />
            </li>
          ))}
        </ul>
      </div>
      <div className="node-footer">
        {spec.permissions.map((p) => (
          <span
            key={p.id}
            className={`perm level-${p.level}`}
            title={t(p.perm_copy_key, { max_runtime: "" })}
          >
            {p.id === "gpu" ? "▣" : p.id === "network" ? "🌐" : "⏱"}{" "}
            {t(`custom_node.badge_perm.${p.id}`)}
          </span>
        ))}
        {customized.length > 0 && (
          <span className="state customized">
            ⚙ {t("canvas.customized", { param_count: customized.length })}
          </span>
        )}
        {errors.length > 0 && (
          <span className="state failed">
            {d4 ? <StateGlyph icon="circle-dashed" tone="bad" /> : "✕"}{" "}
            {d4 ? t("node_state.not_ready") : t("node_state.failed")} ({errors.length})
          </span>
        )}
        {errors.length === 0 && warns.length > 0 && (
          <span className="state warn">
            {d4 ? <StateGlyph icon="triangle-alert" tone="warn" /> : "⚠"} {warns.length}
          </span>
        )}
        {runState?.warnings?.map((w, i) => {
          const text = runWarningText(w, () => nodeTitle(spec, typeId));
          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: warnings are append-only for one run
            <span key={i} className="state warn run-warning" title={text.what}>
              {d4 ? <StateGlyph icon="triangle-alert" tone="warn" /> : "⚠"} {text.title}
            </span>
          );
        })}
        {runState ? (
          <RunBadge state={runState} />
        ) : (
          issues.length === 0 && (
            <span className="state ready">
              {d4 ? <StateGlyph icon="circle-dot" tone="muted" /> : "○"} {t("node_state.ready")}
            </span>
          )
        )}
      </div>
    </div>
  );
}

export const NodeShell = memo(NodeShellImpl);
