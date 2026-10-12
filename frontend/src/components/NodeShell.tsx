import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import type { NodeSpec, PortType, ValidationIssue } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { nodeTitle } from "../lib/nodeText";
import { customizedAdvanced } from "../lib/params";
import { tokenVar } from "../lib/ports";
import { identifierView, untrustedText } from "../lib/safeText";
import { useUi } from "../store/ui";
import type { NodeRunState } from "../store/workflow";

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

/** Run state on the node face: icon + text (never color alone) and a 3px progress bar while running. */
function RunBadge({ state }: { state: NodeRunState }) {
  const percent = state.progress === undefined ? undefined : Math.round(state.progress * 100);
  const text =
    state.status === "running"
      ? percent === undefined
        ? t("node_state.running", { percent: "…" })
        : t("node_state.running", { percent })
      : state.status === "cached"
        ? t("node_state.cached", { time: "" })
        : t(`node_state.${RUN_COPY[state.status]}`);
  return (
    <>
      <span className={`state run ${state.status}`}>
        {RUN_ICON[state.status]} {text}
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

function NodeShellImpl({ data, selected }: NodeProps<ShellNode>) {
  const { spec, ports, issues, typeId, step, params, runState } = data;
  const mode = useUi((s) => s.mode);
  const customized = mode === "simple" ? customizedAdvanced(spec, params) : [];
  // Key choices on the node face (x-ui.summary), e.g. the chosen model/dataset; repo ids shortened to the name.
  const summary = Object.entries(spec?.params_schema.properties ?? {})
    .filter(([, s]) => s["x-ui"]?.summary)
    .map(([name, s]) => {
      const v = String(params[name] ?? s.default ?? "");
      return s["x-ui"]?.option_meta?.[v]?.label ?? v; // catalog id → display name
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
            <li key={name}>
              <Handle
                type="target"
                position={Position.Left}
                id={name}
                className={`port shape-${ps.type}${missing.has(name) ? " needs-wire" : ""}`}
                style={{ background: portColor(ports, ps.type) }}
              />
              <span>{portLabel(ports, typeId, name, ps.type)}</span>
              {missing.has(name) && (
                // Users stumble exactly here (design M1 r1): name what to connect, from CFG-003.
                <span className="port-hint">← {t("canvas.hint.connect_port")}</span>
              )}
            </li>
          ))}
        </ul>
        <ul className="ports out">
          {outputs.map(([name, ps]) => (
            <li key={name}>
              <span>{portLabel(ports, typeId, name, ps.type)}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={name}
                className={`port shape-${ps.type}`}
                style={{ background: portColor(ports, ps.type) }}
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
            ✕ {t("node_state.failed")} ({errors.length})
          </span>
        )}
        {errors.length === 0 && warns.length > 0 && (
          <span className="state warn">⚠ {warns.length}</span>
        )}
        {runState ? (
          <RunBadge state={runState} />
        ) : (
          issues.length === 0 && <span className="state ready">○ {t("node_state.ready")}</span>
        )}
      </div>
    </div>
  );
}

export const NodeShell = memo(NodeShellImpl);
