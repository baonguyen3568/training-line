import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import type { NodeSpec, PortType, ValidationIssue } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { nodeTitle } from "../lib/nodeText";
import { tokenVar } from "../lib/ports";
import { identifierView } from "../lib/safeText";

export interface ShellData extends Record<string, unknown> {
  typeId: string;
  spec: NodeSpec | undefined;
  ports: PortType[];
  issues: ValidationIssue[];
  step: number;
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

function NodeShellImpl({ data, selected }: NodeProps<ShellNode>) {
  const { spec, ports, issues, typeId, step } = data;
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
  const outputs = Object.entries(spec.outputs);
  return (
    <div className={classes} data-testid={`node-${typeId}`}>
      <div className="node-header">
        <span className="step" aria-hidden="true">
          {step}
        </span>
        <span className="node-title" title={nodeTitle(spec, typeId)}>
          {nodeTitle(spec, typeId)}
        </span>
      </div>
      {thirdParty && <ThirdPartyBadge spec={spec} />}
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
                className={`port shape-${ps.type}`}
                style={{ background: portColor(ports, ps.type) }}
              />
              <span>{portLabel(ports, typeId, name, ps.type)}</span>
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
        {errors.length > 0 && (
          <span className="state failed">
            ✕ {t("node_state.failed")} ({errors.length})
          </span>
        )}
        {errors.length === 0 && warns.length > 0 && (
          <span className="state warn">⚠ {warns.length}</span>
        )}
        {issues.length === 0 && <span className="state ready">○ {t("node_state.ready")}</span>}
      </div>
    </div>
  );
}

export const NodeShell = memo(NodeShellImpl);
