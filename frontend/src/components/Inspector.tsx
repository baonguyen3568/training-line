import { useState } from "react";
import type { NodeSpec, ParamSchema, PortType, Workflow } from "../api/types";
import { t, tp } from "../i18n";
import { nodeDesc, nodeTitle } from "../lib/nodeText";
import { isCompatible } from "../lib/ports";
import { useWorkflow } from "../store/workflow";

interface SourceOption {
  value: string; // "nodeId:port"
  label: string;
}

/** Compatible upstream outputs for an input port — powers the keyboard-friendly "Nhận từ" picker (F6b). */
function sourcesFor(
  wf: Workflow,
  specs: Map<string, NodeSpec>,
  ports: PortType[],
  selfId: string,
  inputType: string,
): SourceOption[] {
  const out: SourceOption[] = [];
  for (const n of wf.nodes) {
    if (n.id === selfId) continue;
    const spec = specs.get(n.type);
    for (const [port, ps] of Object.entries(spec?.outputs ?? {})) {
      if (!isCompatible(ps.type, inputType, ports)) continue;
      const pt = ports.find((p) => p.id === ps.type);
      out.push({
        value: `${n.id}:${port}`,
        label: `${nodeTitle(spec, n.type)} · ${pt ? tp(pt.label_key.split(".")) : ps.type}`,
      });
    }
  }
  return out;
}

function ParamField({
  nodeType,
  name,
  schema,
  value,
  onChange,
}: {
  nodeType: string;
  name: string;
  schema: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = tp(["node", nodeType, "param", name, "label"]);
  const id = `param-${name}`;
  const current = value ?? schema.default;
  if (schema.type === "array" && schema.enum) {
    const selected = new Set(Array.isArray(current) ? current : []);
    return (
      <fieldset className="field">
        <legend>{label}</legend>
        {schema.enum.map((opt) => (
          <label key={String(opt)} className="check">
            <input
              type="checkbox"
              checked={selected.has(opt)}
              onChange={(e) => {
                const next = new Set(selected);
                if (e.target.checked) next.add(opt);
                else next.delete(opt);
                onChange(schema.enum?.filter((o) => next.has(o)));
              }}
            />
            {tp(["benchmark_group", String(opt)])}
          </label>
        ))}
      </fieldset>
    );
  }
  if (schema.enum) {
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <select id={id} value={String(current ?? "")} onChange={(e) => onChange(e.target.value)}>
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
      </div>
    );
  }
  if (schema.type === "boolean") {
    return (
      <label className="field check">
        <input
          type="checkbox"
          checked={Boolean(current)}
          onChange={(e) => onChange(e.target.checked)}
        />
        {label}
      </label>
    );
  }
  if (schema.type === "number" || schema.type === "integer") {
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <input
          id={id}
          type="number"
          min={schema.minimum}
          max={schema.maximum}
          step={schema.type === "integer" ? 1 : "any"}
          value={typeof current === "number" ? current : ""}
          onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
        />
      </div>
    );
  }
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="text"
        value={String(current ?? "")}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

export function Inspector() {
  const { workflow, specs, ports, selectedId } = useWorkflow();
  const { setParam, connect, disconnect } = useWorkflow.getState();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const node = workflow?.nodes.find((n) => n.id === selectedId);
  if (!workflow || !node) {
    return <aside className="inspector muted">{t("library.title")}</aside>;
  }
  const spec = specs.get(node.type);
  if (!spec) return <aside className="inspector">{t("error.CFG-004.title")}</aside>;

  const params = Object.entries(spec.params_schema.properties).sort(
    ([, a], [, b]) => (a["x-ui"]?.order ?? 99) - (b["x-ui"]?.order ?? 99),
  );
  const simple = params.filter(([, s]) => !s["x-ui"]?.advanced);
  const advanced = params.filter(([, s]) => s["x-ui"]?.advanced);

  return (
    <aside className="inspector" aria-label={nodeTitle(spec, node.type)}>
      <h2>{nodeTitle(spec, node.type)}</h2>
      <p className="muted">{nodeDesc(spec, node.type)}</p>

      {Object.entries(spec.inputs).map(([port, ps]) => {
        const edge = workflow.edges.find((e) => e.to[0] === node.id && e.to[1] === port);
        const options = sourcesFor(workflow, specs, ports, node.id, ps.type);
        const id = `recv-${port}`;
        const pt = ports.find((p) => p.id === ps.type);
        return (
          <div className="field" key={port}>
            <label htmlFor={id}>
              {t("canvas.receive_from")} — {pt ? tp(pt.label_key.split(".")) : ps.type}
            </label>
            <select
              id={id}
              value={edge ? edge.from.join(":") : ""}
              onChange={(e) => {
                const [srcNode, srcPort] = e.target.value.split(":");
                if (srcNode && srcPort) connect([srcNode, srcPort], [node.id, port]);
                else disconnect([node.id, port]);
              }}
            >
              <option value="">—</option>
              {options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        );
      })}

      {simple.map(([name, schema]) => (
        <ParamField
          key={name}
          nodeType={node.type}
          name={name}
          schema={schema}
          value={node.params[name]}
          onChange={(v) => setParam(node.id, name, v)}
        />
      ))}
      {advanced.length > 0 && (
        <button
          type="button"
          className="link"
          aria-expanded={showAdvanced}
          onClick={() => setShowAdvanced((s) => !s)}
        >
          ▸ {t("canvas.show_advanced", { param_count: advanced.length })}
        </button>
      )}
      {showAdvanced &&
        advanced.map(([name, schema]) => (
          <ParamField
            key={name}
            nodeType={node.type}
            name={name}
            schema={schema}
            value={node.params[name]}
            onChange={(v) => setParam(node.id, name, v)}
          />
        ))}
    </aside>
  );
}
