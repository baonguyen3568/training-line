import { type ReactNode, useState } from "react";
import type { NodeSpec, OptionMeta, ParamSchema, PortType, Range, Workflow } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import {
  durationRunCount,
  formatEstimate,
  formatMinutesRange,
  workflowDurationText,
} from "../lib/format";
import { nodeDesc, nodeTitle, SEVERITY_GLYPH, workflowTitle } from "../lib/nodeText";
import { customizedAdvanced } from "../lib/params";
import { isCompatible } from "../lib/ports";
import { selfAndDownstream } from "../lib/topo";
import { useSystem } from "../store/system";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";
import { ImageUpload } from "./ImageUpload";
import { TrialResult } from "./TrialResult";

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
  const forbidden = selfAndDownstream(wf, selfId); // design review M1 K2: no self / downstream (cycle)
  for (const n of wf.nodes) {
    if (forbidden.has(n.id)) continue;
    const spec = specs.get(n.type);
    for (const [port, ps] of Object.entries(spec?.outputs ?? {})) {
      if (!isCompatible(ps.type, inputType, ports)) continue;
      const pt = ports.find((p) => p.id === ps.type);
      out.push({
        value: `${n.id}:${port}`,
        label: `${nodeTitle(spec, n.type)} · ${
          tpOptional(["node", n.type, "port", port]) ?? (pt ? tp(pt.label_key.split(".")) : ps.type)
        }`,
      });
    }
  }
  return out;
}

const RAM_HEADROOM_GB = 10; // design: ram_gb_train + 10 GB must fit in MemAvailable

/**
 * Enum option label. Catalog options (`x-ui.option_meta`): "{label} — {tagline}" + text badges (gated, RAM fit
 * against the live MemAvailable). Otherwise `x-ui.option_keys` prefix, else the raw value.
 */
function optionLabel(schema: ParamSchema, value: string | number, availGb?: number): string {
  const meta = schema["x-ui"]?.option_meta?.[String(value)];
  if (meta) {
    const tagline = tpOptional(["model_catalog", String(value), "tagline"]);
    const parts = [tagline ? `${meta.label} — ${tagline}` : meta.label];
    if (meta.gated) parts.push(`🔑 ${t("model_picker.gated")}`);
    if (meta.ram_gb !== undefined && availGb !== undefined) {
      const need = Math.round(meta.ram_gb + RAM_HEADROOM_GB);
      parts.push(
        need > availGb
          ? `⚠ ${t("model_picker.needs_free_ram", { need_gb: need, avail_gb: Math.round(availGb) })}`
          : `✓ ${t("model_picker.fits")}`,
      );
    }
    return parts.join(" · ");
  }
  const prefix = schema["x-ui"]?.option_keys;
  return prefix ? tp([...prefix.split("."), String(value)]) : String(value);
}

/** Simple mode hides research/baseline catalog entries (tier rnd/baseline); Advanced shows all. */
function visibleOptions(schema: ParamSchema, mode: string, current: unknown): (string | number)[] {
  const all = schema.enum ?? [];
  const meta = schema["x-ui"]?.option_meta;
  if (!meta || mode === "advanced") return all;
  return all.filter(
    (v) => v === current || !["rnd", "baseline"].includes(meta[String(v)]?.tier ?? ""),
  );
}

/** Reason text when this value is marked x-ui.disabled_options, else undefined. */
function disabledReason(schema: ParamSchema, value: unknown): string | undefined {
  const map = schema["x-ui"]?.disabled_options;
  const key = String(value);
  if (!map || !(key in map)) return undefined;
  const reason = map[key];
  return reason ? tp(reason.split(".")) : t("param_unsupported.default");
}

function ParamField({
  nodeType,
  name,
  schema,
  value,
  onChange,
  optionMinutes,
}: {
  nodeType: string;
  name: string;
  schema: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  /** Measured time per option (estimate per_node.by_epochs), shown next to each option. */
  optionMinutes?: Record<string, Range | null>;
}) {
  const label = tp(["node", nodeType, "param", name, "label"]);
  const id = `param-${name}`;
  const mode = useUi((s) => s.mode);
  const availGb = useSystem((s) => s.status?.mem_available_gb);
  const current = value ?? schema.default;
  const widget = schema["x-ui"]?.widget;
  if (widget === "image_upload") {
    return <ImageUpload id={id} label={label} value={String(current ?? "")} onChange={onChange} />;
  }
  if (widget === "textarea") {
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <textarea
          id={id}
          rows={3}
          maxLength={schema.maxLength}
          value={String(current ?? "")}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    );
  }
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
            {optionLabel(schema, opt)}
          </label>
        ))}
      </fieldset>
    );
  }
  if (schema.enum) {
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <select
          id={id}
          value={String(current ?? "")}
          // Map back to the original enum value so integer enums (epochs 1|3|5) stay numbers.
          onChange={(e) =>
            onChange(schema.enum?.find((o) => String(o) === e.target.value) ?? e.target.value)
          }
        >
          {visibleOptions(schema, mode, current).map((opt) => {
            const off = disabledReason(schema, opt);
            const rec = schema["x-ui"]?.recommended;
            const minutes = optionMinutes?.[String(opt)];
            return (
              // a dimmed option is still announced: its text carries "· chưa hỗ trợ"
              <option
                key={String(opt)}
                value={String(opt)}
                disabled={off !== undefined && opt !== current}
              >
                {optionLabel(schema, opt, availGb)}
                {minutes ? ` · ${formatMinutesRange(minutes)}` : ""}
                {rec !== undefined && String(opt) === String(rec)
                  ? ` · ${t("inspector.recommended")}`
                  : ""}
                {off !== undefined ? ` · ${t("param_unsupported.default")}` : ""}
              </option>
            );
          })}
        </select>
      </div>
    );
  }
  if (schema.type === "boolean") {
    // "true" not runnable yet: locked off; an old file that has it on can still turn it off
    const off = disabledReason(schema, true);
    const locked = off !== undefined && !current;
    return (
      <label className="field check">
        <input
          id={id}
          type="checkbox"
          checked={Boolean(current)}
          disabled={locked}
          aria-describedby={off !== undefined ? `${id}-unsupported` : undefined}
          onChange={(e) => onChange(e.target.checked)}
        />
        {label}
        {off !== undefined && (
          <span id={`${id}-unsupported`} className="unsupported">
            {" "}
            · {t("param_unsupported.default")}
            <span className="sr-only">: {off}</span>
          </span>
        )}
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

type FieldProps = Parameters<typeof ParamField>[0];

/**
 * Field + its plain-language help (node.<type>.param.<name>.help). Classic: a help line under the field.
 * D4: the help sits behind a "?" popover (no stray text), while this field's own errors stay right under it.
 */
function ParamRow(props: FieldProps & { errors?: string[] }) {
  const { errors = [], ...field } = props;
  const help = tpOptional(["node", props.nodeType, "param", props.name, "help"]);
  const d4 = useUi((s) => s.skin) === "d4";
  const popId = `help-${props.name}`;
  return (
    <div className="param-row">
      <ParamField {...field} />
      {help && !d4 && <p className="help">{help}</p>}
      {help && d4 && (
        <>
          <button
            type="button"
            className="help-btn"
            popoverTarget={popId}
            aria-label={t("d4.explain")}
          >
            <Icon name="circle-question-mark" size={14} />
          </button>
          <div id={popId} popover="auto" className="pop">
            {help}
          </div>
        </>
      )}
      {d4 &&
        errors.map((e) => (
          <p key={e} className="field-error" role="alert">
            {e}
          </p>
        ))}
    </div>
  );
}

/** Catalog entry of the model feeding this node: follow model-typed inputs upstream to a model picker. */
function upstreamModelMeta(
  workflow: Workflow,
  specs: Map<string, NodeSpec>,
  nodeId: string,
  seen = new Set<string>(),
): OptionMeta | undefined {
  if (seen.has(nodeId)) return undefined;
  seen.add(nodeId);
  const node = workflow.nodes.find((n) => n.id === nodeId);
  const spec = node && specs.get(node.type);
  if (!node || !spec) return undefined;
  const picked = spec.params_schema.properties.model?.["x-ui"]?.option_meta;
  if (picked && typeof node.params.model === "string") return picked[node.params.model];
  for (const [port, ps] of Object.entries(spec.inputs)) {
    if (ps.type !== "Model" && ps.type !== "Checkpoint") continue;
    const edge = workflow.edges.find((e) => e.to[0] === nodeId && e.to[1] === port);
    const found = edge && upstreamModelMeta(workflow, specs, edge.from[0], seen);
    if (found) return found;
  }
  return undefined;
}

/**
 * D4: with nothing selected the right column still helps (design r1 #3): step count, pre-flight state and the
 * estimate with its source; unknown numbers read "chưa có số đo", never a guess.
 */
function WorkflowSummary() {
  const { workflow, validation, specs } = useWorkflow();
  if (!workflow) return null;
  const errors = validation?.errors.filter((e) => e.severity === "error").length ?? 0;
  const warnings = validation?.errors.filter((e) => e.severity === "warn").length ?? 0;
  const est = validation?.estimate;
  const unknown = !est || est.source === "unknown";
  const duration = est?.duration;
  const runCount = durationRunCount(est);
  const unmeasured = (duration?.unmeasured ?? []).map((id) => {
    const type = workflow.nodes.find((n) => n.id === id)?.type ?? id;
    return nodeTitle(specs.get(type), type);
  });
  const rows: [string, string, ReactNode][] = [
    [
      "timer",
      t("d4.est_time"),
      <>
        {workflowDurationText(est)}
        {duration && (
          <>
            <button
              type="button"
              className="help-btn inline"
              popoverTarget="est-duration-help"
              aria-label={t("d4.explain")}
            >
              <Icon name="circle-question-mark" size={14} />
            </button>
            <span id="est-duration-help" popover="auto" className="pop">
              <span className="pop-p">{t("estimate.duration_scope")}</span>
              {runCount > 0 && (
                <span className="pop-p">
                  {t("estimate.duration_basis", { run_count: runCount })}
                </span>
              )}
              {unmeasured.length > 0 && (
                <span className="pop-p">
                  {t("estimate.duration_unmeasured", { nodes: unmeasured.join(", ") })}
                </span>
              )}
            </span>
          </>
        )}
      </>,
    ],
    [
      "memory-stick",
      "RAM",
      unknown
        ? t("estimate.source.unknown")
        : formatEstimate(est.ram_gb_peak, "ram_gb", est.source),
    ],
    [
      "hard-drive",
      t("header.disk"),
      unknown ? t("estimate.source.unknown") : formatEstimate(est.disk_gb, "disk_gb", est.source),
    ],
  ];
  return (
    <aside className="inspector wf-summary" aria-label={workflowTitle(workflow.name)}>
      <h2>{workflowTitle(workflow.name)}</h2>
      <p>{t("d4.step_count", { step_count: workflow.nodes.length })}</p>
      <p>
        {errors === 0 && warnings === 0 ? (
          <>
            <span className="si si-completed">
              <Icon name="circle-check-big" size={14} />
            </span>{" "}
            {t("preflight.summary_ok")}
          </>
        ) : (
          <>
            <span className={`si ${errors ? "si-bad" : "si-warn"}`}>
              <Icon name={errors ? "circle-x" : "triangle-alert"} size={14} />
            </span>{" "}
            {t("d4.issue_count", { error_count: errors, warning_count: warnings })}
          </>
        )}
      </p>
      <dl className="est">
        {rows.map(([icon, label, value]) => (
          <div key={label}>
            <dt>
              <Icon name={icon} size={14} /> {label}
            </dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

export function Inspector() {
  const { workflow, specs, ports, selectedId, validation, run } = useWorkflow();
  const skin = useUi((st) => st.skin);
  const { setParam, connect, disconnect } = useWorkflow.getState();
  const mode = useUi((s) => s.mode);
  const [expanded, setExpanded] = useState(false);
  const node = workflow?.nodes.find((n) => n.id === selectedId);
  if (!workflow || !node) {
    if (skin === "d4" && workflow) return <WorkflowSummary />;
    return <aside className="inspector muted">{t("inspector.empty_hint")}</aside>;
  }
  const spec = specs.get(node.type);
  if (!spec) return <aside className="inspector">{t("error.CFG-004.title")}</aside>;
  const issues = validation?.errors.filter((e) => e.node_id === node.id) ?? [];
  const fieldErrors = (param: string) => issues.filter((e) => e.param === param).map((e) => e.what);

  // x-ui.visible_when: a field that does not apply right now is hidden but keeps its value, and is shown anyway
  // while it has a validation error (never hide a problem).
  const props = spec.params_schema.properties;
  const applies = (name: string, s: ParamSchema) =>
    Object.entries(s["x-ui"]?.visible_when ?? {}).every(([k, vals]) => {
      const cur = node.params[k] ?? props[k]?.default;
      return vals.map(String).includes(String(cur));
    }) || fieldErrors(name).length > 0;
  const params = Object.entries(props)
    .filter(([name, s]) => applies(name, s))
    .sort(([, a], [, b]) => (a["x-ui"]?.order ?? 99) - (b["x-ui"]?.order ?? 99));
  // "Chạy thử": a model with a fixed task (Nemotron-Parse) ignores the question, so do not ask it (B-7).
  const fixedPrompt = upstreamModelMeta(workflow, specs, node.id)?.task_prompt_fixed === true;
  const simple = params.filter(
    ([name, s]) =>
      !s["x-ui"]?.advanced &&
      !(fixedPrompt && s["x-ui"]?.widget === "textarea" && name === "prompt"),
  );
  const advanced = params.filter(([, s]) => s["x-ui"]?.advanced);
  // Advanced mode: everything visible + technical id. Simple: advanced params behind a per-node toggle,
  // but changes made there stay visible as a badge (P010 §10 — no hidden changes).
  const showAdvanced = mode === "advanced" || expanded;
  const customized = customizedAdvanced(spec, node.params);
  // Measured time of this train step (estimate.duration.per_node): "~1 giờ 37 phút – 1 giờ 52 phút · 300 step".
  const nodeEst = validation?.estimate.duration?.per_node[node.id];

  return (
    <aside className="inspector" aria-label={nodeTitle(spec, node.type)}>
      {skin === "d4" ? (
        <div className="insp-head">
          <h2>{nodeTitle(spec, node.type)}</h2>
          <button
            type="button"
            className="help-btn"
            popoverTarget="help-node"
            aria-label={t("d4.explain")}
          >
            <Icon name="circle-question-mark" size={14} />
          </button>
          <div id="help-node" popover="auto" className="pop">
            {nodeDesc(spec, node.type)}
          </div>
        </div>
      ) : (
        <>
          <h2>{nodeTitle(spec, node.type)}</h2>
          <p className="muted">{nodeDesc(spec, node.type)}</p>
        </>
      )}
      {nodeEst && (
        <p className="node-est" data-testid="node-estimate">
          <Icon name="timer" size={14} /> {formatMinutesRange(nodeEst.minutes)}
          {/* Simple mode: time only; a raw step count is nothing a beginner can act on (design) */}
          {mode === "advanced" && nodeEst.steps != null
            ? ` · ${t("inspector.steps", { step_count: nodeEst.steps })}`
            : ""}
        </p>
      )}
      {mode === "advanced" && (
        <p className="small">
          <code>
            {node.type}@{spec.version}
          </code>
        </p>
      )}
      {run?.nodes[node.id]?.nodeBase && (
        // G-4: keyboard path to "Đặt lại bước" (the canvas also has it on right-click)
        <button
          type="button"
          className="reset-step-btn"
          title={t("storage.reset_step.menu_hint")}
          onClick={() => useUi.getState().setResetNode(run.nodes[node.id]?.nodeBase ?? null)}
        >
          <Icon name="rotate-ccw" size={14} /> {t("storage.reset_step.menu")}
        </button>
      )}
      {mode === "simple" && customized.length > 0 && (
        <p className="badge customized">
          ⚙ {t("canvas.customized", { param_count: customized.length })}
        </p>
      )}

      {issues.length > 0 && (
        <ul className="error-cards">
          {issues.map((e) => (
            <li
              key={`${e.code}-${e.port}-${e.param}-${e.what}`}
              className={`error-card ${e.severity}`}
              data-code={e.code}
            >
              <ErrorIcon
                tone={e.severity === "error" ? "critical" : e.severity === "info" ? "info" : "warn"}
              />
              <strong>
                {SEVERITY_GLYPH[e.severity]} {e.title}
              </strong>
              <p>{e.what}</p>
              {e.why && (
                <p className="muted small">
                  {t("error_card.why")}: {e.why}
                </p>
              )}
              {(e.port || e.param) && (
                <button
                  type="button"
                  className="link"
                  onClick={() =>
                    document.getElementById(e.port ? `recv-${e.port}` : `param-${e.param}`)?.focus()
                  }
                >
                  {t("error_card.fix")} →
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      {Object.entries(spec.inputs).map(([port, ps]) => {
        const edge = workflow.edges.find((e) => e.to[0] === node.id && e.to[1] === port);
        const options = sourcesFor(workflow, specs, ports, node.id, ps.type);
        const id = `recv-${port}`;
        const pt = ports.find((p) => p.id === ps.type);
        return (
          <div className="field" key={port}>
            <label htmlFor={id}>
              {t("canvas.receive_from")} —{" "}
              {tpOptional(["node", node.type, "port", port]) ??
                (pt ? tp(pt.label_key.split(".")) : ps.type)}
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
        <ParamRow
          errors={fieldErrors(name)}
          key={name}
          nodeType={node.type}
          name={name}
          schema={schema}
          value={node.params[name]}
          onChange={(v) => setParam(node.id, name, v)}
          optionMinutes={name === "epochs" ? nodeEst?.by_epochs : undefined}
        />
      ))}
      {advanced.length > 0 && mode === "simple" && (
        <button
          type="button"
          className="link"
          aria-expanded={showAdvanced}
          onClick={() => setExpanded((s) => !s)}
        >
          ▸ {t("canvas.show_advanced", { param_count: advanced.length })}
        </button>
      )}
      {showAdvanced &&
        advanced.map(([name, schema]) => (
          <ParamRow
            errors={fieldErrors(name)}
            key={name}
            nodeType={node.type}
            name={name}
            schema={schema}
            value={node.params[name]}
            onChange={(v) => setParam(node.id, name, v)}
            optionMinutes={name === "epochs" ? nodeEst?.by_epochs : undefined}
          />
        ))}
      {node.type.startsWith("serve.") && run?.nodes[node.id]?.status === "completed" && (
        <TrialResult
          runId={run.id}
          nodeId={node.id}
          imageId={typeof node.params.image === "string" ? node.params.image : undefined}
        />
      )}
    </aside>
  );
}
