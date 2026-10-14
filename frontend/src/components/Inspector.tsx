import { type ReactNode, useEffect, useState } from "react";
import type {
  NodeSpec,
  OptionMeta,
  ParamSchema,
  PortType,
  Range,
  Workflow,
  WorkflowNode,
} from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { downloadTargetOf } from "../lib/downloadTarget";
import {
  durationRunCount,
  formatEstimate,
  formatMinutesRange,
  workflowDurationText,
} from "../lib/format";
import { copyOwner, SEPARATOR } from "../lib/groups";
import { issuesAt, issueTrail } from "../lib/groupView";
import { withCode } from "../lib/machineText";
import { nodeDesc, nodeTitle, workflowTitle } from "../lib/nodeText";
import { openInnerStep } from "../lib/openInner";
import { customizedAdvanced } from "../lib/params";
import { isCompatible } from "../lib/ports";
import { runErrorText } from "../lib/runWarning";
import { selfAndDownstream } from "../lib/topo";
import { useSystem } from "../store/system";
import { useUi } from "../store/ui";
import { useUserModels } from "../store/userModels";
import { useWorkflow } from "../store/workflow";
import { AdvancedOptions } from "./AdvancedOptions";
import { DatasetHubPanel, setUserDataset, USER_DATASET_PARAMS } from "./DatasetSource";
import { ErrorIcon } from "./ErrorIcon";
import { FieldError } from "./FieldError";
import { Icon } from "./Icon";
import { ImageUpload } from "./ImageUpload";
import { ModelHubPanel, SourceTabs, setUserModel, USER_MODEL_PARAMS } from "./ModelSource";
import { QuickFixes } from "./QuickFixes";
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
          ? t("model_picker.needs_free_ram", { need_gb: need, avail_gb: Math.round(availGb) }) // copy says it (design P3)
          : t("model_picker.fits"),
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
  invalidBy,
}: {
  nodeType: string;
  name: string;
  schema: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  /** Measured time per option (estimate per_node.by_epochs), shown next to each option. */
  optionMinutes?: Record<string, Range | null>;
  /** Ids of this field's error lines (ParamRow): the field is marked invalid and read with them. */
  invalidBy?: string;
}) {
  // a group's exposed param reads the copy of the inner step that owns it (P4-4)
  const owner = copyOwner(nodeType, name, "param", useWorkflow.getState().specs);
  const label = tp(["node", owner.typeId, "param", owner.name, "label"]);
  const id = `param-${name}`;
  const invalid = invalidBy
    ? ({ "aria-invalid": true, "aria-describedby": invalidBy } as const)
    : undefined;
  const mode = useUi((s) => s.mode);
  const availGb = useSystem((s) => s.status?.mem_available_gb);
  const current = value ?? schema.default;
  const widget = schema["x-ui"]?.widget;
  // P016 §3 `advanced` (engine options, object): its own widget fed by /api/node-types/{id}/advanced is not built
  // yet; never render an object as a text box ("[object Object]") — the value is kept untouched.
  if ((widget as string | undefined) === "advanced" || schema.type === "object") return null;
  if (widget === "image_upload") {
    return <ImageUpload id={id} label={label} value={String(current ?? "")} onChange={onChange} />;
  }
  if (widget === "textarea") {
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <textarea
          id={id}
          {...invalid}
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
      <fieldset className="field" aria-describedby={invalidBy}>
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
          {...invalid}
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
          aria-invalid={invalidBy ? true : undefined}
          aria-describedby={
            [off !== undefined ? `${id}-unsupported` : "", invalidBy ?? ""]
              .filter(Boolean)
              .join(" ") || undefined
          }
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
          {...invalid}
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
        {...invalid}
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
  const helpOwner = copyOwner(props.nodeType, props.name, "param", useWorkflow.getState().specs);
  const help = tpOptional(["node", helpOwner.typeId, "param", helpOwner.name, "help"]);
  const d4 = useUi((s) => s.skin) === "d4";
  const popId = `help-${props.name}`;
  const errId = (k: number) => `param-${props.name}-err-${k}`;
  const shown = d4 ? errors : [];
  return (
    <div className="param-row">
      <ParamField
        {...field}
        invalidBy={shown.length ? shown.map((_, k) => errId(k)).join(" ") : undefined}
      />
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
      {shown.map((e, k) => (
        <FieldError key={e} id={errId(k)}>
          {e}
        </FieldError>
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
/**
 * P4-4: inside a group with nothing selected, a summary of that group, not of the workflow outside: its title,
 * steps, the errors inside it and the view-only line. No time/RAM/disk until a group has its own (design).
 */
function GroupSummary() {
  const { workflow, validation, specs } = useWorkflow();
  const path = useUi((st) => st.groupPath);
  if (!workflow || path.length === 0) return null;
  let nodes = workflow.nodes;
  let spec: NodeSpec | undefined;
  for (const id of path) {
    const n = nodes.find((x) => x.id === id);
    spec = n && specs.get(n.type);
    if (!spec?.group) return null;
    nodes = spec.group.body.nodes;
  }
  if (!spec?.group) return null;
  const groupId = path[path.length - 1] as string;
  const inner = issuesAt(validation?.errors ?? [], path.slice(0, -1), groupId).filter(
    (e) => e.inner_path?.length && e.severity === "error",
  );
  const title = nodeTitle(spec, spec.type_id);
  return (
    <aside className="inspector wf-summary" aria-label={title}>
      <p className="wf-title">{title}</p>
      <p>{t("d4.step_count", { step_count: spec.group.body.nodes.length })}</p>
      <p>
        {inner.length === 0 ? (
          <>
            <span className="si si-completed">
              <Icon name="circle-check-big" size={14} />
            </span>{" "}
            {t("preflight.summary_ok")}
          </>
        ) : (
          <>
            <span className="si si-bad">
              <Icon name="circle-x" size={14} />
            </span>{" "}
            {inner.length === 1
              ? t("canvas.group.inner_errors_one")
              : t("canvas.group.inner_errors", { error_count: inner.length })}
          </>
        )}
      </p>
      <p className="inner-readonly small" role="note">
        <span className="si si-muted" aria-hidden="true">
          <Icon name="lock" size={14} />
        </span>{" "}
        {t(spec.group.user ? "canvas.group.readonly_user" : "canvas.group.readonly")}
      </p>
    </aside>
  );
}

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
      {/* the workflow name is already the page h1 (toolbar): not a second heading here (design D7 r2) */}
      <p className="wf-title">{workflowTitle(workflow.name)}</p>
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

/**
 * P4-4 §2: a step inside an entered group, view-only in 1.0 (design 818e7c3). Its settings with their values; one
 * the group exposes says "set on the group step" with a way back there; its own issues and run error.
 */
function InnerStepInspector({ flat }: { flat: string }) {
  const { workflow, specs, validation, run } = useWorkflow();
  const path = flat.split(SEPARATOR);
  // walk down: each id but the last is a group node, the last is the step
  let nodes = workflow?.nodes ?? [];
  let parentSpec: NodeSpec | undefined;
  let step: WorkflowNode | undefined;
  for (const [k, id] of path.entries()) {
    const n = nodes.find((x) => x.id === id);
    if (!n) break;
    if (k === path.length - 1) step = n;
    else {
      parentSpec = specs.get(n.type);
      nodes = parentSpec?.group?.body.nodes ?? [];
    }
  }
  const spec = step && specs.get(step.type);
  if (!step || !spec || !parentSpec?.group) return null;
  const exposed = new Map(
    Object.entries(parentSpec.group.body.params)
      .filter(([, [node]]) => node === step.id)
      .map(([name, [, param]]) => [param, name]),
  );
  const issues = issuesAt(validation?.errors ?? [], path.slice(0, -1), step.id).filter(
    (i) => !i.inner_path?.length,
  );
  const runError = run?.nodes[flat]?.status === "failed" ? run.nodes[flat]?.error : undefined;
  const props = Object.entries(spec.params_schema.properties)
    .filter(([, sc]) => sc.type !== "object")
    .sort(([, a], [, b]) => (a["x-ui"]?.order ?? 99) - (b["x-ui"]?.order ?? 99));
  const backToGroup = () => {
    const parent = path.slice(0, -1);
    useUi.getState().setGroupPath(parent.slice(0, -1));
    useWorkflow.getState().select(parent.join(SEPARATOR));
  };
  return (
    <aside className="inspector inner-step" aria-label={nodeTitle(spec, step.type)}>
      <div className="insp-head">
        <h2>{nodeTitle(spec, step.type)}</h2>
      </div>
      <p className="muted small">{nodeDesc(spec, step.type)}</p>
      <p className="inner-readonly small" role="note">
        <span className="si si-muted" aria-hidden="true">
          <Icon name="lock" size={14} />
        </span>{" "}
        {t(parentSpec.group.user ? "canvas.group.readonly_user" : "canvas.group.readonly")}
      </p>
      {runError && (
        <div className="error-card" role="alert" data-code={runError.code}>
          <ErrorIcon tone="critical" />
          <strong>{runErrorText(runError, (id) => id).title}</strong>
        </div>
      )}
      {issues.length > 0 && (
        <ul className="error-cards">
          {issues.map((e) => (
            <li
              key={`${e.code}-${e.param}-${e.what}`}
              className={`error-card ${e.severity}`}
              data-code={e.code}
            >
              <ErrorIcon tone={e.severity === "error" ? "critical" : "warn"} />
              <strong>{e.title}</strong>
              <p>{e.what}</p>
            </li>
          ))}
        </ul>
      )}
      <dl className="inner-params">
        {props.map(([name, sc]) => {
          const value = step.params[name] ?? sc.default;
          const shown =
            value === undefined || value === null
              ? "—"
              : sc.enum
                ? optionLabel(sc, value as string | number)
                : String(value);
          const exposedAs = exposed.get(name);
          return (
            <div key={name} className="inner-param" id={`param-${name}`} tabIndex={-1}>
              <dt>{tp(["node", step.type, "param", name, "label"])}</dt>
              <dd>
                {sc.type === "boolean" ? (
                  // a yes/no setting reads as the same control, disabled: no "true"/"false" text
                  <input
                    type="checkbox"
                    checked={value === true}
                    disabled
                    aria-label={tp(["node", step.type, "param", name, "label"])}
                  />
                ) : (
                  shown
                )}
                {exposedAs && (
                  <span className="small muted">
                    {" · "}
                    <button type="button" className="link" onClick={backToGroup}>
                      {t("canvas.group.param_exposed")}
                    </button>
                  </span>
                )}
              </dd>
            </div>
          );
        })}
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
  const [hubTab, setHubTab] = useState(false);
  // MODEL-006 "Thêm model này": the step's hub tab opens with the repo typed in (the panel takes the prefill)
  const prefillFor = useUserModels((st) => st.prefill?.nodeId);
  useEffect(() => {
    if (prefillFor && prefillFor === selectedId) setHubTab(true);
  }, [prefillFor, selectedId]);
  const groupPath = useUi((st) => st.groupPath);
  const node = workflow?.nodes.find((n) => n.id === selectedId);
  // a step inside a group (flat id "g__step"): its own view-only inspector
  if (workflow && !node && selectedId?.includes(SEPARATOR))
    return <InnerStepInspector flat={selectedId} />;
  if (!workflow || !node) {
    if (skin === "d4" && workflow)
      return groupPath.length > 0 ? <GroupSummary /> : <WorkflowSummary />;
    return <aside className="inspector muted">{t("inspector.empty_hint")}</aside>;
  }
  const spec = specs.get(node.type);
  if (!spec) return <aside className="inspector">{t("error.CFG-004.title")}</aside>;
  const allIssues = validation?.errors.filter((e) => e.node_id === node.id) ?? [];
  // a group: its own problems as usual; its inner steps' errors in their own block, with a way to each (P4-4 §3)
  const issues = allIssues.filter((e) => !e.inner_path?.length);
  const innerIssues = allIssues.filter((e) => e.inner_path?.length && e.severity === "error");
  const runState = run?.nodes[node.id];
  const runError = runState?.status === "failed" ? runState.error : undefined;
  const runErrorTexts = runError
    ? runErrorText(runError, (id) => {
        const n = workflow.nodes.find((x) => x.id === id);
        return n ? nodeTitle(specs.get(n.type), n.type) : id;
      })
    : undefined;
  const fieldErrors = (param: string) => issues.filter((e) => e.param === param).map((e) => e.what);

  // x-ui.visible_when: a field that does not apply right now is hidden but keeps its value, and is shown anyway
  // while it has a validation error (never hide a problem).
  const props = spec.params_schema.properties;
  const applies = (name: string, s: ParamSchema) =>
    Object.entries(s["x-ui"]?.visible_when ?? {}).every(([k, vals]) => {
      const cur = node.params[k] ?? props[k]?.default;
      return vals.map(String).includes(String(cur));
    }) || fieldErrors(name).length > 0;
  const modelTab: "catalog" | "hub" =
    hubTab || node.params.user_model || prefillFor === node.id ? "hub" : "catalog";
  const dataTab: "catalog" | "hub" =
    hubTab || node.params.user_dataset || prefillFor === node.id ? "hub" : "catalog";
  const params = Object.entries(props)
    // the catalog choice lives in the "Đã kiểm chứng" tab only; `user_model` is set by the hub panel, never typed
    // `x-ui.widget: "hidden"` (backend H3): never a row, the value stays and its errors still show in the check bar
    .filter(([, schema]) => schema["x-ui"]?.widget !== "hidden")
    .filter(([name]) =>
      node.type === "model.hf_pick"
        ? !USER_MODEL_PARAMS.includes(name) && !(modelTab === "hub" && name === "model")
        : node.type === "data.hf_load"
          ? // the hub panel sets user_dataset (+ hidden repo/revision); the catalog tab keeps its repo row
            !USER_DATASET_PARAMS.includes(name) && !(dataTab === "hub" && name === "repo")
          : true,
    )
    // P016 `advanced` (object) has no widget yet: no row at all, not even its "?" help
    .filter(([, s]) => s["x-ui"]?.widget !== "advanced" && s.type !== "object")
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
      {node.type === "model.hf_pick" && (
        // ADR-021 / spec H §1: a verified catalog model, or one the user adds from Hugging Face
        <>
          <SourceTabs
            value={modelTab}
            onChange={(v) => {
              setHubTab(v === "hub");
              if (v === "catalog" && node.params.user_model) setUserModel(node.id, null);
            }}
          />
          {modelTab === "hub" && <ModelHubPanel nodeId={node.id} />}
        </>
      )}
      {node.type === "data.hf_load" && (
        // ADR-021 / spec H §4: the catalog dataset, or any Hugging Face dataset with its columns mapped
        <>
          <SourceTabs
            value={dataTab}
            onChange={(v) => {
              setHubTab(v === "hub");
              // back to the catalog dataset: all three user params cleared together (one undo step)
              if (v === "catalog" && node.params.user_dataset) setUserDataset(node.id, null);
            }}
          />
          {dataTab === "hub" && <DatasetHubPanel nodeId={node.id} />}
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
          <Icon name="sliders-horizontal" size={14} />{" "}
          {t("canvas.customized", { param_count: customized.length })}
        </p>
      )}

      {spec.group && innerIssues.length > 0 && (
        <div className="error-card group-inner" role="alert" data-testid="group-inner-issues">
          <ErrorIcon tone="critical" />
          <strong>
            {innerIssues.length === 1
              ? t("canvas.group.inner_errors_one")
              : t("canvas.group.inner_errors", { error_count: innerIssues.length })}
          </strong>
          <ul>
            {innerIssues.map((e) => {
              const trail = issueTrail(workflow, specs, e, nodeTitle);
              return (
                <li key={`${e.inner_path?.join("/")}-${e.code}-${e.param}-${e.port}`}>
                  <span className="issue-trail">{trail?.labels.slice(1).join(" › ")}</span>
                  {" · "}
                  {e.title}{" "}
                  {trail && (
                    <button
                      type="button"
                      className="link"
                      onClick={() => openInnerStep(trail, { param: e.param, port: e.port })}
                    >
                      {t("action.open_inner_step")}
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {spec.group && mode === "advanced" && (
        <div className="group-inside">
          <p className="small muted">{t("canvas.group.inner_title")}</p>
          <ul className="small">
            {spec.group.body.nodes.map((n) => (
              <li key={n.id}>{nodeTitle(specs.get(n.type), n.type)}</li>
            ))}
          </ul>
        </div>
      )}
      {runError && (
        // the step failed while running: say why and how to fix it, like a pre-flight issue
        <div className="error-card" role="alert" data-code={runError.code}>
          <ErrorIcon tone="critical" />
          <strong>{runErrorTexts?.title}</strong>
          {runErrorTexts?.what && <p>{withCode(runErrorTexts.what, runError.params ?? {})}</p>}
          {runErrorTexts?.why && (
            <p className="muted small">
              {t("error_card.why")}: {runErrorTexts.why}
            </p>
          )}
          <DownloadButton issue={runError} specs={specs} />
          {runError.already_optimized && (
            <p className="small">{t("error_card.already_optimized")}</p>
          )}
          <QuickFixes
            nodeId={node.id}
            issue={{
              node_id: node.id,
              code: runError.code,
              severity: "error",
              params: (runError.params ?? {}) as Record<string, string | number>,
              title: runErrorTexts?.title ?? runError.code,
              what: runErrorTexts?.what ?? "",
              action_ids: runError.action_ids ?? [],
              ...(runError.actions ? { actions: runError.actions } : {}),
            }}
          />
        </div>
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
              {/* the icon carries the severity: no second glyph before the title (design G-5 #2) */}
              <strong>{e.title}</strong>
              <p>{withCode(e.what, e.params)}</p>
              <DownloadButton issue={e} specs={specs} />
              <QuickFixes issue={e} nodeId={node.id} />
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
          <Icon name={showAdvanced ? "chevron-down" : "chevron-right"} size={14} />{" "}
          {t("canvas.show_advanced", { param_count: advanced.length })}
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
      {showAdvanced && props.advanced?.["x-ui"]?.widget === "advanced" && (
        // P016 §3 engine options: their own panel, Advanced only (the generic rows skip this object param)
        <section className="field adv-field" aria-labelledby="adv-label">
          <h3 id="adv-label" className="adv-title">
            {tp(["node", node.type, "param", "advanced", "label"])}
            {tpOptional(["node", node.type, "param", "advanced", "help"]) && (
              <>
                <button
                  type="button"
                  className="help-btn inline"
                  popoverTarget="help-advanced-panel"
                  aria-label={t("d4.explain")}
                >
                  <Icon name="circle-question-mark" size={14} />
                </button>
                <span id="help-advanced-panel" popover="auto" className="pop">
                  {tpOptional(["node", node.type, "param", "advanced", "help"])}
                </span>
              </>
            )}
          </h3>
          <AdvancedOptions nodeId={node.id} typeId={node.type} issues={issues} />
        </section>
      )}
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

/** G-5: "Tải về máy" on a NET-002 card (pre-flight or run error); nothing for other codes. */
function DownloadButton({
  issue,
  specs,
}: {
  issue: { code: string; params?: Record<string, unknown> | undefined };
  specs: Map<string, NodeSpec>;
}) {
  const target = downloadTargetOf(issue, specs);
  if (!target) return null;
  return (
    <p>
      <button
        type="button"
        className="primary"
        onClick={() => useUi.getState().setDownloadTarget(target)}
      >
        <Icon name="download" size={14} /> {t("action.download")}
      </button>
    </p>
  );
}
