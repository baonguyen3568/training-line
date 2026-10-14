import { type ReactNode, useState } from "react";
import type { ResolvedAction, ValidationIssue } from "../api/types";
import { t, tpOptional } from "../i18n";
import { downloadTargetOf } from "../lib/downloadTarget";
import { isModelType, nearestUpstream } from "../lib/graph";
import { requestRun } from "../lib/requestRun";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { ISSUES_URL } from "./BugReportDialog";
import { Icon } from "./Icon";

/** Ask the editor to show a toast (it owns the one toast slot). */
export const toast = (message: string) =>
  window.dispatchEvent(new CustomEvent("vlm:toast", { detail: message }));
const navigate = (to: string) =>
  window.dispatchEvent(new CustomEvent("vlm:navigate", { detail: to }));

type Params = Record<string, unknown>;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Label action.<id>, filled from the issue's params; undefined when a placeholder is missing (never "{est_gb}"). */
function label(a: ResolvedAction, params: Params): string | undefined {
  const text = tpOptional(["action", a.id], params as Record<string, string | number>);
  return text === undefined || /\{\w+\}/.test(text) ? undefined : text;
}

/**
 * Graph operations (kind "patch", patch null) the client knows how to do; any other id gets no button (fixcode:
 * never show a button that does nothing).
 */
function graphOp(
  a: ResolvedAction,
  issue: ValidationIssue,
  nodeId: string,
): (() => void) | undefined {
  const s = useWorkflow.getState();
  if (a.id === "remove_step") return () => s.removeNodes([nodeId]);
  if (a.id === "remove_bad_wire" && issue.port) {
    const port = issue.port;
    return () => s.disconnect([nodeId, port]);
  }
  if (a.id === "reset_default" && issue.param) {
    const param = issue.param;
    const node = s.workflow?.nodes.find((n) => n.id === nodeId);
    const def = node && s.specs.get(node.type)?.params_schema.properties[param]?.default;
    if (def === undefined) return undefined;
    return () => s.setParam(nodeId, param, def);
  }
  return undefined;
}

const isModel = isModelType;
const isSource = (type: string) =>
  isModel(type) || type === "data.hf_load" || type === "data.local_folder";

/** Select a node, then focus a control of its inspector once it has rendered. */
function selectAndFocus(nodeId: string, elementId?: string) {
  useWorkflow.getState().select(nodeId);
  if (elementId) requestAnimationFrame(() => document.getElementById(elementId)?.focus());
}

/**
 * Navigate targets of the catalog (fixcode review): node:self the failing step; node:model the model feeding the
 * failing step (upstream, else the workflow's first model); node:port the failing step's "Nhận từ" picker for
 * issue.port; node:asset the source step that brought the asset in. A target that resolves to nothing gets no
 * button — never a button that does nothing.
 */
function navigation(
  a: ResolvedAction,
  issue: ValidationIssue,
  nodeId: string,
): (() => void) | undefined {
  const s = useWorkflow.getState();
  const target = a.target ?? "";
  if (target === "node:self") return () => selectAndFocus(nodeId);
  if (target === "node:model") {
    const find = () =>
      nearestUpstream(useWorkflow.getState().workflow, nodeId, isModel) ??
      useWorkflow.getState().workflow?.nodes.find((n) => isModel(n.type))?.id;
    if (!find()) return undefined;
    return () => {
      const hit = find(); // resolve again on click: wires may have changed since render
      if (hit) selectAndFocus(hit);
    };
  }
  if (target === "node:port") {
    const port = issue.port;
    return port ? () => selectAndFocus(nodeId, `recv-${port}`) : undefined;
  }
  if (target === "node:asset") {
    if (!nearestUpstream(useWorkflow.getState().workflow, nodeId, isSource)) return undefined;
    return () => {
      const hit = nearestUpstream(useWorkflow.getState().workflow, nodeId, isSource);
      if (hit) selectAndFocus(hit);
    };
  }
  if (target === "memory_panel" || target === "cleanup_dry_run") return () => navigate("storage");
  if (target === "queue_until_memory") return requestRun; // the run waits for memory by itself
  // ADR-021 targets (fixcode): the repo's own page, a catalog request, the column mapping, the licence dialog
  const repo = typeof issue.params.repo === "string" ? issue.params.repo : "";
  if (target === "hub_page")
    return /^[\w.-]+\/[\w.-]+$/.test(repo)
      ? () => window.open(`https://huggingface.co/${repo}`, "_blank", "noopener,noreferrer")
      : undefined;
  if (target === "request_catalog")
    return () => window.open(ISSUES_URL, "_blank", "noopener,noreferrer"); // no query: the user writes it
  if (target === "column_mapping") return () => selectAndFocus(nodeId);
  if (target === "license_dialog") {
    const t0 = downloadTargetOf(
      { ...issue, code: "NET-002", params: { reason: "not_cached", ...issue.params } },
      s.specs,
    );
    return t0 ? () => useUi.getState().setDownloadTarget(t0) : undefined;
  }
  if (target === "settings_secrets" || target === "settings_network")
    return () => navigate("settings");
  if (target === "bug_report") {
    // with a run the report carries its reference code; validate/preflight errors report without one (design)
    const runId = s.run?.id ?? null;
    return () => useUi.getState().setBugReport({ runId, nodeId, code: issue.code });
  }
  return undefined;
}

/**
 * G-3 "Sửa nhanh" on an error card: the server's resolved actions as buttons. A param patch applies only while
 * the node still holds the values the server saw (`from`); confirm=true previews the change first. One Undo
 * (Ctrl+Z) restores a patch.
 */
export function QuickFixes({ issue, nodeId }: { issue: ValidationIssue; nodeId: string }) {
  // the whole workflow: buttons depend on wires too (node:model / node:asset walk them)
  const workflow = useWorkflow((s) => s.workflow);
  const node = workflow?.nodes.find((n) => n.id === nodeId);
  const spec = useWorkflow((s) => (node ? s.specs.get(node.type) : undefined));
  const [preview, setPreview] = useState<string | null>(null);
  if (!issue.actions?.length || !node) return null;
  const current = (param: string) =>
    node.params[param] ?? spec?.params_schema.properties[param]?.default;

  // Preview in people's words (design G-3 r1): the param's label (id after it), the option's copy; raw ids and
  // numbers only as code.
  const paramText = (param: string): ReactNode => {
    const name = tpOptional(["node", node.type, "param", param, "label"]);
    return name ? (
      <>
        {name} (<code>{param}</code>)
      </>
    ) : (
      <code>{param}</code>
    );
  };
  const valueText = (param: string, v: unknown): ReactNode => {
    const keys = spec?.params_schema.properties[param]?.["x-ui"]?.option_keys;
    const copy = keys ? tpOptional([...keys.split("."), String(v)]) : undefined;
    return copy ?? <code>{String(v)}</code>;
  };

  const buttons = issue.actions.flatMap((a) => {
    const text = label(a, issue.params);
    if (!text) return [];
    let run: (() => void) | undefined;
    if (a.kind === "patch" && a.patch) {
      const patch = a.patch;
      if (!patch.every((p) => same(current(p.param), p.from))) return []; // stale hint: wait for the next validate
      run = () => {
        useWorkflow
          .getState()
          .setParams(a.node_id ?? nodeId, Object.fromEntries(patch.map((p) => [p.param, p.value])));
        toast(t("quick_fix.applied", { action: text }));
        setPreview(null);
      };
      if (a.confirm) {
        const apply = run;
        return [{ a, text, run: () => setPreview(a.id), apply }];
      }
    } else if (a.kind === "patch") {
      run = graphOp(a, issue, a.node_id ?? nodeId);
    } else if (a.kind === "navigate") {
      run = navigation(a, issue, a.node_id ?? nodeId);
    } else if (a.kind === "trial") {
      // "Chạy thử nhanh": a separate 5-step run of this step; nothing in the workflow changes (backend H5)
      const node = a.node_id ?? nodeId;
      run = () => void useWorkflow.getState().startRun({ node });
    } else if (a.kind === "resume" && !a.patch) {
      // P017: posting the same workflow again resumes from the last valid checkpoint (same cache key)
      run = requestRun;
    }
    // resume WITH a patch (resume_lighter): changed params give a new cache key, so the step starts over; its label
    // ("continue…") would not be true, so no button until the copy says so (backend 2026-10-13)
    return run ? [{ a, text, run, apply: run }] : [];
  });
  if (buttons.length === 0) return null;
  const previewed = buttons.find((b) => b.a.id === preview);

  return (
    <div className="quick-fixes">
      {buttons.map(({ a, text, run }) => (
        <button key={a.id} type="button" className="quick-fix" onClick={run} data-action={a.id}>
          <Icon name={a.kind === "trial" ? "play" : "wrench"} size={14} /> {text}
        </button>
      ))}
      {previewed?.a.patch && (
        <fieldset className="quick-fix-preview">
          <legend className="sr-only">{previewed.text}</legend>
          <ul>
            {previewed.a.patch.map((p) => (
              <li key={p.param}>
                {paramText(p.param)}: {valueText(p.param, p.from)} → {valueText(p.param, p.value)}
              </li>
            ))}
          </ul>
          <button type="button" className="hairline" onClick={() => setPreview(null)}>
            {t("storage.dialog.cancel")}
          </button>{" "}
          <button type="button" className="primary" onClick={previewed.apply}>
            {t("quick_fix.apply")}
          </button>
        </fieldset>
      )}
    </div>
  );
}
