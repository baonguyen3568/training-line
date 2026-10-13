import { useReactFlow } from "@xyflow/react";
import { t } from "../i18n";
import { workflowTitle } from "../lib/nodeText";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { gpuChip, useSystemPoll } from "./Header";
import { Icon } from "./Icon";
import { IconButton } from "./IconButton";
import { useWorkflowFile } from "./Toolbar";

/**
 * D4 top toolbar, 48px, icon-only (DESIGN.md Layout): mark + workflow name | file | edit | view | mode |
 * resource chips | Run/Stop, the only labelled control. Same actions and gating as the classic Toolbar/CheckBar.
 */
export function ToolbarD4({ locked }: { locked: boolean }) {
  const { workflow, selectedId, past, future, validation, run } = useWorkflow();
  const { newWorkflow, undo, redo, duplicateNode, removeNodes, startRun, cancelRun } =
    useWorkflow.getState();
  const mode = useUi((s) => s.mode);
  const setMode = useUi((s) => s.setMode);
  const status = useSystemPoll(); // the classic header is not mounted in the D4 editor
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const file = useWorkflowFile();
  if (!workflow) return null;

  const errors = validation?.errors.filter((e) => e.severity === "error") ?? [];
  const running = !!run && !run.done;
  const canRun = !!validation && errors.length === 0 && workflow.nodes.length > 0;
  const advanced = mode === "advanced";
  const gpu = gpuChip(status?.gpu_lock);

  return (
    <header className="tb" role="toolbar" aria-label={t("app.name")}>
      <button
        type="button"
        className="tb-brand"
        aria-label={t("app.nav_workflows")}
        data-tip={t("app.nav_workflows")}
        onClick={() => window.dispatchEvent(new CustomEvent("vlm:navigate", { detail: "home" }))}
      >
        <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
          <rect width="32" height="32" rx="8" fill="var(--color-accent-brand)" />
          <path
            d="M10.25 22.5V9.5l11.5 13V9.5"
            fill="none"
            stroke="#fff"
            strokeWidth="3.2"
            strokeLinecap="square"
          />
        </svg>
      </button>
      <span className="tb-name" title={workflowTitle(workflow.name)}>
        {workflowTitle(workflow.name)}
      </span>
      <span className="tb-sep" />
      <IconButton
        icon="file-plus"
        label={t("gallery.blank_workflow")}
        onClick={newWorkflow}
        disabled={locked}
      />
      <IconButton
        icon="folder-open"
        label={t("workflow.open")}
        onClick={file.pick}
        disabled={locked}
      />
      {file.input}
      <IconButton icon="save" label={t("workflow.save")} shortcut="Ctrl+S" onClick={file.save} />
      <span className="tb-sep" />
      <IconButton
        icon="undo-2"
        label={t("app.undo")}
        shortcut="Ctrl+Z"
        onClick={undo}
        disabled={locked || past.length === 0}
      />
      <IconButton
        icon="redo-2"
        label={t("app.redo")}
        shortcut="Ctrl+Shift+Z"
        onClick={redo}
        disabled={locked || future.length === 0}
      />
      <IconButton
        icon="copy"
        label={t("workflow.duplicate")}
        shortcut="Ctrl+D"
        onClick={() => selectedId && duplicateNode(selectedId)}
        disabled={locked || !selectedId}
      />
      <IconButton
        icon="x"
        label={t("workflow.delete")}
        shortcut="Delete"
        onClick={() => selectedId && removeNodes([selectedId])}
        disabled={locked || !selectedId}
      />
      <span className="tb-sep" />
      <IconButton
        icon="scan"
        label={t("d4.fit_view")}
        onClick={() => void fitView({ maxZoom: 1, padding: 0.1 })}
      />
      <IconButton icon="zoom-out" label={t("d4.zoom_out")} onClick={() => void zoomOut()} />
      <IconButton icon="zoom-in" label={t("d4.zoom_in")} onClick={() => void zoomIn()} />
      <span className="tb-grow" />
      <IconButton
        icon="wrench"
        label={advanced ? t("d4.mode_to_simple") : t("d4.mode_to_advanced")}
        aria-pressed={advanced}
        onClick={() => setMode(advanced ? "simple" : "advanced")}
      />
      <span className="tb-sep" />
      <span className="chip" data-tip={gpu.text}>
        <span className={`led ${gpu.state}`} />
        <Icon name="cpu" size={14} />
        <span aria-hidden="true">GPU</span>
        <span className="sr-only">{gpu.text}</span>
      </span>
      {status && (
        <>
          <span
            className="chip mono"
            data-tip={t("header.ram_available", { avail_gb: Math.round(status.mem_available_gb) })}
          >
            <Icon name="memory-stick" size={14} />
            {t("header.ram", {
              used_gb: Math.round(status.mem_total_gb - status.mem_available_gb),
              total_gb: Math.round(status.mem_total_gb),
            }).replace(/^RAM\s*/, "")}
          </span>
          <span
            className="chip mono"
            data-tip={t("header.disk_free", { free_gb: Math.round(status.disk_free_gb) })}
          >
            <Icon name="hard-drive" size={14} />
            {Math.round(status.disk_free_gb)} GB
          </span>
        </>
      )}
      <IconButton
        icon="settings"
        label={t("settings.title")}
        onClick={() =>
          window.dispatchEvent(new CustomEvent("vlm:navigate", { detail: "settings" }))
        }
      />
      <span className="tb-sep" />
      {running ? (
        <button type="button" className="btn ghost sm run-btn" onClick={() => void cancelRun()}>
          <span className="stop-glyph" aria-hidden="true" />
          {t("run.stop")}
        </button>
      ) : (
        <button
          type="button"
          className="btn primary sm run-btn"
          disabled={!canRun}
          title={errors.length > 0 ? t("preflight.run_blocked") : undefined}
          onClick={() => void startRun()}
        >
          <Icon name="play" size={14} />
          {t("app.run")}
        </button>
      )}
    </header>
  );
}
