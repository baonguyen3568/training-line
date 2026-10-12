import { useRef, useState } from "react";
import { t } from "../i18n";
import { parseWorkflowFile, serializeWorkflow } from "../lib/workflowFile";
import { useWorkflow } from "../store/workflow";

/** Canvas toolbar: new / open / save (.json) / undo / redo / duplicate / delete (M1-4). */
export function Toolbar({ locked }: { locked: boolean }) {
  const { workflow, selectedId, past, future } = useWorkflow();
  const { newWorkflow, openWorkflow, undo, redo, duplicateNode, removeNodes } =
    useWorkflow.getState();
  const fileInput = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  if (!workflow) return null;

  const save = () => {
    // Plain download (no File System Access API) so it works everywhere and in headless E2E.
    const blob = new Blob([serializeWorkflow(workflow)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${workflow.name.replace(/[^\w.-]+/g, "_") || "workflow"}.vlmflow.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const open = async (file: File | undefined) => {
    if (!file) return;
    const res = parseWorkflowFile(await file.text());
    if (!res.ok) {
      setError(`${t("error.CFG-007.title")}: ${t("error.CFG-007.what", { field: res.field })}`);
      return;
    }
    setError(null);
    openWorkflow(res.workflow);
  };

  return (
    <div className="toolbar" role="toolbar" aria-label={t("app.name")}>
      <button type="button" onClick={newWorkflow} disabled={locked}>
        ＋ {t("gallery.blank_workflow")}
      </button>
      <button type="button" onClick={() => fileInput.current?.click()} disabled={locked}>
        📂 {t("workflow.open")}
      </button>
      <input
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="workflow-file-input"
        onChange={(e) => {
          void open(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <button type="button" onClick={save}>
        💾 {t("workflow.save")}
      </button>
      <span className="sep" />
      <button
        type="button"
        onClick={undo}
        disabled={locked || past.length === 0}
        aria-keyshortcuts="Control+Z"
      >
        ↶ {t("app.undo")}
      </button>
      <button
        type="button"
        onClick={redo}
        disabled={locked || future.length === 0}
        aria-keyshortcuts="Control+Shift+Z"
      >
        ↷ {t("app.redo")}
      </button>
      <span className="sep" />
      <button
        type="button"
        onClick={() => selectedId && duplicateNode(selectedId)}
        disabled={locked || !selectedId}
        aria-keyshortcuts="Control+D"
      >
        ⧉ {t("workflow.duplicate")}
      </button>
      <button
        type="button"
        onClick={() => selectedId && removeNodes([selectedId])}
        disabled={locked || !selectedId}
        aria-keyshortcuts="Delete"
      >
        🗑 {t("workflow.delete")}
      </button>
      {locked && <span className="muted small">🔒 {t("run.locked_while_running")}</span>}
      {error && (
        <span role="alert" className="toolbar-error">
          {error}
        </span>
      )}
    </div>
  );
}
