import { useRef, useState } from "react";
import { t } from "../i18n";
import { parseWorkflowFile, serializeWorkflow } from "../lib/workflowFile";
import { useWorkflow } from "../store/workflow";

/** Save (.json download) and open (file picker) shared by the classic and the D4 toolbars. */
export function useWorkflowFile() {
  const workflow = useWorkflow((st) => st.workflow);
  const { openWorkflow } = useWorkflow.getState();
  const fileInput = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const save = () => {
    if (!workflow) return;
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

  const input = (
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
  );
  return { save, pick: () => fileInput.current?.click(), input, error };
}

/** Canvas toolbar: new / open / save (.json) / undo / redo / duplicate / delete (M1-4). */
export function Toolbar({ locked, running }: { locked: boolean; running: boolean }) {
  const { workflow, selectedId, past, future } = useWorkflow();
  const { newWorkflow, undo, redo, duplicateNode, removeNodes } = useWorkflow.getState();
  const file = useWorkflowFile();
  if (!workflow) return null;

  return (
    <div className="toolbar" role="toolbar" aria-label={t("app.name")}>
      <button type="button" onClick={newWorkflow} disabled={locked}>
        ＋ {t("gallery.blank_workflow")}
      </button>
      <button type="button" onClick={file.pick} disabled={locked}>
        📂 {t("workflow.open")}
      </button>
      {file.input}
      <button type="button" onClick={file.save}>
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
      {running && <span className="muted small">🔒 {t("run.locked_while_running")}</span>}
      {file.error && (
        <span role="alert" className="toolbar-error">
          {file.error}
        </span>
      )}
    </div>
  );
}
