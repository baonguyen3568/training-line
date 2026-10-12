import type { Workflow } from "../api/types";
import { parseWorkflowFile, serializeWorkflow } from "./workflowFile";

// Unsaved-work autosave (design review M1 K1). Per-viewer convenience only: localStorage can be unavailable
// or cleared, so every access is guarded. The draft is re-parsed with the untrusted file parser on restore.
const KEY = "vlm.draft.v1";

export interface Draft {
  workflow: Workflow;
  savedAt: number;
}

export function saveDraft(wf: Workflow): void {
  try {
    localStorage.setItem(
      KEY,
      JSON.stringify({ savedAt: Date.now(), workflow: serializeWorkflow(wf) }),
    );
  } catch {
    /* storage unavailable */
  }
}

export function loadDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as { savedAt?: unknown; workflow?: unknown };
    if (typeof d.workflow !== "string" || typeof d.savedAt !== "number") return null;
    const parsed = parseWorkflowFile(d.workflow);
    return parsed.ok && parsed.workflow.nodes.length > 0
      ? { workflow: parsed.workflow, savedAt: d.savedAt }
      : null;
  } catch {
    return null;
  }
}

export function clearDraft(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}
