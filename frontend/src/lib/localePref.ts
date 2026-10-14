import type { Workflow } from "../api/types";
import { getLocale, type Locale, setLocale } from "../i18n";
import { loadDraft, saveDraft } from "./draft";

// The viewer's language (A5): per-browser convenience in localStorage, read once before the first render.
const KEY = "vlm.locale.v1";
const LOCALES: readonly Locale[] = ["vi", "en"];

export function applyStoredLocale(): void {
  try {
    const v = localStorage.getItem(KEY);
    if (v && (LOCALES as readonly string[]).includes(v)) setLocale(v as Locale);
  } catch {
    /* storage unavailable: the default language */
  }
  document.documentElement.lang = getLocale();
}

/**
 * Switch language by reloading (design A5: the server renders some text in the request's locale, so a reload keeps
 * everything in one language). An open workflow is first saved as the draft the app offers to restore; when that
 * draft cannot be written and read back, nothing changes and false says why (unsaved work would be lost).
 */
export function switchLocale(next: Locale, workflow: Workflow | null): boolean {
  if (next === getLocale()) return true;
  if (workflow && workflow.nodes.length > 0) {
    saveDraft(workflow);
    if ((loadDraft()?.workflow.nodes.length ?? 0) !== workflow.nodes.length) return false;
  }
  try {
    localStorage.setItem(KEY, next);
  } catch {
    return false; // the choice would not survive the reload
  }
  window.location.reload();
  return true;
}
