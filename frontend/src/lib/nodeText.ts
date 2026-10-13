import type { NodeSpec } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { untrustedText } from "./safeText";

/** Node title for display. Third-party text is untrusted: plain text, invisible chars escaped, truncated. */
export function nodeTitle(spec: NodeSpec | undefined, typeId: string): string {
  const raw = spec?.title ?? tp(["node", typeId, "title"]);
  return spec?.origin.kind === "third_party" ? untrustedText(raw, 60) : raw;
}

export function nodeDesc(spec: NodeSpec | undefined, typeId: string): string {
  const raw = spec?.desc ?? tp(["node", typeId, "desc"]);
  return spec?.origin.kind === "third_party" ? untrustedText(raw, 200) : raw;
}

/** Display name of a workflow: a template's title, "untitled" copy, or the file's own (untrusted) name. */
export function workflowTitle(name: string): string {
  if (name === "untitled") return t("workflow.untitled");
  return tpOptional(["template", name, "title"]) ?? untrustedText(name, 60);
}
