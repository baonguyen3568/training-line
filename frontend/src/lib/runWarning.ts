import type { EventError } from "../api/types";
import { t, tpOptional } from "../i18n";
import { formatDateTime } from "./format";

/**
 * Text for a run `warning` event (e.g. CACHE-001). The executor sends raw params: node is a node id,
 * saved_at is epoch seconds, est_minutes may be null. Missing params pick the `what_no_estimate` variant
 * when it exists, otherwise the explanation is left out rather than showing "null".
 */
export function runWarningText(
  err: EventError,
  nodeLabel: (nodeId: string) => string,
): { title: string; what?: string } {
  const raw = err.params ?? {};
  const params: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === null || v === undefined) continue;
    if (k === "node" && typeof v === "string") params[k] = nodeLabel(v);
    else if (k === "saved_at" && typeof v === "number") params[k] = formatDateTime(v * 1000);
    else if (typeof v === "string" || typeof v === "number") params[k] = v;
  }
  const missing = Object.keys(raw).some((k) => !(k in params));
  const title = t(`error.${err.code}.title`);
  const what = missing
    ? tpOptional(["error", err.code, "what_no_estimate"], params)
    : tpOptional(["error", err.code, "what"], params);
  return what === undefined ? { title } : { title, what };
}
