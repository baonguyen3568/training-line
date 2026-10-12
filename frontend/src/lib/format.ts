import type { EstimateSource, Range } from "../api/types";
import { t } from "../i18n";

export type EstimateKind = "minutes" | "ram_gb" | "disk_gb";

/**
 * Estimates are always ranges with a source (P010 §6): never invent a number. Copy comes from
 * `estimate.range_*`; durations of two hours or more switch to hours.
 */
export function formatEstimate(r: Range | null, kind: EstimateKind, source: EstimateSource): string {
  if (!r || source === "unknown") return t("estimate.source.unknown");
  if (kind === "minutes" && r.hi >= 120) {
    const h = (m: number) => Math.round((m / 60) * 10) / 10;
    return t("estimate.range_hours", { lo: h(r.lo), hi: h(r.hi) });
  }
  return t(`estimate.range_${kind}`, { lo: r.lo, hi: r.hi });
}
