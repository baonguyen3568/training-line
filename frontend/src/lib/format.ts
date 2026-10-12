import type { EstimateSource, Range } from "../api/types";
import { getLocale, t } from "../i18n";

export type EstimateKind = "minutes" | "ram_gb" | "disk_gb";

/**
 * Estimates are always ranges with a source (P010 §6): never invent a number. Copy comes from
 * `estimate.range_*`; durations of two hours or more switch to hours.
 */
export function formatEstimate(
  r: Range | null,
  kind: EstimateKind,
  source: EstimateSource,
): string {
  if (!r || source === "unknown") return t("estimate.source.unknown");
  if (kind === "minutes" && r.hi >= 120) {
    const h = (m: number) => Math.round((m / 60) * 10) / 10;
    return t("estimate.range_hours", { lo: h(r.lo), hi: h(r.hi) });
  }
  return t(`estimate.range_${kind}`, { lo: r.lo, hi: r.hi });
}

/** Locale-aware date-time: vi "12/10/2026 17:35", en "10/12/2026, 5:35 PM" (design review M1 r2 #3). */
export function formatDateTime(epochMs: number, locale: string = getLocale()): string {
  return new Intl.DateTimeFormat(locale === "vi" ? "vi-VN" : "en-US", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: locale !== "vi",
  })
    .format(new Date(epochMs))
    .replace(/^(\d{2}:\d{2}) (.*)$/, "$2 $1"); // vi-VN puts the time first; we want date then time
}

/** Short duration: "45 giây" / "3 phút 20 giây" style via i18n-free numbers (mm:ss for < 1h, h:mm otherwise). */
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}
