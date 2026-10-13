import type { EstimateSource, Range, ValidationResult } from "../api/types";
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
  if (kind === "minutes") return formatMinutesRange(r);
  return t(`estimate.range_${kind}`, { lo: r.lo, hi: r.hi });
}

/** 97 → "1 giờ 37 phút", 120 → "2 giờ", 45 → "45 phút". */
function minutesText(min: number): string {
  const total = Math.round(min);
  const h = Math.floor(total / 60);
  const m = total % 60;
  if (h === 0) return t("estimate.dur_m", { m });
  return m ? t("estimate.dur_hm", { h, m }) : t("estimate.dur_h", { h });
}

/**
 * Minute range for people: "~20–35 phút" under an hour, else "~1 giờ 37 phút – 1 giờ 52 phút". A narrow range
 * (< 5 min or < 5 % of hi) reads as one number, "~1 giờ 55 phút": a 1-minute spread is false precision (design).
 */
export function formatMinutesRange(r: Range | null): string {
  if (!r) return t("estimate.source.unknown");
  const lo = Math.round(r.lo);
  const hi = Math.round(r.hi);
  if (r.hi - r.lo < 5 || r.hi - r.lo < 0.05 * r.hi) return `~${minutesText(hi)}`;
  if (hi < 60) return t("estimate.range_minutes", { lo, hi });
  return t("estimate.range_duration", { lo: minutesText(lo), hi: minutesText(hi) });
}

/**
 * Time line of a pre-flight estimate. A server that sends `duration` measured it from real runs, so its range
 * stands on its own (`source` is about RAM/disk only); older servers keep the `source` gate.
 */
export function workflowDurationRange(est: ValidationResult["estimate"] | undefined): Range | null {
  if (!est) return null;
  if (est.duration !== undefined) return est.duration?.duration_min ?? null;
  return est.source === "unknown" ? null : est.duration_min;
}

export function workflowDurationText(est: ValidationResult["estimate"] | undefined): string {
  return formatMinutesRange(workflowDurationRange(est));
}

/** Distinct earlier runs the measured time comes from (for "đo từ N lần chạy trước"). */
export function durationRunCount(est: ValidationResult["estimate"] | undefined): number {
  const per = est?.duration?.per_node ?? {};
  return new Set(Object.values(per).flatMap((n) => n.based_on)).size;
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

/**
 * Pre-flight estimate in one line: the numbers we have, then the missing ones named once
 * ("Dự kiến: ~1 giờ 37 phút – 1 giờ 52 phút · RAM, ổ đĩa: chưa có số đo"), never "chưa có số đo" three times.
 */
export function estimateLine(est: ValidationResult["estimate"]): string {
  const time = workflowDurationRange(est);
  const sized = est.source !== "unknown";
  const known: string[] = [];
  const missing: string[] = [];
  if (time) known.push(formatMinutesRange(time));
  else missing.push(t("estimate.name.time"));
  if (sized && est.ram_gb_peak) known.push(formatEstimate(est.ram_gb_peak, "ram_gb", est.source));
  else missing.push(t("estimate.name.ram"));
  if (sized && est.disk_gb) known.push(formatEstimate(est.disk_gb, "disk_gb", est.source));
  else missing.push(t("estimate.name.disk"));
  if (known.length === 0) return t("estimate.source.unknown_all");
  const tail = missing.length ? ` · ${t("estimate.missing", { names: missing.join(", ") })}` : "";
  return `${t("preflight.estimate_known", { parts: known.join(" · ") })}${tail}`;
}
