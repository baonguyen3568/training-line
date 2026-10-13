/**
 * Pipeline scores for people (design 94ee3e9). A difference metric ("…_delta…", e.g. FP8 minus BF16) is signed with
 * a real minus "−" (U+2212) and one decimal: "−0.6", "+0.3", "0.0"; its interval reads "−1.8 – 0.4". Other metrics
 * (F1…) keep two decimals and "66.33–75.38".
 */
export const isDeltaMetric = (metric: string) => /(^|_)delta(_|$)/.test(metric);

const MINUS = "−";

function signed(v: number): string {
  const r = Math.round(v * 10) / 10;
  if (r === 0) return "0.0";
  return `${r > 0 ? "+" : MINUS}${Math.abs(r).toFixed(1)}`;
}

/** A bound of an interval: a minus when negative, never a plus (the range reads "−1.8 – 0.4"). */
function bound(v: number): string {
  const r = Math.round(v * 10) / 10;
  return r < 0 ? `${MINUS}${Math.abs(r).toFixed(1)}` : r.toFixed(1);
}

export function formatScore(metric: string, value: number): string {
  return isDeltaMetric(metric) ? signed(value) : value.toFixed(2);
}

export function formatCi(metric: string, [lo, hi]: readonly [number, number]): string {
  return isDeltaMetric(metric)
    ? `${bound(lo)} – ${bound(hi)}`
    : `${lo.toFixed(2)}–${hi.toFixed(2)}`;
}
