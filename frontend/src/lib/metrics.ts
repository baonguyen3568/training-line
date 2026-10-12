// Training metrics (contract: frontend-metrics-contract skill — metrics.jsonl, wide rows, units in key names).

export interface MetricPoint {
  schema_version: number;
  ts: number; // unix seconds UTC
  step: number;
  epoch?: number;
  split?: "train" | "val";
  loss?: number | null;
  lr?: number | null;
  grad_norm?: number | null;
  tokens_per_s?: number | null;
  mem_max_alloc_gb?: number | null;
  [extra: string]: unknown;
}

export const SUPPORTED_SCHEMA = [1] as const;

/** Parse JSONL defensively: skip blank/partial lines (writers append+flush, the last line may be in flight). */
export function parseJsonl(text: string): { points: MetricPoint[]; skipped: number } {
  const points: MetricPoint[] = [];
  let skipped = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line) as MetricPoint;
      if (typeof row.step === "number" && typeof row.ts === "number") points.push(row);
      else skipped++;
    } catch {
      skipped++;
    }
  }
  return { points, skipped };
}

/** Non-finite (NaN/inf/null) values break the line instead of drawing a fake zero. */
export function finiteOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Exponential moving average for the "smoothed" curve; gaps (null) are carried through. */
export function ema(values: (number | null)[], alpha = 0.1): (number | null)[] {
  let acc: number | null = null;
  return values.map((v) => {
    if (v === null) return null;
    acc = acc === null ? v : alpha * v + (1 - alpha) * acc;
    return acc;
  });
}

/** Indices of non-finite values (NaN/inf markers on the chart, P006 annex (c)). */
export function nonFiniteSteps(points: MetricPoint[], key: string): number[] {
  return points
    .filter((p) => p[key] !== undefined && finiteOrNull(p[key]) === null)
    .map((p) => p.step);
}

/**
 * Largest-Triangle-Three-Buckets downsampling to keep long runs fast in the browser (≤ threshold points).
 * The backend should downsample too (`?max_points=`); this is the client-side guard.
 */
export function lttb(xs: number[], ys: (number | null)[], threshold: number): number[] {
  const n = xs.length;
  if (threshold >= n || threshold < 3) return xs.map((_, i) => i);
  const keep = [0];
  const bucket = (n - 2) / (threshold - 2);
  let a = 0;
  for (let i = 0; i < threshold - 2; i++) {
    const start = Math.floor((i + 1) * bucket) + 1;
    const end = Math.min(Math.floor((i + 2) * bucket) + 1, n);
    let avgX = 0;
    let avgY = 0;
    let cnt = 0;
    for (let j = start; j < end; j++) {
      const y = ys[j];
      if (y === null || y === undefined) continue;
      avgX += xs[j] ?? 0;
      avgY += y;
      cnt++;
    }
    if (cnt) {
      avgX /= cnt;
      avgY /= cnt;
    }
    const rStart = Math.floor(i * bucket) + 1;
    const rEnd = Math.floor((i + 1) * bucket) + 1;
    const ax = xs[a] ?? 0;
    const ay = ys[a] ?? 0;
    let best = rStart;
    let bestArea = -1;
    for (let j = rStart; j < rEnd; j++) {
      const y = ys[j];
      if (y === null || y === undefined) continue;
      const area = Math.abs((ax - avgX) * (y - ay) - (ax - (xs[j] ?? 0)) * (avgY - ay));
      if (area > bestArea) {
        bestArea = area;
        best = j;
      }
    }
    keep.push(best);
    a = best;
  }
  keep.push(n - 1);
  return keep;
}

export function median(xs: readonly number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? (s[m] ?? null) : ((s[m - 1] ?? 0) + (s[m] ?? 0)) / 2;
}

/**
 * Compact number for axes, legends and tables: scientific for tiny/huge magnitudes so a learning rate of 1e-4
 * never renders as "0" (design review r3 M1).
 */
export function formatNumber(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 0.01 || a >= 1e5) return v.toExponential(Math.max(digits - 2, 1)).replace("e", "e");
  return Number(v.toPrecision(digits)).toString();
}
