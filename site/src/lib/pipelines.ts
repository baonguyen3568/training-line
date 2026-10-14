// Reference pipelines as the app's gallery reads them (bao-research configs/reference_pipelines.yaml, exported by
// scripts/app_copy.py), and the copy that formats their measurements. Never a number that is not in the data.
import copy from "~/generated/app-copy.json";
import type { TrainArch } from "../../../frontend/src/lib/archLayout";

export type Pipe = {
  id: string;
  kind: string;
  status: string;
  difficulty: "basic" | "advanced" | null;
  method: string;
  title: { vi: string };
  summary_key?: string;
  /** method.<method>.when: the method's one "when to use" sentence. */
  when_to_use_key: string;
  train_arch: TrainArch;
  source: { url: string; publisher: string; paper_url?: string | null };
  measured_with?: { model: string; data: string } | null;
  license: { code: string | null; model: string | null; data: string | null };
  steps: { node: string | null; label: { vi: string } }[];
  measured?: {
    train_minutes?: number | null;
    peak_ram_gb?: number;
    score?: {
      benchmark: string;
      metric: string;
      value: number;
      ci95?: [number, number];
      n?: number;
      compare_with?: string;
    };
    config?: string;
    sources?: string[];
  } | null;
};

export const allPipes = copy.pipelines_data as Pipe[];
export const pl = copy.pipelines as Record<string, any>;
const est = copy.estimate as Record<string, string>;

export const fill = (tpl: string, v: Record<string, string | number>) =>
  tpl.replace(/\{(\w+)\}/g, (_, k) => String(v[k] ?? `{${k}}`));

/** A dotted key of the app's copy ("arch.block.vision", "method.lora.title"), filled; the key itself if missing. */
export const copyText = (key: string, params?: Record<string, string | number>): string => {
  const v = key.split(".").reduce<unknown>((o, k) => (o as Record<string, unknown> | undefined)?.[k], copy);
  return typeof v === "string" ? (params ? fill(v, params) : v) : key;
};

export const minutes = (m: number) => {
  const total = Math.round(m);
  const h = Math.floor(total / 60);
  const r = total % 60;
  return h === 0 ? fill(est.dur_m, { m: r }) : r ? fill(est.dur_hm, { h, m: r }) : fill(est.dur_h, { h });
};

/** license.name.<id> from the app's copy (design; security: names live in i18n), else the id as written. The id is one
 * key: dotted ids ("llama3.1") are never split, and only own keys are read. */
const licenseNames = ((copy as Record<string, unknown>).license as { name?: Record<string, string> } | undefined)?.name ?? {};
const licenseName = (id: string) => (Object.hasOwn(licenseNames, id) ? (licenseNames[id] as string) : id);
export const METRIC = (pl.metric ?? {}) as Record<string, string>; // pipelines.metric.<id> (design copy)
/** pipelines.metric_help.<id>: what the number means, first in the card's "?" when design wrote it ({n}). */
export const metricHelp = (s: { metric: string; n?: number }) => {
  const tpl = (pl.metric_help ?? {})[s.metric] as string | undefined;
  return tpl ? fill(tpl, { n: s.n ?? "" }) : null;
};

// Same rule as the app (frontend/src/lib/score.ts, design 94ee3e9): a difference metric is signed with a real
// minus and one decimal ("−0.6", "+0.3", "0.0"), its interval "−1.8 – 0.4"; other metrics keep two decimals.
const MINUS = "\u2212";
const isDelta = (metric: string) => /(^|_)delta(_|$)/.test(metric);
const round1 = (v: number) => Math.round(v * 10) / 10;
export const formatScore = (metric: string, v: number) => {
  if (!isDelta(metric)) return v.toFixed(2);
  const r = round1(v);
  return r === 0 ? "0.0" : `${r > 0 ? "+" : MINUS}${Math.abs(r).toFixed(1)}`;
};
export const formatCi = (metric: string, [lo, hi]: [number, number]) => {
  if (!isDelta(metric)) return `${lo.toFixed(2)}–${hi.toFixed(2)}`;
  const b = (v: number) => (round1(v) < 0 ? `${MINUS}${Math.abs(round1(v)).toFixed(1)}` : round1(v).toFixed(1));
  return `${b(lo)} – ${b(hi)}`;
};
export const licenseOf = (p: Pipe) =>
  [...new Set([p.license.code, p.license.model].filter(Boolean).map((x) => licenseName(x as string)))].join(" · ");
/** The measured line's parts (the card wraps only between them): time · peak RAM · score. */
export const measuredParts = (m: NonNullable<Pipe["measured"]>) =>
  [
    m.train_minutes ? `~${minutes(m.train_minutes)}` : null,
    m.peak_ram_gb ? fill(pl.ram_peak, { gb: m.peak_ram_gb }) : null,
    m.score ? `${METRIC[m.score.metric] ?? m.score.metric} ${formatScore(m.score.metric, m.score.value)}` : null,
  ].filter((x): x is string => !!x);
export const measuredLine = (m: NonNullable<Pipe["measured"]>) => measuredParts(m).join(" · ");
/** The licences, each once (code, model, data), for the badge; the line wraps only between them. */
export const licenseParts = (p: Pipe) => [
  ...new Set(
    [p.license.code, p.license.model, p.license.data].map((x) => (x ? licenseName(x) : pl.see_source)),
  ),
];
/** The pipeline this one's score is compared with (generator-checked: same benchmark and metric). */
export const compareOf = (p: Pipe) => allPipes.find((x) => x.id === p.measured?.score?.compare_with);
/** score_compare around its {other} placeholder, so {other} can be a link: [before, after]. */
export const compareParts = (other: Pipe) => {
  const score = other.measured?.score;
  const tpl = fill(pl.score_compare as string, { other_value: score ? formatScore(score.metric, score.value) : "" });
  const [before, after = ""] = tpl.split("{other}");
  return [before, after] as const;
};
