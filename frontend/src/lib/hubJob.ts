import { ApiError } from "../api/client";
import type { EventError, HubJob } from "../api/types";

export const POLL_MS = 1000;

/**
 * Poll a plan-only hub job (P021 §8: resolve a model / inspect a dataset) every second until it is done or failed;
 * each answer goes to `onUpdate` (queued, waiting_for…). `stopped()` true ends it quietly (the panel closed).
 */
export async function pollHubJob<J extends Pick<HubJob<unknown>, "status">>(
  fetchJob: () => Promise<J>,
  onUpdate: (job: J) => void,
  stopped: () => boolean,
  wait: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<J | null> {
  for (;;) {
    if (stopped()) return null;
    const job = await fetchJob();
    if (stopped()) return null;
    onUpdate(job);
    if (job.status === "done" || job.status === "failed") return job;
    await wait(POLL_MS);
  }
}

/** The 5 repos pasted last (spec H §1), per browser: a convenience only, never trusted for anything. */
const RECENT_KEY = "vlm.byo.recent.v1";

export function recentRepos(kind: "model" | "dataset"): string[] {
  try {
    const all = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "{}") as Record<string, unknown>;
    const list = all[kind];
    return Array.isArray(list)
      ? list.filter((x): x is string => typeof x === "string").slice(0, 5)
      : [];
  } catch {
    return [];
  }
}

export function rememberRepo(kind: "model" | "dataset", repo: string): void {
  try {
    const all = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "{}") as Record<string, string[]>;
    const list = [repo, ...(all[kind] ?? []).filter((r) => r !== repo)].slice(0, 5);
    localStorage.setItem(RECENT_KEY, JSON.stringify({ ...all, [kind]: list }));
  } catch {
    /* storage unavailable */
  }
}

/** A Hugging Face repo id, as the server accepts it (org/name). */
export const isRepoId = (s: string) => /^[\w.-]+\/[\w.-]+$/.test(s.trim());

/**
 * The server's code and params of a refused add or inspect: the 400/409 detail ({code, params}: a malformed id, the
 * download slot busy…), or a 422 schema refusal said as the format error of the field it names. Anything else is SYS-000.
 */
export function hubError(e: unknown): EventError {
  if (!(e instanceof ApiError)) return { code: "SYS-000", params: {} };
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (Array.isArray(detail)) {
    const loc = (detail[0] as { loc?: unknown[] } | undefined)?.loc ?? [];
    const param = loc.includes("revision")
      ? "revision"
      : loc.includes("split")
        ? "split"
        : loc.includes("config")
          ? "config"
          : "repo";
    return { code: "CFG-005", params: { param, rule: "format" } };
  }
  const d = detail as { code?: unknown; params?: Record<string, unknown> } | undefined;
  return typeof d?.code === "string"
    ? { code: d.code, params: d.params ?? {} }
    : { code: "SYS-000", params: {} };
}

/** One segment of a user dataset id, as the server builds it (common/catalog.py _id_part). */
export function idPart(text: string): string {
  const part = text
    .toLowerCase()
    .replace(/[^a-z0-9.]+/g, "-")
    .replace(/\.{2,}/g, ".")
    .replace(/^[.-]+|[.-]+$/g, "");
  return part || "x";
}

/**
 * The config + split a user dataset id names ("…--<config>--<split>"), among those an inspection lists: so a workflow
 * opened on another machine (DATA-013) re-adds the same split, and its step's id matches again (backend H5 review).
 */
export function configSplitOf(
  datasetId: string,
  configs: string[],
  splits: string[],
): { config: string; split: string } | null {
  for (const config of configs.length ? configs : ["default"])
    for (const split of splits)
      if (datasetId.endsWith(`--${idPart(config)}--${idPart(split)}`)) return { config, split };
  return null;
}
