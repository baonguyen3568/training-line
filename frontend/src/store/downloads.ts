import { create } from "zustand";
import { api } from "../api";
import { ApiError } from "../api/client";
import type { EventError } from "../api/types";
import { useWorkflow } from "./workflow";

export type DownloadKind = "model" | "dataset";
export type DownloadStatus =
  | "starting"
  | "needs_confirm"
  | "queued"
  | "downloading"
  | "verifying"
  | "completed"
  | "unverified"
  | "failed";

export interface DownloadEntry {
  kind: DownloadKind;
  catalogId: string;
  status: DownloadStatus;
  sizeGb?: number;
  thresholdGb?: number;
  bytesDone?: number;
  bytesTotal?: number;
  filesDone?: number;
  filesTotal?: number;
  error?: EventError;
  /** Queued until enough RAM is free (event "waiting"); cleared once it starts downloading. */
  waiting?: { need_gb?: number; avail_gb?: number };
  /** From the server's answer (the pinned catalog): preferred over the client's option_meta once known. */
  name?: string;
  license?: string;
  gated?: boolean;
}

/** The server frees its one download slot before it says completed; if a POST still lands on it, try once more. */
const BUSY_RETRY_MS = 2000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const busyCode = (e: unknown) =>
  e instanceof ApiError
    ? (e.body as { detail?: { code?: string } } | null)?.detail?.code === "SYS-002"
    : false;

export const downloadKey = (kind: DownloadKind, catalogId: string) => `${kind}:${catalogId}`;

interface DownloadsState {
  entries: Record<string, DownloadEntry>;
  /** POST /api/downloads; follows its events until completed/failed, then re-validates the open workflow. */
  start(
    kind: DownloadKind,
    catalogId: string,
    opts?: { confirm?: boolean; check?: boolean },
  ): Promise<void>;
  /**
   * "Bấm Chạy khi thiếu": one download after another (the server runs one at a time), each confirmed by the
   * user's single OK on the list. Resolves with the first one that did not complete, or null when all did; a
   * failure stops the chain, so the caller never starts the run on a half-downloaded workflow.
   */
  downloadAll(targets: { kind: DownloadKind; catalogId: string }[]): Promise<string | null>;
}

const SETTLED: DownloadStatus[] = ["completed", "unverified", "failed", "needs_confirm"];

/**
 * G-5 (P018) downloads of this app session, keyed by kind + catalog id, so the node face, the inspector and the
 * dialog all read the same progress. One download at a time is the server's rule (SYS-002 when busy).
 */
export const useDownloads = create<DownloadsState>((set, get) => {
  const put = (key: string, patch: Partial<DownloadEntry>) => {
    const cur = get().entries[key];
    if (!cur) return;
    set({ entries: { ...get().entries, [key]: { ...cur, ...patch } } });
  };
  return {
    entries: {},
    async downloadAll(targets) {
      for (const { kind, catalogId } of targets) {
        const key = downloadKey(kind, catalogId);
        await get().start(kind, catalogId, { confirm: true });
        const status = await new Promise<DownloadStatus>((resolve) => {
          const now = get().entries[key]?.status;
          if (now && SETTLED.includes(now)) return resolve(now);
          const off = useDownloads.subscribe((st) => {
            const s2 = st.entries[key]?.status;
            if (s2 && SETTLED.includes(s2)) {
              off();
              resolve(s2);
            }
          });
        });
        if (status !== "completed") return key;
      }
      return null;
    },
    async start(kind, catalogId, opts = {}) {
      const key = downloadKey(kind, catalogId);
      set({ entries: { ...get().entries, [key]: { kind, catalogId, status: "starting" } } });
      try {
        // ADR-021: a model the user added needs its licence confirmed first (SEC-005 otherwise); the dialog's
        // "I agree" tick is that confirmation
        if (kind === "model" && catalogId.startsWith("user--"))
          await api.confirmUserModelLicense(catalogId);
        const body = { kind, catalog_id: catalogId, ...opts };
        const res = await api.startDownload(body).catch(async (e: unknown) => {
          if (!busyCode(e)) throw e;
          await sleep(BUSY_RETRY_MS);
          return api.startDownload(body);
        });
        put(key, {
          ...(res.name ? { name: res.name } : {}),
          ...(res.license ? { license: res.license } : {}),
          ...(res.gated !== undefined ? { gated: res.gated } : {}),
        });
        if (res.needs_confirm) {
          put(key, { status: "needs_confirm", sizeGb: res.size_gb, thresholdGb: res.threshold_gb });
          return;
        }
        if (!res.download_id) {
          // already in the cache: verified, or present but not checked by the app yet
          put(key, {
            status: res.verified === false ? "unverified" : "completed",
            sizeGb: res.size_gb,
          });
          if (res.verified !== false) void useWorkflow.getState().validate();
          return;
        }
        put(key, { status: "queued", sizeGb: res.size_gb });
        const stop = api.watchDownload(res.download_id, (ev) => {
          if (ev.type === "progress") {
            put(key, {
              status: "downloading",
              waiting: undefined,
              bytesDone: ev.bytes_done,
              bytesTotal: ev.bytes_total,
              filesDone: ev.files_done,
              filesTotal: ev.files_total,
            });
          } else if (ev.type === "status") {
            put(key, { status: ev.status, ...(ev.error ? { error: ev.error } : {}) });
            if (ev.status === "completed") {
              stop();
              void useWorkflow.getState().validate(); // the step's NET-002 goes away
            }
            if (ev.status === "failed") stop();
          } else if (ev.type === "waiting") {
            put(key, {
              waiting: {
                ...(ev.need_gb !== undefined ? { need_gb: ev.need_gb } : {}),
                ...(ev.avail_gb !== undefined ? { avail_gb: ev.avail_gb } : {}),
              },
            });
          } else if (ev.type === "download_not_found" || ev.type === "connection_lost") {
            put(key, { status: "failed" }); // socket lost / unknown download: say it failed, no made-up code
          }
          // any other event type is a newer server's: ignored, never read as a failure
        });
      } catch (e) {
        const detail =
          e instanceof ApiError
            ? (e.body as { detail?: { code?: string; params?: Record<string, unknown> } } | null)
                ?.detail
            : undefined;
        put(key, {
          status: "failed",
          error: { code: detail?.code ?? "SYS-000", params: detail?.params ?? {} },
        });
      }
    },
  };
});

/** 0–100 from bytes, or undefined before the first progress event. */
export function downloadPercent(e: DownloadEntry | undefined): number | undefined {
  if (!e?.bytesTotal) return undefined;
  return Math.min(100, Math.round(((e.bytesDone ?? 0) / e.bytesTotal) * 100));
}
