// Same-origin API client. Auth is the HttpOnly session cookie `vlm_session` set by the server (P003 D13).
// CSRF: the token comes from `GET /api/session` (JSON body) and is kept ONLY in module memory — never a
// readable cookie (cookies are not port-isolated on 127.0.0.1, so other local apps could toss/overwrite them)
// and never localStorage. It changes on every server restart; a 403 csrf_invalid triggers one refresh + retry.
import { getLocale } from "../i18n";
import type {
  AdvancedOption,
  BugReport,
  DatasetInspection,
  DownloadEvent,
  DownloadRequest,
  DownloadStart,
  HealthInfo,
  HubJob,
  LanInfo,
  MetricsPage,
  MetricsQuery,
  ModelResolveJob,
  NodeResetPlan,
  NodeResetResult,
  NodeTypesResponse,
  RunEvent,
  RunInfo,
  SecretField,
  SecretState,
  SessionInfo,
  SettingsView,
  StorageApplyRequest,
  StorageApplyResult,
  StoragePlan,
  SystemStatus,
  Template,
  TemplateMeta,
  TrialResult,
  UploadInfo,
  UserAssetLimits,
  UserDataset,
  UserModel,
  ValidationResult,
  Workflow,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: unknown,
  ) {
    super(`API ${status}`);
  }
}

/**
 * Where the app is mounted: "" at the site root, "/app" behind nginx under a prefix (D-8). Read from the page's
 * own path at call time (the SPA has one URL), so one build works at both; every API, upload and socket URL
 * goes through it.
 */
export function appPrefix(): string {
  return location.pathname.replace(/\/[^/]*$/, "");
}
const at = (path: string) => `${appPrefix()}${path}`;

/**
 * D-8: the session ended (signed out here or everywhere, expired). An event socket closed with 4401 or an API
 * 401 `unauthenticated` calls this once; the app reloads to the server's sign-in page, or, where none is served
 * (Vite dev, a reload that just happened), says to open the link printed in the terminal. Never a reconnect loop.
 */
let sessionEndedHandler: (() => void) | null = null;
let sessionEndedFired = false;
export function onSessionEnded(handler: (() => void) | null): void {
  sessionEndedHandler = handler;
  sessionEndedFired = false;
}
function sessionEnded(): void {
  if (sessionEndedFired) return;
  sessionEndedFired = true;
  sessionEndedHandler?.();
}
const WS_SESSION_ENDED = 4401;

/**
 * An authenticated event socket (P003 D13: cookie + Origin at handshake, CSRF token as first message).
 * `onFrame` returns true on the stream's last frame; a socket that closes before it reports `onLost`.
 */
function socket(path: string, onFrame: (ev: unknown) => boolean, onLost: () => void): () => void {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}${at(path)}`);
  let closed = false;
  let finished = false;
  ws.onopen = () => {
    void csrfToken().then((token) => ws.send(JSON.stringify({ type: "auth", csrf_token: token })));
  };
  ws.onmessage = (m) => {
    try {
      if (onFrame(JSON.parse(String(m.data)))) finished = true;
    } catch {
      /* ignore malformed frames */
    }
  };
  ws.onclose = (e) => {
    if (e.code === WS_SESSION_ENDED) return sessionEnded(); // not "connection lost": signed out
    if (!closed && !finished) onLost();
  };
  return () => {
    if (!closed) ws.close();
    closed = true;
  };
}

let csrf: Promise<string> | null = null;

function csrfToken(refresh = false): Promise<string> {
  if (refresh || !csrf) {
    csrf = fetch(at("/api/session"), {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new ApiError(r.status, null))))
      .then((d: { csrf_token?: unknown }) => (typeof d.csrf_token === "string" ? d.csrf_token : ""))
      .catch((e: unknown) => {
        csrf = null;
        throw e;
      });
  }
  return csrf;
}

/** Test hook: forget the in-memory token. */
export function resetCsrfForTests(): void {
  csrf = null;
}

const isCsrfRejection = (status: number, data: unknown) =>
  status === 403 &&
  typeof data === "object" &&
  data !== null &&
  (data as { code?: unknown }).code === "csrf_invalid";

/** A non-JSON request body (POST /api/uploads takes the raw file, no multipart parser on the server). */
export interface RawBody {
  raw: Blob;
  contentType: string;
  headers?: Record<string, string>;
}
const isRaw = (b: unknown): b is RawBody => typeof b === "object" && b !== null && "raw" in b;

async function request<T>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
  retried = false,
): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") {
    headers["Content-Type"] = isRaw(body) ? body.contentType : "application/json";
    if (isRaw(body)) Object.assign(headers, body.headers);
    headers["X-VLM-CSRF"] = await csrfToken(retried);
  }
  const res = await fetch(at(path), {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? null : isRaw(body) ? body.raw : JSON.stringify(body),
  });
  const data: unknown = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) {
    if (res.status === 401 && (data as { code?: unknown } | null)?.code === "unauthenticated")
      sessionEnded();
    if (method !== "GET" && !retried && isCsrfRejection(res.status, data))
      return request(method, path, body, true);
    throw new ApiError(res.status, data);
  }
  return data as T;
}

export interface Api {
  nodeTypes(locale: string): Promise<NodeTypesResponse>;
  templates(): Promise<TemplateMeta[]>;
  template(id: string): Promise<Template>;
  validate(workflow: Workflow, locale: string): Promise<ValidationResult>;
  system(): Promise<SystemStatus>;
  runs(): Promise<RunInfo[]>;
  run(id: string): Promise<RunInfo>;
  runMetrics(id: string, q: MetricsQuery): Promise<MetricsPage>;
  /** `trial`: run up to that step with a 5-step trial (ADR-021, backend H5); params are never changed. */
  startRun(workflow: Workflow, trial?: { node: string }): Promise<{ run_id: string }>;
  cancelRun(id: string): Promise<void>;
  /** Subscribe to run events; returns an unsubscribe function. */
  watchRun(id: string, onEvent: (ev: RunEvent) => void): () => void;
  storagePlan(locale: string): Promise<StoragePlan>;
  /** Rejects with ApiError: 409 {code: PLAN_EXPIRED|CLEANUP_IN_PROGRESS}, 403 {error: DISK-002 + admin_command}. */
  storageApply(body: StorageApplyRequest, locale: string): Promise<StorageApplyResult>;
  /** The 5 safe fields of a bug report for one failed step (server-built, filtered). */
  /** With a run: /api/runs/{id}/bug-report (adds ref_code); without (validate/preflight errors): /api/bug-report. */
  bugReport(
    runId: string | null,
    q: { nodeId?: string; code?: string },
    locale: string,
  ): Promise<BugReport>;
  /** G-5: start (or check) a catalog download. 409/404 {detail: {code, params}}: NET-004, DISK-001, SYS-002, CFG-004. */
  /** The code this app runs (version, commit, dirty): Settings › Về app. */
  health(): Promise<HealthInfo>;
  /** A5: how other devices reach this app on the LAN, and its CA; local session only. */
  lan(): Promise<LanInfo>;
  /** D-8: who this browser is (local / LAN) and, on the local machine, the LAN sign-in lock. */
  session(): Promise<SessionInfo>;
  /** DELETE /api/session: this browser only. */
  signOut(): Promise<void>;
  /** Every browser signs in again with the new link the app prints (the token rotates). */
  revokeAllSessions(): Promise<void>;
  /** Local session only (LAN: 403 local_only). */
  unlockLan(): Promise<SessionInfo["lan_login"]>;
  /** ADR-021: start resolving a Hugging Face model (plan-only job); 400 {detail} for a malformed repo id. */
  addUserModel(body: { repo: string; revision?: string }): Promise<{ resolve_id: string }>;
  userModelResolve(id: string): Promise<ModelResolveJob>;
  confirmUserModelLicense(id: string): Promise<{ model: UserModel }>;
  /** ADR-021 A1: the size cap of one user-added model, in GB (settings); PUT outside the range → 422. */
  userAssetLimits(): Promise<UserAssetLimits>;
  setUserAssetLimits(body: { max_model_gb: number }): Promise<UserAssetLimits>;
  /** The user-added models this machine has a record of (the `user_model` widget's choices, backend H3). */
  listUserModels(): Promise<{ models: UserModel[] }>;
  /** The user-added datasets of this machine (licence words for the download dialog, backend H5). */
  listUserDatasets(): Promise<{ datasets: UserDataset[] }>;
  /** H5: add an inspected split with its column mapping; 400 {detail} DATA-010 for a missing role. */
  addUserDataset(body: { inspect_id: string; columns: Record<string, string> }): Promise<{
    dataset: UserDataset;
  }>;
  confirmUserDatasetLicense(id: string): Promise<{ dataset: UserDataset }>;
  inspectDataset(body: {
    repo: string;
    revision?: string;
    config?: string;
    split?: string;
  }): Promise<{ inspect_id: string }>;
  datasetInspection(id: string): Promise<HubJob<DatasetInspection>>;
  startDownload(body: DownloadRequest): Promise<DownloadStart>;
  /** G-5 progress; the socket closes after completed/failed. Returns an unsubscribe function. */
  watchDownload(id: string, onEvent: (ev: DownloadEvent) => void): () => void;
  /** P016 §3: engine options of a train node, filtered to the model's template when `model` is given. */
  advancedOptions(typeId: string, model: string | null): Promise<{ options: AdvancedOption[] }>;
  /** G-4. 422 on a malformed node. */
  nodeResetPlan(node: string, locale: string): Promise<NodeResetPlan>;
  /** 400 {code: CONFIRM_REQUIRED}, 403 {error: DISK-002 + admin_command}, 409 {code: PLAN_EXPIRED | … }. */
  nodeReset(
    body: { plan_id: string; node: string; confirm?: string },
    locale: string,
  ): Promise<NodeResetResult>;
  /** B-7. Rejects with ApiError {error: DATA-005 rendered (title/what)} on 400/413/415/507. */
  upload(file: File, locale: string): Promise<UploadInfo>;
  /** 404 {error: {code: RESULT_NOT_READY}}, 502 {error: {code: RESULT_TOO_LARGE}}. */
  nodeResult(runId: string, nodeId: string): Promise<TrialResult>;
  /** D-7 settings (P015 §4.5). Secrets never come back: only set / last4 / updated_at. */
  settings(): Promise<SettingsView>;
  putSecret(field: SecretField, value: string): Promise<SecretState>;
  deleteSecret(field: SecretField): Promise<{ set: false }>;
}

/** Preview URL of an upload (served as image/png, same origin, cookie-authenticated). */
export const uploadUrl = (id: string) => at(`/api/uploads/${encodeURIComponent(id)}`);

export const httpApi: Api = {
  nodeTypes: (locale) => request("GET", `/api/node-types?locale=${encodeURIComponent(locale)}`),
  templates: () => request("GET", "/api/templates"),
  template: (id) => request("GET", `/api/templates/${encodeURIComponent(id)}`),
  validate: (workflow, locale) =>
    request("POST", `/api/graphs/validate?locale=${encodeURIComponent(locale)}`, workflow),
  system: () => request("GET", "/api/system"),
  runs: () => request("GET", "/api/runs"),
  run: (id) => request("GET", `/api/runs/${encodeURIComponent(id)}`),
  runMetrics: (id, q) => {
    const qs = new URLSearchParams({ limit: String(q.limit) });
    if (q.after) qs.set("after", q.after);
    if (q.downsample_to) qs.set("downsample_to", String(q.downsample_to));
    return request("GET", `/api/runs/${encodeURIComponent(id)}/metrics?${qs}`);
  },
  startRun: async (workflow, trial) => {
    // a normal run posts the workflow itself (unchanged); a trial wraps it as {workflow, trial}
    const body = trial ? { workflow, trial } : workflow;
    const res = await request<{ graph_run_id: string }>("POST", "/api/runs", body);
    return { run_id: res.graph_run_id };
  },
  cancelRun: (id) => request("POST", `/api/runs/${encodeURIComponent(id)}/cancel`, {}),
  storagePlan: (locale) => request("GET", `/api/storage/plan?locale=${encodeURIComponent(locale)}`),
  storageApply: (body, locale) =>
    request("POST", `/api/storage/apply?locale=${encodeURIComponent(locale)}`, body),
  bugReport: (runId, q, locale) => {
    const qs = new URLSearchParams({ locale });
    if (q.nodeId) qs.set("node_id", q.nodeId);
    if (q.code) qs.set("code", q.code);
    return request(
      "GET",
      runId ? `/api/runs/${encodeURIComponent(runId)}/bug-report?${qs}` : `/api/bug-report?${qs}`,
    );
  },
  advancedOptions: (typeId, model) =>
    request(
      "GET",
      `/api/node-types/${encodeURIComponent(typeId)}/advanced${model ? `?model=${encodeURIComponent(model)}` : ""}`,
    ),
  nodeResetPlan: (node, locale) =>
    request(
      "GET",
      `/api/storage/node-reset?node=${encodeURIComponent(node)}&locale=${encodeURIComponent(locale)}`,
    ),
  nodeReset: (body, locale) =>
    request("POST", `/api/storage/node-reset?locale=${encodeURIComponent(locale)}`, body),
  upload: (file, locale) =>
    request("POST", `/api/uploads?locale=${encodeURIComponent(locale)}`, {
      raw: file,
      contentType: file.type || "application/octet-stream", // a hint only: the server sniffs the bytes
      headers: { "X-Upload-Label": encodeURIComponent(file.name) },
    } satisfies RawBody),
  settings: () => request("GET", "/api/settings"),
  putSecret: (field, value) =>
    request("PUT", `/api/settings/${encodeURIComponent(field)}`, { value }),
  deleteSecret: (field) => request("DELETE", `/api/settings/${encodeURIComponent(field)}`),
  nodeResult: (runId, nodeId) =>
    request(
      "GET",
      `/api/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/result`,
    ),
  health: () => request("GET", "/api/health"),
  lan: () => request("GET", "/api/lan"),
  session: async () => {
    const { csrf_token, ...info } = await request<SessionInfo & { csrf_token?: string }>(
      "GET",
      "/api/session",
    );
    if (typeof csrf_token === "string") csrf = Promise.resolve(csrf_token); // same token, one request fewer
    return info;
  },
  signOut: async () => {
    await request("DELETE", "/api/session");
  },
  revokeAllSessions: async () => {
    await request("POST", "/api/session/revoke-all");
  },
  unlockLan: async () =>
    (await request<{ lan_login?: SessionInfo["lan_login"] }>("POST", "/api/session/unlock-lan"))
      .lan_login,
  addUserModel: (body) => request("POST", "/api/user-assets/models", body),
  userModelResolve: (id) => request("GET", `/api/user-assets/resolves/${encodeURIComponent(id)}`),
  confirmUserModelLicense: (id) =>
    request("POST", `/api/user-assets/models/${encodeURIComponent(id)}/confirm-license`, {
      accept: true,
    }),
  listUserModels: () => request("GET", "/api/user-assets/models"),
  listUserDatasets: () => request("GET", "/api/user-assets/datasets"),
  addUserDataset: (body) => request("POST", "/api/user-assets/datasets", body),
  confirmUserDatasetLicense: (id) =>
    request("POST", `/api/user-assets/datasets/${encodeURIComponent(id)}/confirm-license`, {
      accept: true,
    }),
  userAssetLimits: () => request("GET", "/api/user-assets/limits"),
  setUserAssetLimits: (body) => request("PUT", "/api/user-assets/limits", body),
  inspectDataset: (body) => request("POST", "/api/user-assets/datasets/inspect", body),
  datasetInspection: (id) =>
    request("GET", `/api/user-assets/inspections/${encodeURIComponent(id)}`),
  startDownload: (body) => request("POST", "/api/downloads", body),
  watchDownload: (id, onEvent) =>
    // same handshake as run events (D13): cookie + Origin, then the CSRF token as the first message
    socket(
      `/api/downloads/${encodeURIComponent(id)}/events`,
      (raw) => {
        const ev = raw as DownloadEvent;
        onEvent(ev);
        return ev.type === "status" && (ev.status === "completed" || ev.status === "failed");
      },
      () => onEvent({ type: "connection_lost" }),
    ),
  watchRun: (id, onEvent) => {
    // Browsers cannot set headers on WebSocket: auth = Origin + session cookie at handshake, then the
    // first message carries the CSRF token (P003 D13). Server closes with 4401 if missing/invalid.
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${proto}//${location.host}${at(`/api/runs/${encodeURIComponent(id)}/events`)}?locale=${getLocale()}`,
    );
    let closed = false;
    ws.onopen = () => {
      void csrfToken().then((token) =>
        ws.send(JSON.stringify({ type: "auth", csrf_token: token })),
      );
    };
    let finished = false;
    ws.onmessage = (m) => {
      try {
        const ev = JSON.parse(String(m.data)) as RunEvent;
        if (ev.type === "run_status" && ev.status !== "running") finished = true;
        onEvent(ev);
      } catch {
        /* ignore malformed frames */
      }
    };
    // Never leave nodes silently "pending": a socket that drops (or never opens) before the run ends is reported.
    ws.onclose = (e) => {
      if (e.code === WS_SESSION_ENDED) return sessionEnded(); // signed out: not a lost connection
      if (!closed && !finished) onEvent({ type: "connection_lost" });
    };
    return () => {
      if (!closed) ws.close();
      closed = true;
    };
  },
};
