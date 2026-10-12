// Same-origin API client. Auth is the HttpOnly session cookie `vlm_session` set by the server (P003 D13).
// CSRF: the token comes from `GET /api/session` (JSON body) and is kept ONLY in module memory — never a
// readable cookie (cookies are not port-isolated on 127.0.0.1, so other local apps could toss/overwrite them)
// and never localStorage. It changes on every server restart; a 403 csrf_invalid triggers one refresh + retry.
import type {
  NodeTypesResponse,
  SystemStatus,
  Template,
  TemplateMeta,
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

let csrf: Promise<string> | null = null;

function csrfToken(refresh = false): Promise<string> {
  if (refresh || !csrf) {
    csrf = fetch("/api/session", {
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

async function request<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
  retried = false,
): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") {
    headers["Content-Type"] = "application/json";
    headers["X-VLM-CSRF"] = await csrfToken(retried);
  }
  const res = await fetch(path, {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? null : JSON.stringify(body),
  });
  const data: unknown = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) {
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
}

export const httpApi: Api = {
  nodeTypes: (locale) => request("GET", `/api/node-types?locale=${encodeURIComponent(locale)}`),
  templates: () => request("GET", "/api/templates"),
  template: (id) => request("GET", `/api/templates/${encodeURIComponent(id)}`),
  validate: (workflow, locale) =>
    request("POST", `/api/graphs/validate?locale=${encodeURIComponent(locale)}`, workflow),
  system: () => request("GET", "/api/system"),
};
