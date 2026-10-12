// Same-origin API client. Auth is the HttpOnly session cookie set by the server (P003 D13); this wrapper only
// adds the CSRF header on mutating requests. CSRF token: double-submit cookie `vlm_csrf` (readable, not HttpOnly)
// — exact mechanism to be confirmed with bao-backend/bao-security.
import type { NodeTypesResponse, SystemStatus, Template, TemplateMeta, ValidationResult, Workflow } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: unknown,
  ) {
    super(`API ${status}`);
  }
}

function csrfToken(): string {
  const match = /(?:^|;\s*)vlm_csrf=([^;]+)/.exec(document.cookie);
  return match?.[1] ? decodeURIComponent(match[1]) : "";
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") {
    headers["Content-Type"] = "application/json";
    headers["X-VLM-CSRF"] = csrfToken();
  }
  const res = await fetch(path, {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? null : JSON.stringify(body),
  });
  const data: unknown = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new ApiError(res.status, data);
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
