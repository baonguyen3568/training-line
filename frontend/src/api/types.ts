// INTERIM: hand-written until backend exports schemas/ (P004 §8.2) — replace with src/generated/api.ts.
// API types for P009 §2–§6. Hand-written until the backend exports `schemas/` (OpenAPI + NodeSpec JSON Schema);
// then they are replaced by openapi-typescript output in src/generated/ and this file re-exports from there.

export type Category =
  | "source"
  | "data"
  | "train"
  | "test"
  | "quantize"
  | "eval"
  | "export"
  | "util";
export type TrustLevel = "first_party" | "reviewed" | "unverified";
export type PermissionLevel = "info" | "warning" | "critical";
export type Severity = "error" | "warn" | "info";
export type EstimateSource = "measured" | "formula" | "declared" | "unknown";

export interface PortType {
  id: string; // "Dataset", "Model", "QuantizedModel"…
  parent: string | null; // subtype → supertype ("QuantizedModel" → "Model")
  color_token: string; // e.g. "color.port.model"
  label_key: string; // e.g. "port.quantized_model"
}

export interface PortSpec {
  type: string;
  optional?: boolean;
  many?: boolean;
}

/** UI hints carried on each JSON Schema property (`x-ui`, P004 §8.2). */
export interface UiHint {
  widget?: "slider" | "select" | "segmented" | "text" | "number" | "toggle";
  order?: number;
  preset?: boolean;
  advanced?: boolean;
  /** i18n prefix for enum option labels, e.g. "train_preset" → train_preset.<value>. */
  option_keys?: string;
}

export interface ParamSchema {
  type: "string" | "number" | "integer" | "boolean" | "array";
  enum?: (string | number)[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
  "x-ui"?: UiHint;
}

export interface ParamsSchema {
  type: "object";
  properties: Record<string, ParamSchema>;
  required?: string[];
}

export interface Permission {
  id: "gpu" | "network" | "runtime" | "memory";
  level: PermissionLevel;
  requires_typed_confirm: boolean;
  perm_copy_key: string;
  confirm_copy_key: string;
}

export interface NodeOrigin {
  kind: "first_party" | "third_party";
  publisher?: string;
  pack?: string;
  version?: string;
  commit_sha?: string;
  license?: string;
  installed_at?: string;
  reviewed?: boolean;
}

export interface NodeSpec {
  type_id: string;
  /** Resolved by the server for `?locale=` (P009 §2.2); absent in mock mode → looked up locally. */
  title?: string;
  desc?: string;
  version: string;
  category: Category;
  icon: string;
  inputs: Record<string, PortSpec>;
  outputs: Record<string, PortSpec>;
  params_schema: ParamsSchema;
  cacheable: boolean;
  runner: "host" | "container";
  origin: NodeOrigin;
  permissions: Permission[];
  policy_block: { code: string; reason_key: string } | null;
}

export interface NodeTypesResponse {
  node_types: NodeSpec[];
  port_types: PortType[];
}

export interface WorkflowNode {
  id: string;
  type: string;
  version: string;
  params: Record<string, unknown>;
  ui?: { x: number; y: number };
}

export interface WorkflowEdge {
  from: [string, string]; // [node_id, output port]
  to: [string, string]; // [node_id, input port]
}

export interface Workflow {
  format: "vlm-workflow";
  schema_version: 1;
  name: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface TemplateMeta {
  id: string;
  title_key: string;
  description_key: string;
  difficulty: "basic" | "advanced";
  tags: string[];
  sample_data: boolean;
  estimate: Estimate;
}

export interface Template extends TemplateMeta {
  workflow: Workflow;
}

export interface Range {
  lo: number;
  hi: number;
}

export interface Estimate {
  ram_gb: Range | null;
  disk_gb: Range | null;
  minutes: Range | null;
  gpu_exclusive: boolean;
  source: EstimateSource;
}

export interface ValidationIssue {
  node_id?: string;
  port?: string;
  param?: string;
  code: string;
  severity: Severity;
  params: Record<string, string | number>;
  title: string;
  what: string;
  why?: string;
  action_ids: string[];
}

export interface ValidationResult {
  ok: boolean;
  errors: ValidationIssue[];
  estimate: {
    ram_gb_peak: Range | null;
    disk_gb: Range | null;
    duration_min: Range | null;
    gpu_exclusive: boolean;
    source: EstimateSource;
  };
}

export interface SystemStatus {
  mem_total_gb: number;
  mem_available_gb: number;
  gpu_lock: { held: boolean; who?: string; job?: string; since?: string; stale?: boolean };
  disk_free_gb: number;
}

/** Mirrors `common/runinfo.RunStatus` (source of truth, DECISIONS.md); drift-checked in tests/run-status.test.ts. */
export const RUN_STATUSES = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
  "evicted",
  "discarded",
] as const;
export type RunStatus = (typeof RUN_STATUSES)[number];

export interface RunInfo {
  schema_version: number;
  run_id: string;
  name?: string;
  task: string;
  stage?: string;
  base_model?: string;
  status: RunStatus;
  started_at: number;
  ended_at?: number | null;
  parent_run_id?: string | null;
  container_image?: string;
  max_steps?: number | null;
  tags?: string[];
}

/** GET /api/runs/{id}/metrics?after=<cursor>&max_points=<n> — incremental tail of metrics.jsonl. */
/**
 * GET /api/runs/{id}/metrics?after=<cursor>&limit=<rows>[&downsample_to=<n>]
 * - `downsample_to` (first load only): server returns the whole history reduced to ~n points, `downsampled: true`.
 * - Otherwise a raw page of at most `limit` rows after `cursor`; the client keeps fetching while `has_more`,
 *   regardless of run status, then tails every few seconds only while the run is live.
 */
export interface MetricsPage {
  points: import("../lib/metrics").MetricPoint[];
  next_cursor: number; // opaque to the UI (backend: byte offset); pass back as `after`
  has_more: boolean;
  downsampled: boolean;
}

export interface MetricsQuery {
  after: number;
  limit: number;
  downsample_to?: number;
}
