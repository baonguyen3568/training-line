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
  /** Show this param's value on the node face (e.g. chosen model / dataset). */
  summary?: boolean;
  /** Per-option metadata (model catalog): label, tier (Simple hides rnd/baseline), RAM, gated, license. */
  option_meta?: Record<string, OptionMeta>;
}

export interface OptionMeta {
  label: string;
  tier?: "default" | "template" | "catalog" | "baseline" | "rnd";
  ram_gb?: number;
  ram_source?: "measured" | "estimated";
  gated?: boolean;
  license?: string;
  vendor?: string;
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
  /** Gated model: show the "needs Hugging Face account" chip + gated wizard (P010 §16). */
  requires_hf_account?: boolean;
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
  /** Which worker executes nodes; "fake" = simulated (demo banner). */
  worker?: "fake" | "container";
}

import type {
  MetricsPage as GeneratedMetricsPage,
  RunManifest,
  RunStatus,
} from "../generated/schemas";
import type { MetricPoint } from "../lib/metrics";

export type { RunStatus } from "../generated/schemas";

/** All RunStatus values in display order; `satisfies` + the exhaustiveness check keep it equal to the schema. */
export const RUN_STATUSES = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
  "evicted",
  "discarded",
] as const satisfies readonly RunStatus[];
type _AllCovered = Exclude<RunStatus, (typeof RUN_STATUSES)[number]> extends never ? true : never;
export const _runStatusExhaustive: _AllCovered = true;

/** run.json (generated from schemas/run_manifest.schema.json). */
export type RunInfo = RunManifest;

/**
 * GET /api/runs/{id}/metrics?after=<cursor>&limit=<rows>[&downsample_to=<n>] (schemas/metrics_page.schema.json)
 * - `downsample_to` (first load only): whole history reduced to ~n points, `downsampled: true`.
 * - Otherwise a raw page of at most `limit` rows after `cursor`; the client keeps fetching while `has_more`,
 *   regardless of run status, then tails every few seconds only while the run is live.
 */
export type MetricsPage = Omit<GeneratedMetricsPage, "points"> & { points: MetricPoint[] };

export interface MetricsQuery {
  /** Opaque cursor from the previous page (`next_cursor`); omitted on the first request. */
  after?: string;
  limit: number;
  downsample_to?: number;
}

/**
 * WS /api/runs/{graph_run_id}/events (backend app/main.py + graph/executor.py): backlog replayed, then live.
 * Every event carries `graph_run_id` and `ts`.
 */
export type NodeRunStatus =
  | "pending"
  | "running"
  | "completed"
  | "cached"
  | "failed"
  | "interrupted"
  | "skipped";
export interface EventError {
  code: string;
  params?: Record<string, unknown>;
}
export interface NodeStatusEvent {
  type: "node_status";
  node_id: string;
  status: NodeRunStatus;
  progress?: number; // 0..1 when the worker reports it
  error?: EventError;
}
export interface RunStatusEvent {
  type: "run_status";
  status: "running" | "invalid" | RunStatus;
  nodes?: string[]; // execution order, on "running"
  error?: EventError;
}
export interface OtherEvent {
  /** `connection_lost` is synthesized by the client when the event socket closes before the run ends. */
  type: "warning" | "run_not_found" | "connection_lost" | "backlog_truncated";
  node_id?: string;
  error?: EventError;
}
export type RunEvent = NodeStatusEvent | RunStatusEvent | OtherEvent;
