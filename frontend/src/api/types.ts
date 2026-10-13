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
  widget?:
    | "slider"
    | "select"
    | "segmented"
    | "text"
    | "number"
    | "toggle"
    | "textarea"
    | "image_upload"
    | "advanced"; // P016 §3 engine options (object); widget not built yet
  order?: number;
  preset?: boolean;
  advanced?: boolean;
  /** i18n prefix for enum option labels, e.g. "train_preset" → train_preset.<value>. */
  option_keys?: string;
  /** Show this param's value on the node face (e.g. chosen model / dataset). */
  summary?: boolean;
  /** Per-option metadata (model catalog): label, tier (Simple hides rnd/baseline), RAM, gated, license. */
  option_meta?: Record<string, OptionMeta>;
  /** Values the worker cannot run yet: shown dimmed and not selectable. Value → i18n reason key (null = generic).
   * A value already in an old file stays shown (never silently changed); the STD-001 error still blocks Run. */
  disabled_options?: Record<string, string | null>;
  /** The value the app recommends (e.g. epochs 3): its option reads "· khuyên dùng". */
  recommended?: string | number;
  /** Show only while these params of the same node hold one of the values (AND across keys, OR in a list). */
  visible_when?: Record<string, (string | number | boolean)[]>;
}

export interface OptionMeta {
  label: string;
  tier?: "default" | "template" | "catalog" | "baseline" | "rnd";
  ram_gb?: number;
  ram_source?: "measured" | "estimated";
  gated?: boolean;
  license?: string;
  vendor?: string;
  /** The publisher as people write it ("NVIDIA", "Qwen (Alibaba)"); `vendor` is a slug and is never shown. */
  vendor_name?: string;
  /** The model always does one task (Nemotron-Parse): the "Chạy thử" question is ignored, so the UI hides it. */
  task_prompt_fixed?: boolean;
}

export interface ParamSchema {
  type: "string" | "number" | "integer" | "boolean" | "array" | "object"; // object: P016 `advanced`
  enum?: (string | number)[];
  maxLength?: number;
  pattern?: string;
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
  /** P016 subgraph definitions; instances are ordinary nodes whose `type` is a group id. Validated by the backend. */
  groups?: GroupDefinition[];
}

/** P016 group (subgraph) definition, as stored in the workflow file. */
export interface GroupDefinition {
  id: string;
  version: string;
  title: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  inputs: Record<string, [string, string][]>;
  outputs: Record<string, [string, string]>;
  params: Record<string, [string, string]>;
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
  /** Gallery order (lower first); the default template is first. */
  order?: number;
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

/**
 * G-3: an action the server already resolved for this issue (error_catalog.resolved_actions, 6c780e0): filtered,
 * bounded, final values. `patch: null` with kind "patch" is a graph operation named by `id`.
 */
export interface ResolvedAction {
  id: string;
  kind: "patch" | "resume" | "navigate";
  confirm: boolean;
  target?: string | null;
  node_id?: string;
  /** `from` is what the server saw; apply only while the param still holds it (else the hint is stale). */
  patch?: { param: string; from: unknown; value: unknown }[] | null;
}

export interface ValidationIssue {
  node_id?: string;
  /** G-3 resolved actions; older servers send only action_ids. */
  actions?: ResolvedAction[];
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
    /** Provenance of RAM/disk only; `duration` carries its own (measured runs). */
    source: EstimateSource;
    /** Measured training time; null on the fake worker; absent on servers that predate it. */
    duration?: DurationEstimate | null;
  };
}

/** Train-node time measured from earlier runs on this machine; never summed past a missing node. */
export interface DurationEstimate {
  /** Train nodes only; null as soon as one train node has no measurement. */
  duration_min: Range | null;
  per_node: Record<
    string,
    {
      steps: number | null;
      minutes: Range | null;
      based_on: string[];
      /** Same measurement per `epochs` option ("1" | "2" | "3" | "5"), for the epochs picker. */
      by_epochs?: Record<string, Range | null>;
    }
  >;
  /** Heavy non-train nodes (eval/quantize/bench/serve/merge) whose time is not counted yet. */
  unmeasured: string[];
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
  /** Rendered in the WS locale when the server does it (events ?locale=); else the client renders a basic text. */
  title?: string;
  what?: string;
  why?: string | null;
  /** G-3 resolved quick fixes (a3f5bbb), same shape as validate issues. */
  actions?: ResolvedAction[];
  action_ids?: string[];
  already_optimized?: boolean;
}
export interface NodeStatusEvent {
  type: "node_status";
  node_id: string;
  status: NodeRunStatus;
  progress?: number; // 0..1 when the worker reports it
  error?: EventError;
  /** G-4: "<nid>-<key16>", the node's run-dir base that "Đặt lại bước" acts on (every event with a cache key). */
  node_base?: string;
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
/** D5: a running node queued for memory (or the GPU lock) before its job really starts (executor `waiting`). */
export interface NodeStateEvent {
  type: "node_state";
  node_id: string;
  state: "waiting_memory" | "waiting_gpu";
  need_gb?: number;
  avail_gb?: number;
  since?: number;
  /** Seconds waited so far (run_job queueing, 287ea77). */
  waited_s?: number;
}
export type RunEvent = NodeStatusEvent | RunStatusEvent | NodeStateEvent | OtherEvent;

/**
 * Storage panel (bao-cleanup app/storage.py, P013 G-1, plan schema v3). Hand-written: no JSON schema is exported
 * for it yet. Text for the user comes as i18n keys + params, except `error` which the server renders.
 */
/** G-4 "Đặt lại bước" (bao-cleanup app/storage_reset.py): read-only plan, then a one-shot apply. */
export interface NodeResetPlan {
  schema_version: number;
  /** null when blocked or there is nothing to delete. */
  plan_id: string | null;
  expires_at: number;
  node: string;
  node_title: string;
  attempts: { run_dir: string; status: string; size_gb: number; has_checkpoint: boolean }[];
  total_gb: number;
  /** storage.lock_reason.<code> (RUN_NOT_FINISHED, GPU_LOCK_HOLDER, … UNKNOWN_STATUS); params always carry node. */
  blocked_by: string | null;
  blocked_params: Record<string, string | number>;
  consequence_code: "PERMANENT_LOSS" | null;
  consequence_params: { name?: string };
  confirm_required: boolean;
  /** The node title; the server matches it ignoring accents, case and repeated spaces. */
  confirm_text: string | null;
  next_generation: number;
}
/**
 * "Gửi báo cáo lỗi" (design ba32c2c, security-approved fields): only these, any of them may be absent. Pasted into
 * a PUBLIC GitHub issue, so no logs, paths, tokens or user data.
 */
export interface BugReport {
  code?: string;
  error_title?: string;
  app_version?: string;
  gpu?: string;
  /** Short hashed reference the team can ask about; never the run's (user-named) id (design 262b579). */
  ref_code?: string;
}

/** P016 §3 engine option (GET /api/node-types/{id}/advanced?model=), from configs/engine/automodel-*.json. */
export interface AdvancedOption {
  key: string;
  group: string;
  type: "bool" | "int" | "float" | "enum" | "list";
  default: unknown;
  choices: (string | number)[] | null;
  min?: number | null;
  max?: number | null;
  templates: string[];
}

/** G-5 (P018): POST /api/downloads. The server resolves repo/revision from the catalog; a client never sends one. */
export interface DownloadRequest {
  kind: "model" | "dataset";
  catalog_id: string;
  confirm?: boolean;
  check?: boolean;
}
export interface DownloadStart {
  download_id: string | null;
  status?: "queued" | "completed";
  /** completed: false = in the cache but not checked by the app yet ("Kiểm tra" re-sends with check). */
  verified?: boolean;
  /** Over the size threshold: re-send with confirm. */
  needs_confirm?: boolean;
  name?: string;
  size_gb?: number;
  threshold_gb?: number;
  license?: string;
  gated?: boolean;
}
export type DownloadEvent =
  | {
      type: "progress";
      files_done: number;
      files_total: number;
      bytes_done: number;
      bytes_total: number;
    }
  | {
      type: "status";
      status: "downloading" | "verifying" | "completed" | "failed";
      error?: EventError;
    }
  | { type: "download_not_found" | "connection_lost" };

export interface NodeResetResult {
  deleted: string[];
  failed: string[];
  freed_gb: number;
  generation: number;
}

export type StorageSection = "auto" | "manual" | "kept";
export interface StorageItem {
  item_id: string;
  /** G-4: the graph node this item belongs to ("<nid>-<key16>", every generation/attempt), null for CLI runs. */
  node_base?: string | null;
  source: "node_cache" | "checkpoint" | "set_aside";
  section: StorageSection;
  auto: boolean;
  group: string;
  display_name: string | null;
  display_key: string;
  display_params: Record<string, string | number>;
  size_gb: number | null;
  last_used?: number | null;
  run_id: string;
  node_type?: string;
  action: "delete" | "keep";
  protected_reason: string | null;
  lock_reason?: string | null;
  lock_params?: Record<string, string | number>;
  regenerable: boolean;
  consequence_code: string | null;
  consequence_params: Record<string, string | number>;
  confirm_required: boolean;
  confirm_text: string | null;
}
export interface RenderedError {
  code: string;
  params?: Record<string, unknown>;
  title: string;
  what: string;
  why?: string | null;
}
export interface StoragePlan {
  schema_version: number;
  plan_id: string;
  expires_at: number;
  approval_policy: "none" | "threshold";
  threshold_gb: number | null;
  cap_gb: number;
  cache_gb: number;
  auto_reclaim_gb: number;
  shortfall_gb: number;
  items: StorageItem[];
  /** Reminders that need the user soon (P013 G-1), e.g. an interrupted run with ≤ 24 h left to resume. */
  attention?: StorageAttention[];
  error: RenderedError | null;
}

export interface StorageAttention {
  code: string;
  run_id: string;
  /** Node title (i18n), else the run's name, else its id. */
  node: string;
  /** Local time "YYYY-MM-DD HH:MM". */
  resume_until: string;
  /** Server-rendered in the request locale; used only when the app has no copy for `code`. */
  message: string;
}
export interface StorageApplyRequest {
  plan_id: string;
  item_ids: string[];
  confirm?: Record<string, string>;
}
export interface StorageApplyResult {
  evicted: unknown[];
  pruned: unknown[];
  skipped: { item_id: string; reason: string }[];
  freed_gb: number;
}

/** B-7 (backend app/uploads.py): POST /api/uploads response. `label` is the client's file name: untrusted. */
export interface UploadInfo {
  upload_id: string;
  label: string;
  width: number;
  height: number;
  size_bytes: number;
}
/** GET /api/runs/{id}/nodes/{node}/result for serve.* ("Chạy thử"). Written by a container: untrusted text. */
export interface TrialResult {
  engine_used: string;
  engine_requested?: string;
  text: string;
  n_tokens: number;
  truncated: boolean;
  decoding: Record<string, unknown>;
  overrides: Record<string, unknown>;
  model_class?: string;
}

/** D-7 settings (backend, P015 §4.5). The token value never leaves the server again. */
export type SecretField = "HF_TOKEN" | "NGC_API_KEY" | "WANDB_API_KEY";
export interface SecretState {
  set: boolean;
  updated_at: number | null;
  last4: string;
  readable: boolean;
  scope: "unchecked" | "read" | "write";
}
export interface SettingsView {
  secrets: Record<SecretField, SecretState>;
  /** Deployment policy: read-only in the UI (403 settings_admin_only on write). source "default" = value null:
   * the app's built-in default applies. reason_key may come later (optional). */
  admin_policy: Record<
    string,
    {
      value: unknown;
      source: "deployment" | "default";
      reason_key?: string;
      /** e.g. "APPROVAL_OFF_SHARED": a risky value on a shared machine → settings.policy_warning.<code>. */
      warning?: string;
    }
  >;
  key_ok: boolean;
}

/** D-8 GET /api/session without its CSRF token (that stays in client.ts memory). */
export interface SessionInfo {
  origin_class: "local" | "lan";
  /** Local session only: the LAN sign-in lock (too many wrong tokens); epoch seconds or null. */
  lan_login?: { locked_until: number | null; recent_failures: number };
}
