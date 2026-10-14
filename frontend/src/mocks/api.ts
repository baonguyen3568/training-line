// In-browser mock of the P009 API for development (VITE_MOCK_API=1) and tests. It mirrors the documented
// shapes only; the real validation lives in the backend (`POST /api/graphs/validate`) and is the source of truth.

import type { Api } from "../api/client";
import { ApiError } from "../api/client";
import type {
  AdvancedOption,
  BugReport,
  DatasetInspection,
  DownloadStart,
  HealthInfo,
  LanInfo,
  NodeResetPlan,
  NodeResetResult,
  NodeSpec,
  NodeTypesResponse,
  ParamSchema,
  ParamsSchema,
  PortType,
  RunInfo,
  SecretState,
  SessionInfo,
  SettingsView,
  StorageApplyResult,
  StorageItem,
  StoragePlan,
  Template,
  TrialResult,
  UploadInfo,
  UserAssetLimits,
  UserDataset,
  UserModel,
  ValidationIssue,
  ValidationResult,
  Workflow,
} from "../api/types";
import groupExamples from "../generated/group-examples.json";
import registrySnapshot from "../generated/node-types.json";
import templatesBundle from "../generated/templates.json";
import { tp, tpOptional } from "../i18n";
import { type GroupTypeEntry, specsFromCatalog, withUserGroups } from "../lib/groups";
import { parseJsonl } from "../lib/metrics";
import { isCompatible } from "../lib/ports";
import { stepNumbers } from "../lib/topo";
import advancedFixture from "./fixtures/advanced-options.json";
import s3aMetrics from "./fixtures/s3a.metrics.jsonl?raw";
import s3aRun from "./fixtures/s3a.run.json";
import storagePlanSample from "./fixtures/storage-plan.v3.json";

// First-party groups (016b) for the mock: cleancode's golden fixtures, as the server derives them; titles from i18n
// when a key exists, else a demo name (the real groups ship their own groups.<slug> copy).
const MOCK_GROUP_TITLE: Record<string, string> = {
  "group.example_lora": "Fine-tune LoRA rồi gộp",
  "group.example_outer": "Fine-tune và chấm điểm",
};
function mockGroups(): GroupTypeEntry[] {
  return Object.values(
    groupExamples as unknown as Record<string, { expected: Record<string, unknown> }>,
  )
    .map((e) => e.expected)
    .filter((e) => !("error" in e))
    .map((e) => {
      const entry = e as unknown as GroupTypeEntry;
      const key = entry.title_key?.split(".");
      return {
        ...entry,
        title: (key && tpOptional(key)) ?? MOCK_GROUP_TITLE[entry.group_id] ?? entry.group_id,
      };
    });
}

const mockJobs = new Map<
  string,
  {
    polls: number;
    repo: string;
    revision?: string | undefined;
    config?: string | undefined;
    split?: string | undefined;
  }
>();
let mockJobSeq = 0;
const mockUserModels = new Map<string, UserModel>();
const mockUserDatasets = new Map<string, UserDataset>();
const mockLimits: UserAssetLimits = {
  max_model_gb: 40,
  max_model_gb_range: [1, 1000],
  default_max_model_gb: 40,
};

const mockSession: SessionInfo = {
  origin_class: "local",
  lan_login: { locked_until: Math.floor(Date.now() / 1000) + 15 * 60, recent_failures: 5 },
};

const portTypes: PortType[] = [
  { id: "Dataset", parent: null, color_token: "color.port.dataset", label_key: "port.dataset" },
  { id: "Model", parent: null, color_token: "color.port.model", label_key: "port.model" },
  { id: "Adapter", parent: null, color_token: "color.port.model", label_key: "port.adapter" },
  {
    id: "Checkpoint",
    parent: "Model",
    color_token: "color.port.model",
    label_key: "port.checkpoint",
  },
  {
    id: "QuantizedModel",
    parent: "Model",
    color_token: "color.port.model",
    label_key: "port.quantized_model",
  },
  {
    id: "ExportArtifact",
    parent: null,
    color_token: "color.port.model",
    label_key: "port.export_artifact",
  },
  {
    id: "EvalReport",
    parent: null,
    color_token: "color.port.report",
    label_key: "port.eval_report",
  },
  {
    id: "BenchReport",
    parent: null,
    color_token: "color.port.report",
    label_key: "port.bench_report",
  },
  { id: "Endpoint", parent: null, color_token: "color.port.endpoint", label_key: "port.endpoint" },
];

const noParams: ParamsSchema = { type: "object", properties: {} };
const firstParty = { kind: "first_party" as const };

function spec(partial: Partial<NodeSpec> & Pick<NodeSpec, "type_id" | "category">): NodeSpec {
  return {
    version: "1.0.0",
    icon: partial.category,
    inputs: {},
    outputs: {},
    params_schema: noParams,
    cacheable: true,
    runner: "container",
    origin: firstParty,
    permissions: [],
    policy_block: null,
    ...partial,
  };
}

/** Hand-written specs for unit/security tests (third-party, payloads). The mock API serves `registry`. */
export const mockNodeTypes: NodeSpec[] = [
  spec({
    type_id: "model.hf_pick",
    category: "source",
    runner: "host",
    outputs: { model: { type: "Model" } },
    // Mirrors the real node (afdc074): `model` = catalog id, option_meta per option, Simple hides rnd/baseline.
    params_schema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          enum: [
            "nemotron-parse-v1.1",
            "cosmos-reason2-2b",
            "qwen3.5-0.8b",
            "qwen3.5-2b",
            "nemotron-parse-2.0",
          ],
          default: "nemotron-parse-v1.1",
          "x-ui": {
            summary: true,
            option_meta: {
              "nemotron-parse-v1.1": {
                label: "NVIDIA Nemotron-Parse v1.1",
                tier: "default",
                ram_gb: 24,
                ram_source: "estimated",
                gated: false,
                license: "NVIDIA Open Model License",
                vendor: "NVIDIA",
              },
              "cosmos-reason2-2b": {
                label: "NVIDIA Cosmos-Reason2 2B",
                tier: "template",
                ram_gb: 26,
                ram_source: "estimated",
                gated: true,
                license: "NVIDIA Open Model License",
                vendor: "NVIDIA",
              },
              "qwen3.5-0.8b": {
                label: "Qwen3.5 0.8B",
                tier: "baseline",
                ram_gb: 9,
                ram_source: "measured",
                gated: false,
                license: "Apache-2.0",
                vendor: "Qwen",
              },
              "qwen3.5-2b": {
                label: "Qwen3.5 2B",
                tier: "catalog",
                ram_gb: 18,
                ram_source: "estimated",
                gated: false,
                license: "Apache-2.0",
                vendor: "Qwen",
              },
              "nemotron-parse-2.0": {
                label: "NVIDIA Nemotron-Parse 2.0",
                tier: "rnd",
                ram_gb: 30,
                ram_source: "estimated",
                gated: false,
                license: "NVIDIA Open Model License",
                vendor: "NVIDIA",
              },
            },
          },
        },
      },
      required: ["model"],
    },
  }),
  spec({
    type_id: "data.local_folder",
    category: "source",
    runner: "host",
    outputs: { dataset: { type: "Dataset" } },
    params_schema: {
      type: "object",
      properties: {
        path: { type: "string", default: "data/sample-invoices", "x-ui": { summary: true } },
      },
      required: ["path"],
    },
  }),
  spec({
    type_id: "data.hf_load",
    category: "source",
    runner: "host",
    outputs: { dataset: { type: "Dataset" } },
    params_schema: {
      type: "object",
      properties: {
        repo: {
          type: "string",
          enum: ["naver-clova-ix/cord-v2"],
          default: "naver-clova-ix/cord-v2",
          "x-ui": { summary: true },
        },
      },
      required: ["repo"],
    },
  }),
  spec({
    type_id: "data.split",
    category: "data",
    runner: "host",
    inputs: { dataset: { type: "Dataset" } },
    outputs: { train: { type: "Dataset" }, val: { type: "Dataset" } },
    params_schema: {
      type: "object",
      properties: {
        val_fraction: {
          type: "number",
          minimum: 0.05,
          maximum: 0.5,
          default: 0.1,
          "x-ui": { widget: "slider" },
        },
      },
    },
  }),
  spec({
    type_id: "train.lora_vlm",
    category: "train",
    inputs: {
      base_model: { type: "Model" },
      train_data: { type: "Dataset" },
      val_data: { type: "Dataset", optional: true },
    },
    outputs: { adapter: { type: "Adapter" } },
    permissions: [],
    params_schema: {
      type: "object",
      properties: {
        preset: {
          type: "string",
          enum: ["fast", "balanced", "quality"],
          default: "balanced",
          "x-ui": { widget: "segmented", order: 0, preset: true, option_keys: "train_preset" },
        },
        epochs: { type: "integer", minimum: 1, maximum: 10, default: 2, "x-ui": { order: 1 } },
        learning_rate: { type: "number", default: 0.0001, "x-ui": { advanced: true, order: 2 } },
      },
    },
  }),
  spec({
    type_id: "train.sft_vlm",
    category: "train",
    inputs: {
      base_model: { type: "Model" },
      train_data: { type: "Dataset" },
      val_data: { type: "Dataset", optional: true },
    },
    outputs: { model: { type: "Checkpoint" } },
    params_schema: {
      type: "object",
      properties: {
        epochs: { type: "integer", minimum: 1, maximum: 10, default: 2, "x-ui": { order: 1 } },
        learning_rate: { type: "number", default: 0.00002, "x-ui": { advanced: true, order: 2 } },
      },
    },
  }),
  spec({
    type_id: "train.merge_lora",
    category: "train",
    inputs: { base_model: { type: "Model" }, adapter: { type: "Adapter" } },
    outputs: { model: { type: "Checkpoint" } },
  }),
  spec({
    type_id: "quantize.modelopt_ptq",
    category: "quantize",
    inputs: { model: { type: "Model" } },
    outputs: { quantized: { type: "QuantizedModel" } },
    params_schema: {
      type: "object",
      properties: {
        format: {
          type: "string",
          enum: ["fp8", "nvfp4", "int4_awq"],
          default: "fp8",
          "x-ui": { widget: "segmented", preset: true, option_keys: "quantize_preset" },
        },
      },
    },
  }),
  spec({
    type_id: "eval.benchmarks",
    category: "eval",
    inputs: { model: { type: "Model" } },
    outputs: { report: { type: "EvalReport" } },
    params_schema: {
      type: "object",
      properties: {
        groups: {
          type: "array",
          enum: ["documents", "charts", "reasoning", "perception", "math", "video"],
          default: ["documents"],
          "x-ui": { option_keys: "benchmark_group" },
        },
      },
    },
  }),
  // Third-party example used to exercise badges/permissions (F6d). Never shipped.
  spec({
    type_id: "acme.ocr_clean",
    category: "data",
    inputs: { dataset: { type: "Dataset" } },
    outputs: { dataset: { type: "Dataset" } },
    origin: {
      kind: "third_party",
      publisher: "acme",
      pack: "vision-tools",
      version: "0.3.1",
      reviewed: false,
    },
    permissions: [
      {
        id: "gpu",
        level: "critical",
        requires_typed_confirm: true,
        perm_copy_key: "custom_node.perm.gpu",
        confirm_copy_key: "custom_node.typed_confirm_gpu_unverified",
      },
    ],
    policy_block: { code: "POLICY", reason_key: "custom_node.blocked_reason.policy_block" },
  }),
];

const unknownEstimate = {
  ram_gb: null,
  disk_gb: null,
  minutes: null,
  gpu_exclusive: true,
  source: "unknown" as const,
};

const unsetSecret = (): SecretState => ({
  set: false,
  updated_at: null,
  last4: "",
  readable: true,
  scope: "unchecked",
});
const mockSettings: SettingsView = {
  secrets: { HF_TOKEN: unsetSecret(), NGC_API_KEY: unsetSecret(), WANDB_API_KEY: unsetSecret() },
  admin_policy: {
    "deployment.profile": { value: "shared_dev", source: "deployment" },
    "storage.schedule": { value: "dry_run", source: "deployment" },
    "storage.approval_policy": { value: "threshold", source: "deployment" },
    "storage.approval_threshold_gb": { value: 20, source: "deployment" },
    "plugins.allowlist": { value: null, source: "default" },
  },
  key_ok: true,
};

// Templates come from configs/templates/*.json (same files the backend serves); the server adds estimates and,
// like the server (app/templates.py), never offers a hidden one (e.g. quantize_bench until its worker can run it).
const templates: Template[] = (
  templatesBundle as (Omit<Template, "estimate"> & { hidden?: boolean })[]
)
  .filter((tpl) => !tpl.hidden)
  .map(({ hidden: _hidden, ...tpl }) => ({ ...tpl, estimate: unknownEstimate }));

function issue(
  code: string,
  partial: Partial<ValidationIssue>,
  params: Record<string, string | number>,
): ValidationIssue {
  return {
    code,
    severity: "error",
    params,
    title: tp(["error", code, "title"]),
    what: tp(["error", code, "what"], params),
    action_ids: [],
    ...partial,
  };
}

/** Minimal mirror of validate steps 1–2 (structure, types) so the UI can be exercised without a backend. */
export function mockValidate(
  workflow: Workflow,
  types: NodeSpec[],
  ports: PortType[],
): ValidationResult {
  const byType = new Map(types.map((s) => [s.type_id, s]));
  const nodeTitle = (typeId: string) => byType.get(typeId)?.title ?? tp(["node", typeId, "title"]);
  const errors: ValidationIssue[] = [];
  for (const node of workflow.nodes) {
    const nodeSpec = byType.get(node.type);
    if (!nodeSpec) {
      errors.push(issue("CFG-004", { node_id: node.id }, { node_type: node.type }));
      continue;
    }
    if (nodeSpec.policy_block) {
      errors.push({
        ...issue(nodeSpec.policy_block.code, { node_id: node.id }, {}),
        title: tp(["custom_node", "badge", "policy_block"]),
        what: tp(nodeSpec.policy_block.reason_key.split(".")),
      });
    }
    for (const [port, ps] of Object.entries(nodeSpec.inputs)) {
      const connected = workflow.edges.some((e) => e.to[0] === node.id && e.to[1] === port);
      if (!ps.optional && !connected) {
        errors.push(
          issue("CFG-003", { node_id: node.id, port }, { node: nodeTitle(node.type), port }),
        );
      }
    }
  }
  for (const edge of workflow.edges) {
    const src = workflow.nodes.find((n) => n.id === edge.from[0]);
    const dst = workflow.nodes.find((n) => n.id === edge.to[0]);
    const outType = src && byType.get(src.type)?.outputs[edge.from[1]]?.type;
    const inType = dst && byType.get(dst.type)?.inputs[edge.to[1]]?.type;
    if (src && dst && outType && inType && !isCompatible(outType, inType, ports)) {
      errors.push(
        issue(
          "CFG-001",
          { node_id: dst.id, port: edge.to[1] },
          { a: nodeTitle(src.type), b: nodeTitle(dst.type), type_a: outType, type_b: inType },
        ),
      );
    }
  }
  // ADR-021 MODEL-005: a train step fed by a user-added model gets the catalog's "try a quick run first" hint
  for (const node of workflow.nodes) {
    // like the server: a trial run exists for LoRA / full SFT only (DPO, GRPO, merge get no MODEL-005)
    if (node.type !== "train.lora_vlm" && node.type !== "train.sft_vlm") continue;
    const src = workflow.edges
      .filter((e) => e.to[0] === node.id)
      .map((e) => workflow.nodes.find((n) => n.id === e.from[0]))
      .find((n) => typeof n?.params.user_model === "string" && n.params.user_model);
    if (!src) continue;
    // like the server: only for a model this machine added, and only until its trial ran
    const rec = mockUserModels.get(String(src.params.user_model));
    if (!rec || rec.trial_run_id) continue;
    const model = String(src.params.user_model).split("--")[2] ?? String(src.params.user_model);
    errors.push(
      issue(
        "MODEL-005",
        {
          node_id: node.id,
          severity: "info",
          action_ids: ["run_smoke"],
          actions: [
            { id: "run_smoke", kind: "trial", confirm: false, node_id: node.id, target: null },
          ],
        },
        { model, node: nodeTitle(node.type) },
      ),
    );
  }
  const heavy = workflow.nodes.some((n) => byType.get(n.type)?.runner === "container");
  // Never invent numbers: reuse the template's own estimate (unknown until measured), else unknown.
  const est = templates.find((x) => x.id === workflow.name)?.estimate;
  return {
    ok: !errors.some((e) => e.severity === "error"),
    errors,
    estimate: {
      ram_gb_peak: est?.ram_gb ?? null,
      disk_gb: est?.disk_gb ?? null,
      duration_min: est?.minutes ?? null,
      gpu_exclusive: heavy,
      source: est?.source ?? "unknown",
      duration: null, // like the server's fake worker: no measured runs behind the demo
    },
  };
}

const delay = <T>(value: T) =>
  new Promise<T>((r) => setTimeout(() => r(structuredClone(value)), 50));

export const mockApi: Api = {
  // The third-party fixture (acme.*) exists for security tests only; the library must not offer it.
  nodeTypes: () =>
    delay<NodeTypesResponse>({
      node_types: registry.node_types,
      port_types: registry.port_types,
      groups: mockGroups(),
    }),
  templates: () => delay(templates.map(({ workflow: _w, ...meta }) => meta)),
  template: (id) => {
    const found = templates.find((x) => x.id === id);
    return found ? delay(found) : Promise.reject(new Error(`template ${id} not found`));
  },
  validate: (workflow) =>
    delay(
      mockValidate(
        workflow,
        // node types + first-party groups + the file's own user.* groups, as the server expands them
        [
          ...withUserGroups(
            specsFromCatalog(registry.node_types, mockGroups()),
            workflow.groups,
          ).values(),
        ],
        registry.port_types,
      ),
    ),
  system: () =>
    delay({
      mem_total_gb: 121,
      mem_available_gb: 42,
      gpu_lock: { held: false },
      disk_free_gb: 2000,
    }),
  runs: () => delay([s3aRun as RunInfo]),
  run: (id) =>
    id === s3aRun.run_id
      ? delay(s3aRun as RunInfo)
      : Promise.reject(new Error(`run ${id} not found`)),
  // Real fixture: S-3 smoke run (AutoModel 26.08 training.jsonl mapped to the contract, 20 steps).
  runMetrics: (id, q) => {
    if (id !== s3aRun.run_id) return Promise.reject(new Error(`run ${id} not found`));
    const { points } = parseJsonl(s3aMetrics);
    const from = Number(q.after ?? 0);
    const page = points.slice(from, from + q.limit);
    const next = from + page.length;
    return delay({
      points: page,
      next_cursor: String(next),
      has_more: next < points.length,
      downsampled: false,
    });
  },
  // FakeWorker: walks the graph in topological order and emits node_status events like the real executor.
  startRun: (workflow, trial) => {
    const id = `mock-${Date.now()}`;
    fakeRuns.set(id, structuredClone(workflow));
    if (trial) {
      // the server records the trial on the user model once it finishes; the mock does it when it starts
      for (const e of workflow.edges.filter((x) => x.to[0] === trial.node)) {
        const src = workflow.nodes.find((n) => n.id === e.from[0]);
        const rec = mockUserModels.get(String(src?.params.user_model ?? ""));
        if (rec) mockUserModels.set(rec.id, { ...rec, trial_run_id: id });
      }
    }
    return delay({ run_id: id });
  },
  cancelRun: (id) => {
    fakeCancel.get(id)?.();
    return delay(undefined);
  },
  // D-7 settings: the mock keeps only what the server would return (never the value itself).
  settings: () => delay<SettingsView>(structuredClone(mockSettings)),
  putSecret: (field, value) => {
    const state: SecretState = {
      set: true,
      updated_at: Date.now() / 1000,
      last4: value.slice(-4),
      readable: true,
      scope: "unchecked",
    };
    mockSettings.secrets[field] = state;
    return delay(state);
  },
  deleteSecret: (field) => {
    mockSettings.secrets[field] = {
      set: false,
      updated_at: null,
      last4: "",
      readable: true,
      scope: "unchecked",
    };
    return delay({ set: false as const });
  },
  // B-7: the server keeps only a re-encoded PNG and a random id; the mock mirrors the response shape.
  upload: (file) =>
    delay<UploadInfo>({
      upload_id: Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
        b.toString(16).padStart(2, "0"),
      ).join(""),
      label: file.name.slice(0, 120),
      width: 0,
      height: 0,
      size_bytes: file.size,
    }),
  nodeResult: () =>
    delay<TrialResult>({
      engine_used: "hf",
      text: '{"store": "minh hoạ", "total": "—"}',
      n_tokens: 12,
      truncated: false,
      decoding: { temperature: 0, top_p: 1 },
      overrides: {},
    }),
  // Real sample generated by bao-cleanup from app/storage.py (P013, plan v3): all 6 item kinds.
  bugReport: (runId, q) =>
    delay<BugReport>({
      ...(q.code ? { code: q.code } : {}),
      app_version: "1.0.0-dev",
      gpu: "NVIDIA GB10 (DGX Spark)",
      ...(runId ? { ref_code: "a1b2c3d4e5f6" } : {}), // the server derives it; the mock only pretends
    }),
  // D-8: the local machine, LAN sign-in locked for a few minutes after wrong tokens
  health: () =>
    delay<HealthInfo>({
      ok: true,
      version: "1.0.0.dev0",
      commit: "4b04214c0ffee0123456789abcdef0123456789a",
      dirty: false,
      app_version: "1.0.0.dev0+4b04214c0ffe",
    }),
  // A5: LAN on, one address, a made-up CA fingerprint (32 pairs, as browsers print it)
  lan: () =>
    delay<LanInfo>({
      enabled: true,
      urls: ["https://192.168.2.178:4000/"],
      ca: {
        fingerprint_sha256: Array.from({ length: 32 }, (_, i) =>
          ((i * 37 + 11) % 256).toString(16).toUpperCase().padStart(2, "0"),
        ).join(":"),
        download: "/api/lan/ca.crt",
      },
    }),
  session: () => delay<SessionInfo>({ ...mockSession }),
  signOut: () => delay(undefined),
  revokeAllSessions: () => delay(undefined),
  unlockLan: () => {
    mockSession.lan_login = { locked_until: null, recent_failures: 0 };
    return delay(mockSession.lan_login);
  },
  // G-5: a fake download, events in the server's order (progress… → verifying → completed)
  // ADR-021 / P021 §8 (luồng H): plan-only hub jobs, polled. The repo picks the outcome so every screen is reachable:
  // "remote" → MODEL-004, "nope" → NET-005, "huge" → NET-004 over the size cap, "busy" → 409 SYS-002, "script" → DATA-009,
  // "license" → unknown licence; the first poll is queued. A malformed id → 400 CFG-005, as the server says it.
  addUserModel: ({ repo, revision }) => {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || repo.includes("--"))
      return Promise.reject(
        new ApiError(400, {
          detail: { code: "CFG-005", params: { param: "repo", rule: "format" } },
        }),
      );
    if (repo.includes("busy"))
      return Promise.reject(
        new ApiError(409, { detail: { code: "SYS-002", params: { minutes: 0 } } }),
      );
    const id = `res-${++mockJobSeq}`;
    mockJobs.set(id, { polls: 0, repo, revision });
    return delay({ resolve_id: id });
  },
  userModelResolve: (id) => {
    const job = mockJobs.get(id);
    if (!job) return Promise.reject(new ApiError(404, { detail: { code: "SYS-000", params: {} } }));
    job.polls++;
    if (job.polls === 1) return delay({ status: "queued" as const });
    const fail = (code: string) =>
      delay({ status: "failed" as const, error: { code, params: { repo: job.repo } } });
    if (job.repo.includes("remote")) return fail("MODEL-004");
    if (job.repo.includes("nope")) return fail("NET-005");
    if (job.repo.includes("huge"))
      return delay({
        status: "failed" as const,
        error: {
          code: "NET-004",
          params: { size_gb: 72, quota_gb: mockLimits.max_model_gb, reason: "model_cap" },
        },
      });
    const [org = "", name = ""] = job.repo.split("/");
    const model: UserModel = {
      id: `user--${org}--${name}--0f3a9c1e`,
      name,
      repo: job.repo,
      revision: "0f3a9c1e2b7d4a6f8e9c0b1a2d3e4f5a6b7c8d9e",
      family: "Qwen2.5-VL",
      architecture: "Qwen2_5_VLForConditionalGeneration",
      params_b: 3.8,
      size_gb: 7.6,
      license: job.repo.includes("license") ? "unknown" : "apache-2.0",
      license_card: job.repo.includes("license") ? "Custom research licence, ask the author" : null,
      // from the server's obligations table; an id outside it is "unknown" (fail closed)
      license_class: job.repo.includes("license") ? "unknown" : "commercial_ok",
      obligations: [],
      license_confirmed: false,
      gated: job.repo.includes("gated"),
      ram_gb_train: 31.5,
      ram_source: "estimated",
      verified: false,
    };
    mockUserModels.set(model.id, model);
    return delay({ status: "done" as const, model });
  },
  confirmUserModelLicense: (id) => {
    const model = mockUserModels.get(id);
    if (!model)
      return Promise.reject(new ApiError(404, { detail: { code: "SYS-000", params: {} } }));
    const confirmed = { ...model, license_confirmed: true };
    mockUserModels.set(id, confirmed);
    return delay({ model: confirmed });
  },
  listUserModels: () => delay({ models: [...mockUserModels.values()] }),
  listUserDatasets: () => delay({ datasets: [...mockUserDatasets.values()] }),
  addUserDataset: ({ inspect_id, columns }) => {
    const job = mockJobs.get(inspect_id);
    if (!job) return Promise.reject(new ApiError(404, { detail: { code: "CFG-004", params: {} } }));
    if (!columns.image || (!columns.answer && !columns.messages))
      return Promise.reject(
        new ApiError(400, {
          detail: {
            code: "DATA-010",
            params: { repo: job.repo, missing: columns.image ? "answer" : "image", available: "" },
          },
        }),
      );
    const [org = "", name = ""] = job.repo.toLowerCase().split("/");
    const split = job.split ?? "train";
    const config = job.config ?? "default";
    const dataset: UserDataset = {
      id: `user--ds--${org}--${name}--7f0115a4--${config}--${split}`,
      name: `${name} (${config}/${split})`,
      repo: job.repo,
      revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
      config,
      split,
      size_gb: 0.05,
      license: "cc-by-4.0",
      license_card: null,
      license_class: "commercial_ok",
      obligations: [],
      license_confirmed: false,
      verified: false,
    };
    mockUserDatasets.set(dataset.id, dataset);
    return delay({ dataset });
  },
  confirmUserDatasetLicense: (id) => {
    const ds = mockUserDatasets.get(id);
    if (!ds) return Promise.reject(new ApiError(404, { detail: { code: "CFG-004", params: {} } }));
    const confirmed = { ...ds, license_confirmed: true };
    mockUserDatasets.set(id, confirmed);
    return delay({ dataset: confirmed });
  },
  userAssetLimits: () => delay({ ...mockLimits }),
  setUserAssetLimits: ({ max_model_gb }) => {
    const [lo, hi] = mockLimits.max_model_gb_range;
    if (!Number.isFinite(max_model_gb) || max_model_gb < lo || max_model_gb > hi)
      return Promise.reject(new ApiError(422, { detail: [{ loc: ["body", "max_model_gb"] }] }));
    mockLimits.max_model_gb = max_model_gb;
    return delay({ ...mockLimits });
  },
  inspectDataset: ({ repo, config, split }) => {
    const id = `ins-${++mockJobSeq}`;
    mockJobs.set(id, { polls: 0, repo, config, split });
    return delay({ inspect_id: id });
  },
  datasetInspection: (id) => {
    const job = mockJobs.get(id);
    if (!job) return Promise.reject(new ApiError(404, { detail: { code: "SYS-000", params: {} } }));
    job.polls++;
    if (job.polls === 1)
      return delay({ status: "running" as const, waiting_for: "download_slot" as const });
    if (job.repo.includes("script"))
      return delay({
        status: "failed" as const,
        error: { code: "DATA-009", params: { repo: job.repo, has_parquet_export: false } },
      });
    const result: DatasetInspection = {
      repo: job.repo,
      revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
      configs: ["default"],
      splits: ["train", "validation", "test"],
      config: job.config ?? "default",
      split: job.split ?? "train",
      format: job.repo.includes("wds") ? "webdataset" : "parquet",
      columns: [
        { name: "image", kind: "image" },
        { name: "query", kind: "text" },
        { name: "ground_truth", kind: "text" },
        { name: "id", kind: "other" },
      ],
      suggested: { image: "image", question: "query", answer: "ground_truth" },
      samples: [
        { question: "Tổng tiền trên hoá đơn là bao nhiêu?", answer: '{"total": "125.000"}' },
        { question: "Cửa hàng tên gì?", answer: '{"store": "Bánh mì Hoà"}' },
        { error: "image_unreadable" },
      ],
      rows_estimate: null,
      split_bytes: 48_000_000,
      has_script: false,
      sample_error: null,
      default_question: "Describe this image.",
    };
    return delay({ status: "done" as const, result });
  },
  startDownload: (body) =>
    delay<DownloadStart>({
      download_id: `dl-${body.kind}-${body.catalog_id}`,
      status: "queued",
      name: body.catalog_id,
      size_gb: 1.6,
    }),
  watchDownload: (_id, onEvent) => {
    const total = 1_600_000_000;
    const timers = [1, 2, 3].map((i) =>
      setTimeout(
        () =>
          onEvent({
            type: "progress",
            files_done: i,
            files_total: 3,
            bytes_done: (total * i) / 3,
            bytes_total: total,
          }),
        i * FAKE_STEP_MS,
      ),
    );
    timers.push(
      setTimeout(() => onEvent({ type: "status", status: "verifying" }), 4 * FAKE_STEP_MS),
      setTimeout(() => onEvent({ type: "status", status: "completed" }), 5 * FAKE_STEP_MS),
    );
    return () => {
      for (const x of timers) clearTimeout(x);
    };
  },
  // P016 §3: 8 real keys of configs/engine/automodel-26.08-6.json; filtered by template like the server
  advancedOptions: (_typeId, model) =>
    delay({
      options: (advancedFixture.options as AdvancedOption[]).filter(
        (o) => !model || o.templates.includes(model),
      ),
    }),
  // G-4 (storage_reset.py shape): one finished attempt with a checkpoint; typed confirmation with the node title.
  nodeResetPlan: (node) => {
    const nid = node.split("-")[0] ?? node;
    return delay<NodeResetPlan>({
      schema_version: 1,
      plan_id: `reset-${node}`,
      expires_at: Date.now() / 1000 + 600,
      node,
      node_title: mockNodeTitle(nid),
      attempts: [
        { run_dir: `${node}-a1`, status: "completed", size_gb: 4.2, has_checkpoint: true },
      ],
      total_gb: 4.2,
      blocked_by: null,
      blocked_params: {},
      consequence_code: "PERMANENT_LOSS",
      consequence_params: { name: mockNodeTitle(nid) },
      confirm_required: true,
      confirm_text: mockNodeTitle(nid),
      next_generation: 2,
    });
  },
  nodeReset: (body) =>
    delay<NodeResetResult>({
      deleted: [`${body.node}-a1`],
      failed: [],
      freed_gb: 4.2,
      generation: 2,
    }),
  storagePlan: () =>
    delay({
      ...(storagePlanSample as unknown as StoragePlan),
      expires_at: Date.now() / 1000 + 600,
    }),
  storageApply: (body) => {
    const items = (storagePlanSample as unknown as StoragePlan).items;
    const skipped: StorageApplyResult["skipped"] = [];
    const done: StorageItem[] = [];
    for (const id of body.item_ids) {
      const item = items.find((i) => i.item_id === id);
      if (!item || item.section === "kept") skipped.push({ item_id: id, reason: "NOT_APPLICABLE" });
      else if (item.confirm_required && body.confirm?.[id] !== item.confirm_text)
        skipped.push({ item_id: id, reason: "CONFIRM_REQUIRED" });
      else done.push(item);
    }
    return delay<StorageApplyResult>({
      evicted: done.filter((i) => i.source === "node_cache"),
      pruned: done.filter((i) => i.source !== "node_cache"),
      skipped,
      freed_gb: done.reduce((sum, i) => sum + (i.size_gb ?? 0), 0),
    });
  },
  watchRun: (id, onEvent) => {
    const wf = fakeRuns.get(id);
    const timers: ReturnType<typeof setTimeout>[] = [];
    if (!wf) return () => {};
    const order = [...stepNumbers(wf).entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([nodeId]) => nodeId);
    let t = 200;
    for (const nodeId of order) {
      const heavy =
        registry.node_types.find((s) => s.type_id === wf.nodes.find((n) => n.id === nodeId)?.type)
          ?.runner === "container";
      const steps = heavy ? 5 : 1;
      for (let i = 0; i < steps; i++) {
        const progress = (i + 1) / steps;
        timers.push(
          setTimeout(
            () =>
              onEvent({
                type: "node_status",
                node_id: nodeId,
                status: "running",
                progress: i / steps,
                node_base: mockNodeBase(nodeId),
              }),
            t,
          ),
        );
        t += heavy ? FAKE_STEP_MS : FAKE_STEP_MS / 2;
        if (progress === 1)
          timers.push(
            setTimeout(
              () =>
                onEvent({
                  type: "node_status",
                  node_id: nodeId,
                  status: "completed",
                  progress: 1,
                  node_base: mockNodeBase(nodeId),
                }),
              t,
            ),
          );
      }
    }
    timers.push(setTimeout(() => onEvent({ type: "run_status", status: "completed" }), t + 50));
    fakeCancel.set(id, () => {
      for (const x of timers) clearTimeout(x);
      onEvent({ type: "run_status", status: "cancelled" });
    });
    return () => {
      for (const x of timers) clearTimeout(x);
    };
  },
};

const fakeRuns = new Map<string, Workflow>();

// Nodes of the storage-plan fixture (node_base n4-… / n5-…) when no fake run has happened yet.
const FIXTURE_NODE_TYPES: Record<string, string> = { n4: "train.lora_vlm", n5: "eval.benchmarks" };
// G-4: the node base the executor names run dirs with ("<nid>-<key16>"); the mock uses a fixed key.
const mockNodeBase = (nodeId: string) => `${nodeId}-0123456789abcdef`;
/** Title of a node in the most recent fake run (the reset dialog types it to confirm). */
function mockNodeTitle(nodeId: string): string {
  const runs = [...fakeRuns.values()];
  const type =
    runs[runs.length - 1]?.nodes.find((n) => n.id === nodeId)?.type ?? FIXTURE_NODE_TYPES[nodeId];
  return type ? tp(["node", type, "title"]) : nodeId;
}
/** Real registry (REGISTRY.to_api() snapshot from schemas/); the hand-written mockNodeTypes are for tests only. */
const registry = registrySnapshot as unknown as NodeTypesResponse;

const fakeCancel = new Map<string, () => void>();
/** Duration of one fake progress step; tests shorten it via setFakeStepMs. */
let FAKE_STEP_MS = 400;
export function setFakeStepMs(ms: number): void {
  FAKE_STEP_MS = ms;
}
