// In-browser mock of the P009 API for development (VITE_MOCK_API=1) and tests. It mirrors the documented
// shapes only; the real validation lives in the backend (`POST /api/graphs/validate`) and is the source of truth.
import { tp } from "../i18n";
import { isCompatible } from "../lib/ports";
import type { Api } from "./client";
import type {
  NodeSpec,
  NodeTypesResponse,
  ParamsSchema,
  PortType,
  Template,
  ValidationIssue,
  ValidationResult,
  Workflow,
} from "./types";

const portTypes: PortType[] = [
  { id: "Dataset", parent: null, color_token: "color.port.dataset", label_key: "port.dataset" },
  { id: "Model", parent: null, color_token: "color.port.model", label_key: "port.model" },
  { id: "Adapter", parent: null, color_token: "color.port.model", label_key: "port.adapter" },
  { id: "Checkpoint", parent: "Model", color_token: "color.port.model", label_key: "port.model" },
  { id: "QuantizedModel", parent: "Model", color_token: "color.port.model", label_key: "port.quantized_model" },
  { id: "ExportArtifact", parent: null, color_token: "color.port.model", label_key: "port.export_artifact" },
  { id: "EvalReport", parent: null, color_token: "color.port.report", label_key: "port.eval_report" },
  { id: "BenchReport", parent: null, color_token: "color.port.report", label_key: "port.bench_report" },
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

export const mockNodeTypes: NodeSpec[] = [
  spec({
    type_id: "model.hf_pick",
    category: "source",
    runner: "host",
    outputs: { model: { type: "Model" } },
    params_schema: {
      type: "object",
      properties: {
        repo: { type: "string", enum: ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"], default: "Qwen/Qwen3.5-0.8B" },
      },
      required: ["repo"],
    },
  }),
  spec({
    type_id: "data.local_folder",
    category: "source",
    runner: "host",
    outputs: { dataset: { type: "Dataset" } },
    params_schema: {
      type: "object",
      properties: { path: { type: "string", default: "data/sample-invoices" } },
      required: ["path"],
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
        val_fraction: { type: "number", minimum: 0.05, maximum: 0.5, default: 0.1, "x-ui": { widget: "slider" } },
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
          "x-ui": { widget: "segmented", order: 0, preset: true },
        },
        epochs: { type: "integer", minimum: 1, maximum: 10, default: 2, "x-ui": { order: 1 } },
        learning_rate: { type: "number", default: 0.0001, "x-ui": { advanced: true, order: 2 } },
      },
    },
  }),
  spec({
    type_id: "export.merge_lora",
    category: "export",
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
          "x-ui": { widget: "segmented", preset: true },
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
        groups: { type: "array", enum: ["documents", "charts", "reasoning", "math"], default: ["documents"] },
      },
    },
  }),
  // Third-party example used to exercise badges/permissions (F6d). Never shipped.
  spec({
    type_id: "acme.ocr_clean",
    category: "data",
    inputs: { dataset: { type: "Dataset" } },
    outputs: { dataset: { type: "Dataset" } },
    origin: { kind: "third_party", publisher: "acme", pack: "vision-tools", version: "0.3.1", reviewed: false },
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

const templates: Template[] = [
  {
    id: "lora_finetune",
    title_key: "template.lora_finetune.title",
    description_key: "template.lora_finetune.desc",
    difficulty: "basic",
    tags: ["finetune"],
    sample_data: true,
    estimate: { ram_gb: { lo: 18, hi: 26 }, disk_gb: { lo: 4, hi: 8 }, minutes: { lo: 20, hi: 45 }, gpu_exclusive: true, source: "formula" },
    workflow: {
      format: "vlm-workflow",
      schema_version: 1,
      name: "lora_finetune",
      nodes: [
        { id: "n1", type: "model.hf_pick", version: "1.0.0", params: { repo: "Qwen/Qwen3.5-0.8B" } },
        { id: "n2", type: "data.local_folder", version: "1.0.0", params: { path: "data/sample-invoices" } },
        { id: "n3", type: "data.split", version: "1.0.0", params: { val_fraction: 0.1 } },
        { id: "n4", type: "train.lora_vlm", version: "1.0.0", params: { preset: "balanced", epochs: 2 } },
        { id: "n5", type: "export.merge_lora", version: "1.0.0", params: {} },
        { id: "n6", type: "eval.benchmarks", version: "1.0.0", params: { groups: ["documents"] } },
      ],
      edges: [
        { from: ["n2", "dataset"], to: ["n3", "dataset"] },
        { from: ["n1", "model"], to: ["n4", "base_model"] },
        { from: ["n3", "train"], to: ["n4", "train_data"] },
        { from: ["n3", "val"], to: ["n4", "val_data"] },
        { from: ["n1", "model"], to: ["n5", "base_model"] },
        { from: ["n4", "adapter"], to: ["n5", "adapter"] },
        { from: ["n5", "model"], to: ["n6", "model"] },
      ],
    },
  },
  {
    id: "quantize_bench",
    title_key: "template.quantize_bench.title",
    description_key: "template.quantize_bench.desc",
    difficulty: "basic",
    tags: ["quantize", "eval"],
    sample_data: false,
    estimate: { ram_gb: { lo: 12, hi: 20 }, disk_gb: { lo: 2, hi: 4 }, minutes: { lo: 15, hi: 40 }, gpu_exclusive: true, source: "formula" },
    workflow: {
      format: "vlm-workflow",
      schema_version: 1,
      name: "quantize_bench",
      nodes: [
        { id: "n1", type: "model.hf_pick", version: "1.0.0", params: { repo: "Qwen/Qwen3.5-2B" } },
        { id: "n2", type: "quantize.modelopt_ptq", version: "1.0.0", params: { format: "fp8" } },
        { id: "n3", type: "eval.benchmarks", version: "1.0.0", params: { groups: ["documents", "charts"] } },
      ],
      edges: [
        { from: ["n1", "model"], to: ["n2", "model"] },
        { from: ["n2", "quantized"], to: ["n3", "model"] },
      ],
    },
  },
];

function issue(code: string, partial: Partial<ValidationIssue>, params: Record<string, string | number>): ValidationIssue {
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
export function mockValidate(workflow: Workflow, types: NodeSpec[], ports: PortType[]): ValidationResult {
  const byType = new Map(types.map((s) => [s.type_id, s]));
  const nodeTitle = (typeId: string) => tp(["node", typeId, "title"]);
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
        errors.push(issue("CFG-003", { node_id: node.id, port }, { node: nodeTitle(node.type), port }));
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
  const heavy = workflow.nodes.some((n) => byType.get(n.type)?.runner === "container");
  return {
    ok: !errors.some((e) => e.severity === "error"),
    errors,
    estimate: {
      ram_gb_peak: heavy ? { lo: 18, hi: 26 } : null,
      disk_gb: heavy ? { lo: 4, hi: 8 } : null,
      duration_min: heavy ? { lo: 20, hi: 45 } : null,
      gpu_exclusive: heavy,
      source: heavy ? "formula" : "unknown",
    },
  };
}

const delay = <T,>(value: T) => new Promise<T>((r) => setTimeout(() => r(structuredClone(value)), 50));

export const mockApi: Api = {
  nodeTypes: () => delay<NodeTypesResponse>({ node_types: mockNodeTypes, port_types: portTypes }),
  templates: () => delay(templates.map(({ workflow: _w, ...meta }) => meta)),
  template: (id) => {
    const found = templates.find((x) => x.id === id);
    return found ? delay(found) : Promise.reject(new Error(`template ${id} not found`));
  },
  validate: (workflow) => delay(mockValidate(workflow, mockNodeTypes, portTypes)),
  system: () =>
    delay({
      mem_total_gb: 121,
      mem_available_gb: 42,
      gpu_lock: { held: false },
      disk_free_gb: 2000,
    }),
};
