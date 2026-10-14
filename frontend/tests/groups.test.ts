// P016 groups in the UI (016b): server entries and user.* groups become one spec shape; a collapsed group's status
// folds from its inner steps' flat ids (P4-4 §1). Derivation follows the server's rules (golden fixtures later).
import { describe, expect, it } from "vitest";
import type { GroupDefinition, NodeSpec } from "../src/api/types";
import {
  deriveGroupSpec,
  foldGroupState,
  innerStepCount,
  specsFromCatalog,
  withUserGroups,
} from "../src/lib/groups";
import type { NodeRunState } from "../src/store/workflow";

const node = (type_id: string, over: Partial<NodeSpec> = {}): NodeSpec => ({
  type_id,
  version: "1.0.0",
  category: "train",
  icon: "x",
  inputs: {},
  outputs: {},
  params_schema: { type: "object", properties: {} },
  cacheable: true,
  runner: "host",
  origin: { kind: "first_party" },
  permissions: [],
  policy_block: null,
  ...over,
});

const catalog = new Map<string, NodeSpec>([
  [
    "data.prep",
    node("data.prep", {
      category: "data",
      inputs: { data: { type: "Dataset" } },
      outputs: { out: { type: "Dataset" } },
    }),
  ],
  [
    "train.lora",
    node("train.lora", {
      runner: "container",
      inputs: {
        base: { type: "Model" },
        train: { type: "Dataset" },
        val: { type: "Dataset", optional: true },
      },
      outputs: { model: { type: "Model" } },
      params_schema: {
        type: "object",
        properties: { epochs: { type: "integer", default: 3, "x-ui": { order: 1 } } },
      },
      permissions: [
        {
          id: "gpu",
          level: "warning",
          requires_typed_confirm: false,
          perm_copy_key: "p.gpu",
          confirm_copy_key: "p.gpu.confirm",
        } as NodeSpec["permissions"][number],
      ],
    }),
  ],
]);

const lora: GroupDefinition = {
  id: "user.lora",
  version: "1.0.0",
  title: "LoRA của tôi",
  nodes: [
    { id: "prep", type: "data.prep", version: "1.0.0", params: {} },
    { id: "fit", type: "train.lora", version: "1.0.0", params: { epochs: 5 } },
  ],
  edges: [{ from: ["prep", "out"], to: ["fit", "train"] }],
  inputs: { model: [["fit", "base"]], data: [["prep", "data"]], val: [["fit", "val"]] },
  outputs: { model: ["fit", "model"] },
  params: { epochs: ["fit", "epochs"] },
};

describe("group specs", () => {
  it("derives a user group like the server: exposed ports and params, lane of the first output, union, presets", () => {
    const spec = deriveGroupSpec(lora, catalog);
    expect(spec).toMatchObject({
      type_id: "user.lora",
      title: "LoRA của tôi",
      icon: "layers",
      category: "train", // the node behind the first exposed output
      runner: "container",
      inputs: {
        model: { type: "Model" },
        data: { type: "Dataset" },
        val: { type: "Dataset", optional: true },
      },
      outputs: { model: { type: "Model" } },
      group: { readOnly: true, user: true },
    });
    expect(spec?.inputs.model?.optional).toBeUndefined();
    expect(spec?.params_schema.properties.epochs).toMatchObject({
      default: 5,
      "x-ui": { order: 1 },
    });
    expect(spec?.permissions.map((p) => p.id)).toEqual(["gpu"]);
  });

  it("refuses what the server refuses: mixed types on one input, an unknown inner type", () => {
    const mixed = {
      ...lora,
      inputs: {
        x: [
          ["fit", "base"],
          ["prep", "data"],
        ] as [string, string][],
      },
    };
    expect(deriveGroupSpec(mixed, catalog)).toBeNull();
    const unknown = {
      ...lora,
      nodes: [...lora.nodes, { id: "z", type: "nope.x", version: "1.0.0", params: {} }],
    };
    expect(deriveGroupSpec(unknown, catalog)).toBeNull();
  });

  it("a user group may use another user group, in any order in the file", () => {
    const outer: GroupDefinition = {
      id: "user.outer",
      version: "1.0.0",
      title: "Ngoài",
      nodes: [{ id: "g", type: "user.lora", version: "1.0.0", params: {} }],
      edges: [],
      inputs: { model: [["g", "model"]], data: [["g", "data"]] },
      outputs: { model: ["g", "model"] },
      params: {},
    };
    const specs = withUserGroups(catalog, [outer, lora]);
    expect(specs.get("user.outer")?.inputs.model?.type).toBe("Model");
    expect(innerStepCount(specs.get("user.outer"), specs)).toBe(2); // the leaves, through the inner group
  });

  it("takes the server's groups[] and the alternative entry inside node_types alike", () => {
    const body = {
      nodes: lora.nodes,
      edges: lora.edges,
      inputs: lora.inputs,
      outputs: lora.outputs,
      params: lora.params,
    };
    const entry = { ...node("x"), group_id: "group.lora", body } as never;
    const alt = {
      ...node("group.alt"),
      group: { ...body, read_only: true },
    } as unknown as NodeSpec;
    const specs = specsFromCatalog([...catalog.values(), alt], [entry]);
    expect(specs.get("group.lora")?.group?.body.nodes).toHaveLength(2);
    expect(specs.get("group.lora")?.group?.readOnly).toBe(true);
    expect(specs.get("group.alt")?.group?.body.outputs.model).toEqual(["fit", "model"]);
    expect(specs.get("train.lora")?.group).toBeUndefined();
  });
});

describe("folded group status", () => {
  const run = (states: Record<string, NodeRunState["status"]>) =>
    Object.fromEntries(Object.entries(states).map(([k, status]) => [k, { status }]));

  it("failed wins, running shows x/y done, all reused is cached, any real run is completed", () => {
    expect(
      foldGroupState(run({ g__prep: "completed", g__fit: "failed" }), "g", 2)?.state.status,
    ).toBe("failed");
    const going = foldGroupState(run({ g__prep: "completed", g__fit: "running" }), "g", 2);
    expect(going?.state.status).toBe("running");
    expect(going?.doneCount).toBe(1);
    expect(foldGroupState(run({ g__prep: "cached", g__fit: "cached" }), "g", 2)?.state.status).toBe(
      "cached",
    );
    expect(
      foldGroupState(run({ g__prep: "cached", g__fit: "completed" }), "g", 2)?.state.status,
    ).toBe("completed");
    // other nodes and other groups never count; nothing reported yet = no state
    expect(foldGroupState(run({ h__prep: "failed", n1: "failed" }), "g", 2)).toBeUndefined();
  });

  it("an inner step waiting for memory makes the group wait, with its numbers", () => {
    const f = foldGroupState(
      {
        g__prep: { status: "completed" },
        g__fit: { status: "running", waiting: { state: "waiting_memory", need_gb: 40 } },
      },
      "g",
      2,
    );
    expect(f?.state.waiting?.need_gb).toBe(40);
  });
});
