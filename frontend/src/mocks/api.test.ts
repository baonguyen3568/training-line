import { describe, expect, it } from "vitest";
import type { Workflow } from "../api/types";
import { mockApi, mockNodeTypes, mockValidate } from "./api";

describe("mock API", () => {
  it("templates validate clean", async () => {
    const tpl = await mockApi.template("lora_finetune");
    const res = await mockApi.validate(tpl.workflow, "vi");
    expect(res.errors).toEqual([]);
    expect(res.ok).toBe(true);
  });

  it("reports missing required input (CFG-003) and unknown node type (CFG-004)", async () => {
    const { port_types } = await mockApi.nodeTypes("vi");
    const wf: Workflow = {
      format: "vlm-workflow",
      schema_version: 1,
      name: "x",
      nodes: [
        { id: "a", type: "train.lora_vlm", version: "1.0.0", params: {} },
        { id: "b", type: "nope.unknown", version: "1.0.0", params: {} },
      ],
      edges: [],
    };
    const codes = mockValidate(wf, mockNodeTypes, port_types).errors.map((e) => e.code);
    expect(codes).toContain("CFG-003");
    expect(codes).toContain("CFG-004");
  });
});
