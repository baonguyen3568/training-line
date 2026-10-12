import { describe, expect, it } from "vitest";
import { emptyWorkflow } from "../store/workflow";
import { parseWorkflowFile, serializeWorkflow } from "./workflowFile";

const wf = {
  format: "vlm-workflow",
  schema_version: 1,
  name: "x",
  nodes: [
    {
      id: "n1",
      type: "model.hf_pick",
      version: "1.0.0",
      params: { repo: "a/b" },
      ui: { x: 1, y: 2 },
    },
  ],
  edges: [],
};

describe("workflow file (untrusted)", () => {
  it("round-trips", () => {
    const res = parseWorkflowFile(JSON.stringify(wf));
    expect(res.ok).toBe(true);
    if (res.ok) expect(JSON.parse(serializeWorkflow(res.workflow))).toEqual(wf);
  });

  it("rejects install sources and other unknown fields (P003 D11)", () => {
    const bad = { ...wf, nodes: [{ ...wf.nodes[0], install_url: "https://evil.example/x.git" }] };
    expect(parseWorkflowFile(JSON.stringify(bad))).toEqual({
      ok: false,
      field: "nodes.install_url",
    });
    expect(parseWorkflowFile(JSON.stringify({ ...wf, source: "x" }))).toEqual({
      ok: false,
      field: "source",
    });
  });

  it("rejects malformed ids, duplicate ids, bad JSON and oversize files", () => {
    const dup = { ...wf, nodes: [wf.nodes[0], wf.nodes[0]] };
    expect(parseWorkflowFile(JSON.stringify(dup)).ok).toBe(false);
    expect(
      parseWorkflowFile(JSON.stringify({ ...wf, nodes: [{ ...wf.nodes[0], id: "<img>" }] })).ok,
    ).toBe(false);
    expect(parseWorkflowFile("{not json")).toEqual({ ok: false, field: "json" });
    expect(parseWorkflowFile(" ".repeat(3 * 1024 * 1024))).toEqual({ ok: false, field: "size" });
  });

  it("serializes an empty workflow", () => {
    expect(JSON.parse(serializeWorkflow(emptyWorkflow())).nodes).toEqual([]);
  });
});
