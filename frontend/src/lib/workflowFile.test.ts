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

  it("keeps P016 group definitions through a round-trip, and rejects a malformed groups key", () => {
    const group = {
      id: "user.prep",
      version: "1.0.0",
      title: "Chuẩn bị",
      nodes: [{ id: "a", type: "data.split", version: "1.0.0", params: {} }],
      edges: [],
      inputs: { data: [["a", "dataset"]] },
      outputs: { train: ["a", "train"] },
      params: {},
    };
    const withGroups = { ...wf, groups: [group] };
    const res = parseWorkflowFile(JSON.stringify(withGroups));
    expect(res.ok).toBe(true);
    if (res.ok) expect(JSON.parse(serializeWorkflow(res.workflow))).toEqual(withGroups);
    const { edges: _edges, ...noEdges } = group;
    expect(parseWorkflowFile(JSON.stringify({ ...wf, groups: [noEdges] })).ok).toBe(true); // optional
    const rejected: [unknown, string][] = [
      [{}, "groups"],
      [Array(51).fill(group), "groups"],
      [[{ ...group, id: "group.official" }], "groups.id"], // first-party namespace never comes from a file
      [[{ ...group, id: "user.a/b" }], "groups.id"],
      [[{ ...group, title_key: "template.invoice_to_json.title" }], "groups.title_key"],
      [[{ ...group, install_url: "https://evil.example" }], "groups.install_url"],
      [
        [{ ...group, nodes: [{ ...group.nodes[0], install_url: "x" }] }],
        "groups.nodes.install_url",
      ],
      [[{ ...group, inputs: { data: "a" } }], "groups.inputs"],
      [
        [{ ...group, edges: Array(2001).fill({ from: ["a", "x"], to: ["b", "y"] }) }],
        "groups.edges",
      ],
      [
        [
          {
            ...group,
            params: Object.fromEntries(
              Array.from({ length: 101 }, (_, i) => [`p${i}`, ["a", "x"]]),
            ),
          },
        ],
        "groups.params",
      ],
      [
        [
          {
            ...group,
            inputs: Object.fromEntries(
              Array.from({ length: 8 }, (_, i) => [`i${i}`, Array(64).fill(["a", "x"])]),
            ),
          },
        ],
        "groups.inputs",
      ],
    ];
    for (const [groups, field] of rejected)
      expect(parseWorkflowFile(JSON.stringify({ ...wf, groups }))).toEqual({ ok: false, field });
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
