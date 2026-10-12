import { afterEach, expect, it } from "vitest";
import { clearDraft, loadDraft, saveDraft } from "./draft";

afterEach(() => clearDraft());

it("round-trips a non-empty workflow and ignores empty or tampered drafts", () => {
  const wf = {
    format: "vlm-workflow" as const,
    schema_version: 1 as const,
    name: "x",
    nodes: [{ id: "n1", type: "model.hf_pick", version: "1.0.0", params: {}, ui: { x: 0, y: 0 } }],
    edges: [],
  };
  saveDraft(wf);
  expect(loadDraft()?.workflow.nodes).toHaveLength(1);
  saveDraft({ ...wf, nodes: [] });
  expect(loadDraft()).toBeNull();
  localStorage.setItem(
    "vlm.draft.v1",
    JSON.stringify({
      savedAt: 1,
      workflow: '{"format":"vlm-workflow","schema_version":1,"nodes":[{"id":"<x>"}],"edges":[]}',
    }),
  );
  expect(loadDraft()).toBeNull();
});
