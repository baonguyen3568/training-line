import { expect, it } from "vitest";
import { stepNumbers } from "./topo";

it("numbers steps in dependency order", () => {
  const steps = stepNumbers({
    format: "vlm-workflow",
    schema_version: 1,
    name: "x",
    nodes: ["c", "a", "b"].map((id) => ({ id, type: "t", version: "1", params: {} })),
    edges: [
      { from: ["a", "o"], to: ["b", "i"] },
      { from: ["b", "o"], to: ["c", "i"] },
    ],
  });
  expect([steps.get("a"), steps.get("b"), steps.get("c")]).toEqual([1, 2, 3]);
});
