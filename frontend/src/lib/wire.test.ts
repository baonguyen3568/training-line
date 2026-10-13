import { describe, expect, it } from "vitest";
import { connectionBetween, handleUnder } from "./wire";

function handle(kind: "source" | "target", nodeId: string, id: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `react-flow__handle ${kind} port`;
  el.setAttribute("data-nodeid", nodeId);
  el.setAttribute("data-handleid", id);
  const inner = document.createElement("span");
  el.append(inner);
  return inner; // elementFromPoint may return a child of the handle
}

describe("wire drop fallback", () => {
  it("reads the handle under the pointer from the DOM", () => {
    expect(handleUnder(handle("target", "n3", "base_model"))).toEqual({
      nodeId: "n3",
      id: "base_model",
      type: "target",
    });
    expect(handleUnder(document.createElement("div"))).toBeNull();
    expect(handleUnder(null)).toBeNull();
  });

  it("orients the connection source → target whichever end the drag started from", () => {
    const out = { nodeId: "n1", id: "model", type: "source" };
    const inp = { nodeId: "n3", id: "base_model", type: "target" };
    const want = { source: "n1", sourceHandle: "model", target: "n3", targetHandle: "base_model" };
    expect(connectionBetween(out, inp)).toEqual(want);
    expect(connectionBetween(inp, out)).toEqual(want);
    expect(connectionBetween(out, { ...out, nodeId: "n2" })).toBeNull(); // two outputs
    expect(connectionBetween(out, { ...inp, nodeId: "n1" })).toBeNull(); // same node
  });
});
