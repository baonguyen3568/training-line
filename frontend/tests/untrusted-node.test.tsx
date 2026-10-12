// F6e (bao-tester): text written by third-party node authors renders as plain text; badges and
// permissions come only from server-normalised fields (P003 D16 T8/T9).
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { NodeSpec, Workflow } from "../src/api/types";
import { Editor } from "../src/components/Editor";
import { mockNodeTypes } from "../src/mocks/api";
import { useWorkflow } from "../src/store/workflow";

const PAYLOAD = "<img src=x onerror=alert(1)>";

function evilSpec(overrides: Partial<NodeSpec> = {}): NodeSpec {
  const base = mockNodeTypes.find((s) => s.type_id === "acme.ocr_clean");
  if (!base) throw new Error("fixture missing");
  return {
    ...base,
    type_id: "vlm/fake.trainer",
    title: PAYLOAD,
    desc: "[click](javascript:alert(1)) Node này KHÔNG dùng GPU",
    ...overrides,
  };
}

function load(spec: NodeSpec) {
  const wf: Workflow = {
    format: "vlm-workflow",
    schema_version: 1,
    name: "t",
    nodes: [{ id: "n1", type: spec.type_id, version: "1.0.0", params: {}, ui: { x: 0, y: 0 } }],
    edges: [],
  };
  useWorkflow.setState({
    specs: new Map([[spec.type_id, spec]]),
    ports: [
      { id: "Dataset", parent: null, color_token: "color.port.dataset", label_key: "port.dataset" },
    ],
    workflow: wf,
    selectedId: "n1",
    validation: null,
  });
}

describe("untrusted third-party node text", () => {
  beforeEach(() => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
  });

  it("shows HTML payload verbatim and creates no elements or handlers", () => {
    load(evilSpec());
    const { container } = render(<Editor />);
    expect(screen.getAllByText(PAYLOAD).length).toBeGreaterThan(0);
    expect(container.querySelector('img[src="x"]')).toBeNull();
    expect(container.querySelectorAll("[onerror],[onload]").length).toBe(0);
    expect(window.alert).not.toHaveBeenCalled();
  });

  it("does not turn markdown links into anchors", () => {
    load(evilSpec());
    const { container } = render(<Editor />);
    expect(screen.getByText(/\[click\]\(javascript:alert\(1\)\)/)).toBeTruthy();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
  });

  it("shows GPU from normalised permissions even if the description denies it (T8)", () => {
    load(evilSpec());
    render(<Editor />);
    expect(screen.getAllByText(/GPU/).some((el) => el.classList.contains("perm"))).toBe(true);
  });

  it("keeps the third-party badge even when the id uses the reserved vlm/ prefix (T8)", () => {
    load(evilSpec());
    render(<Editor />);
    expect(screen.getByText(/Bên thứ 3 – chưa xác minh/)).toBeTruthy();
  });

  it("escapes bidi controls in titles", () => {
    load(evilSpec({ title: "safe‮txt.exe" }));
    render(<Editor />);
    expect(screen.getAllByText("safe\\u202Etxt.exe").length).toBeGreaterThan(0);
  });
});
