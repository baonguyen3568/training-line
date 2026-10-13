// A step that fails while running gets an error card in the inspector (title, what, quick fixes, report), not
// only a red "failed" state. Text comes rendered from the server; an older server's bare code still gets a title.
import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { EventError } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

async function failed(error: EventError) {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
  useWorkflow.getState().select(id);
  const { Inspector } = await import("../src/components/Inspector");
  const view = render(<Inspector />);
  act(() =>
    useWorkflow.setState({
      run: {
        id: "r1",
        done: true,
        startedAt: 0,
        nodes: { [id]: { status: "failed", errorCode: error.code, error } },
      },
    }),
  );
  return view;
}

describe("runtime error card", () => {
  it("shows the server's text and its quick fixes, including the bug report", async () => {
    const { container } = await failed({
      code: "MEM-003",
      params: {},
      title: "Máy hết bộ nhớ khi đang train",
      what: "Bước này dùng nhiều bộ nhớ hơn máy còn trống.",
      actions: [
        {
          id: "send_report",
          kind: "navigate",
          confirm: false,
          target: "bug_report",
          node_id: "n1",
        },
      ],
      action_ids: ["send_report"],
    });
    const card = container.querySelector('[data-code="MEM-003"]');
    expect(card?.textContent).toContain("Máy hết bộ nhớ khi đang train");
    expect(card?.textContent).toContain("Bước này dùng nhiều bộ nhớ hơn");
    expect(screen.getByRole("button", { name: /Gửi báo cáo lỗi/ })).toBeTruthy();
  });

  it("an older server's bare code still gets its catalog title, never the raw code alone", async () => {
    const { container } = await failed({ code: "MEM-003", params: {} });
    const card = container.querySelector('[data-code="MEM-003"]');
    expect(card?.querySelector("strong")?.textContent).not.toBe("MEM-003");
    expect(card?.querySelector("strong")?.textContent?.length).toBeGreaterThan(5);
  });
});
