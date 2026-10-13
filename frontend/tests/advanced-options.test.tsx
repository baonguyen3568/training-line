// "Tuỳ chọn engine" (P016 §3, design 5ae3674): only changed keys are stored; a changed row is bold with a reset
// and an accessible note; a key's CFG-005 shows under its row; search says when nothing matches.
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ValidationIssue } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

async function setup(issues: ValidationIssue[] = []) {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
  const { AdvancedOptions } = await import("../src/components/AdvancedOptions");
  const view = render(<AdvancedOptions nodeId={id} typeId="train.sft_vlm" issues={issues} />);
  await screen.findByRole("searchbox");
  const adv = () =>
    useWorkflow.getState().workflow?.nodes[0]?.params.advanced as
      | Record<string, unknown>
      | undefined;
  return { id, adv, ...view };
}

describe("advanced engine options", () => {
  it("stores only changed keys and resets a key by removing it", async () => {
    const { adv, container } = await setup();
    expect(screen.getByText(/Chưa nối bước Chọn model gốc/)).toBeTruthy(); // no model wired yet
    const input = container.querySelector<HTMLInputElement>('[id="adv-dataloader.num_workers"]');
    if (!input) throw new Error("no num_workers");
    fireEvent.change(input, { target: { value: "4" } });
    expect(adv()).toEqual({ "dataloader.num_workers": 4 });
    const row = input.closest("li");
    expect(row?.classList.contains("changed")).toBe(true);
    expect(row?.querySelector("label")?.getAttribute("aria-description")).toBeTruthy();
    expect(row?.closest("details")?.open).toBe(true);
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Đặt về mặc định" }));
    });
    expect(adv()).toEqual({});
  });

  it("shows a key's CFG-005 under its own row, and says when a search matches nothing", async () => {
    const { container } = await setup([
      {
        node_id: "x",
        param: "advanced",
        code: "CFG-005",
        severity: "error",
        params: { key: "model.backend.attn", rule: "choice" },
        title: "Tham số không hợp lệ",
        what: "Giá trị này không có trong danh sách.",
        action_ids: [],
      },
    ]);
    const row = container.querySelector('[id="adv-model.backend.attn"]')?.closest("li");
    expect(row?.querySelector(".adv-error")?.textContent).toContain(
      "Giá trị này không có trong danh sách.",
    );
    expect(row?.closest("details")?.open).toBe(true);
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "khongcokhoanay" } });
    expect(screen.getByText('Không có tuỳ chọn nào khớp "khongcokhoanay".')).toBeTruthy();
  });
});
