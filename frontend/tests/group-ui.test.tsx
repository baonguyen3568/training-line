// P4-4 groups on the canvas (mock serves cleancode's golden groups): a collapsed group shows its mark, folded progress
// and the errors inside it; entering it shows a breadcrumb and the two boundary nodes, view-only in 1.0; Esc and
// Alt+↑ go up; "Mở bước bị lỗi" opens the right level, selects the step and focuses the param.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { RunEvent, ValidationIssue } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

beforeAll(async () => {
  await import("../src/components/Editor");
}, 60_000);

async function setup() {
  const { useWorkflow } = await import("../src/store/workflow");
  const { useUi } = await import("../src/store/ui");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const g = useWorkflow.getState().addNode("group.example_lora", { x: 530, y: 70 }) as string;
  const { Editor } = await import("../src/components/Editor");
  const view = render(<Editor />);
  await waitFor(() => expect(view.container.querySelector(`[data-id="${g}"]`)).not.toBeNull());
  return { useWorkflow, useUi, g, ...view };
}

const inner = (g: string, step: string, over: Partial<ValidationIssue> = {}): ValidationIssue => ({
  node_id: g,
  inner_path: [step],
  code: "CFG-005",
  severity: "error",
  params: {},
  title: "Giá trị không hợp lệ",
  what: "Số lượt phải từ 1 tới 5.",
  action_ids: [],
  ...over,
});

describe("collapsed group", () => {
  it("shows the layers mark, an open button, the errors inside it and x/y steps while running", async () => {
    const { useWorkflow, g, container } = await setup();
    const node = container.querySelector(`[data-id="${g}"]`) as HTMLElement;
    expect(node.querySelector(".vlm-node.group .group-mark")).not.toBeNull();
    // React Flow keeps unmeasured nodes visibility:hidden in jsdom, so role queries skip them: query the DOM
    expect(node.querySelector('button[aria-label="Mở nhóm"]')).not.toBeNull();
    act(() =>
      useWorkflow.setState({
        validation: {
          ok: false,
          errors: [inner(g, "train", { param: "epochs" })],
          estimate: {
            ram_gb_peak: null,
            disk_gb: null,
            duration_min: null,
            gpu_exclusive: false,
            source: "unknown",
          },
        },
      }),
    );
    expect(within(node).getByTestId("group-inner-errors").textContent).toContain(
      "1 lỗi bên trong nhóm",
    );
    expect(node.textContent).not.toContain("Chưa đủ"); // inner errors are not the group's own
    // a run: inner steps report under flat ids; the group folds them
    const { api } = await import("../src/api");
    let emit: (ev: RunEvent) => void = () => {};
    vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "r1" });
    vi.spyOn(api, "watchRun").mockImplementation((_id, onEvent) => {
      emit = onEvent;
      return () => {};
    });
    await useWorkflow.getState().startRun();
    act(() =>
      emit({ type: "node_status", node_id: `${g}__split`, status: "completed" } as RunEvent),
    );
    act(() => emit({ type: "node_status", node_id: `${g}__train`, status: "running" } as RunEvent));
    await waitFor(() => expect(node.textContent).toContain("1/3 bước"));
    vi.restoreAllMocks();
  });
});

describe("inside a group", () => {
  it("has a breadcrumb with the lock, two boundary nodes, no editing; Esc and Alt+↑ go up", async () => {
    const { useUi, g, container } = await setup();
    const node = container.querySelector(`[data-id="${g}"]`) as HTMLElement;
    fireEvent.click(node.querySelector('button[aria-label="Mở nhóm"]') as HTMLElement);
    const crumbs = await screen.findByRole("navigation", { name: "Vị trí trong workflow" });
    expect(within(crumbs).getByText("Workflow")).toBeTruthy();
    expect(crumbs.querySelector("[aria-current=page]")?.textContent).toContain(
      "Fine-tune LoRA rồi gộp",
    );
    expect(within(crumbs).getByRole("img", { name: /Nhóm dựng sẵn chỉ xem được/ })).toBeTruthy();
    await waitFor(() =>
      expect(container.querySelector('[data-testid="group-boundary-in"]')?.textContent).toContain(
        "Đầu vào nhóm",
      ),
    );
    expect(container.querySelector('[data-testid="group-boundary-out"]')?.textContent).toContain(
      "Đầu ra nhóm",
    );
    expect(container.querySelector('[data-id="split"]')).not.toBeNull(); // the inner steps
    // nothing selected inside: the inspector sums up the group, not the workflow outside (design r1 #1)
    const { useWorkflow } = await import("../src/store/workflow");
    act(() => useWorkflow.getState().select(null));
    const summary = await screen.findByRole("complementary", { name: "Fine-tune LoRA rồi gộp" });
    expect(summary.textContent).toContain("3 bước");
    expect(summary.textContent).toContain("Nhóm dựng sẵn chỉ xem được");
    expect(summary.textContent).not.toMatch(/Thời gian|RAM/);
    // the parent crumb says where Alt+↑ goes
    expect(within(crumbs).getByText("Workflow").getAttribute("data-tip")).toBe(
      "Lên một cấp (Alt+↑)",
    );
    expect(container.querySelector('[data-id="split"].draggable')).toBeNull(); // view-only
    // Esc while typing in a field stays; Esc on the canvas goes up
    const field = document.createElement("input");
    document.body.appendChild(field);
    field.focus();
    fireEvent.keyDown(field, { key: "Escape" });
    expect(useUi.getState().groupPath).toEqual([g]);
    field.remove();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(useUi.getState().groupPath).toEqual([]);
    act(() => useUi.getState().setGroupPath([g])); // rendered first, so the breadcrumb listens again
    fireEvent.keyDown(window, { key: "ArrowUp", altKey: true });
    expect(useUi.getState().groupPath).toEqual([]);
  });

  it("'Mở bước bị lỗi' opens the right level, selects the step and focuses its param", async () => {
    const { useWorkflow, useUi, g } = await setup();
    act(() =>
      useWorkflow.setState({
        validation: {
          ok: false,
          errors: [inner(g, "train", { param: "epochs" })],
          estimate: {
            ram_gb_peak: null,
            disk_gb: null,
            duration_min: null,
            gpu_exclusive: false,
            source: "unknown",
          },
        },
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Xem chi tiết/ }));
    const row = document.querySelector("[data-code='CFG-005']") as HTMLElement;
    expect(row.textContent).toContain("›"); // the trail of titles, never ids
    expect(row.textContent).not.toContain(`${g}__`);
    fireEvent.click(within(row).getByRole("button", { name: "Mở bước bị lỗi" }));
    expect(useUi.getState().groupPath).toEqual([g]);
    expect(useWorkflow.getState().selectedId).toBe(`${g}__train`);
    const insp = await screen.findByRole("complementary", { name: /LoRA/ });
    expect(insp.textContent).toContain("Nhóm dựng sẵn chỉ xem được");
    await waitFor(() => expect(document.activeElement?.id).toBe("param-epochs"));
  });
});
