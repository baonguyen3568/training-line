// Storage panel (G-1) on the real plan v3 sample from bao-cleanup: sections, typed confirmation, 409/403 handling.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

async function setup() {
  const { StoragePanel } = await import("../src/components/StoragePanel");
  const view = render(<StoragePanel />);
  // first test of the file pays the cold dynamic import; 1 s (default) is too tight on the shared host
  await screen.findByRole("heading", { name: "Dung lượng" }, { timeout: 5000 });
  return view;
}
const row = (container: HTMLElement, id: string) =>
  container.querySelector(`[data-item="${id}"]`) as HTMLElement;

afterEach(() => vi.restoreAllMocks());

describe("StoragePanel", () => {
  it("groups by section and never offers a kept item for deletion", async () => {
    const { container } = await setup();
    for (const s of ["Cần bạn chọn", "App sẽ tự dọn khi cần", "Được giữ lại"])
      expect(screen.getByRole("heading", { name: s })).toBeTruthy();
    const kept = row(container, "r-keep/checkpoints/epoch_0_step_10");
    expect(kept.querySelector("input[type=checkbox]")).toBeNull();
    expect(kept.textContent).toContain("Đã được ghim");
    expect(kept.querySelector(".storage-size")?.textContent).toBe("—"); // size_gb null
    expect(screen.getByText(/Bộ nhớ đệm đầy/)).toBeTruthy(); // DISK-003 rendered by the server
  });

  it("reminds about an interrupted run close to its resume deadline, and says why its checkpoint is kept", async () => {
    const { container } = await setup();
    const card = container.querySelector('[data-code="RUN_INTERRUPTED_EXPIRING"]');
    expect(card?.textContent).toContain('Bước "Dạy model bằng LoRA" (run "r-int")');
    expect(card?.textContent).toContain("trước 09:00 14/10");
    expect(card?.textContent).not.toMatch(/\{\w+\}/);
    const kept = row(container, "r-int/checkpoints/epoch_0_step_10");
    expect(kept.querySelector("input[type=checkbox]")).toBeNull();
    expect(kept.textContent).toContain("tới 09:00 14/10");
  });

  it("falls back to the server sentence when the app has no copy for a reminder code", async () => {
    const { api } = await import("../src/api");
    const real = await api.storagePlan("vi");
    real.attention = [
      { code: "NEW_CODE", run_id: "r", node: "n", resume_until: "", message: "Câu từ server." },
    ];
    vi.spyOn(api, "storagePlan").mockResolvedValue(real);
    await setup();
    expect(screen.getByText("Câu từ server.")).toBeTruthy();
  });

  it("shows how much the kept section holds once sizes are known", async () => {
    const { api } = await import("../src/api");
    const real = await api.storagePlan("vi");
    real.items = real.items.map((i) => (i.section === "kept" ? { ...i, size_gb: 31.5 } : i));
    vi.spyOn(api, "storagePlan").mockResolvedValue(real);
    await setup();
    expect(screen.getByText("Đang giữ 63.0 GB")).toBeTruthy(); // 2 kept items × 31.5
  });

  it("locks an item that needs typing but has no confirm_text (it could never pass)", async () => {
    const { api } = await import("../src/api");
    const real = await api.storagePlan("vi");
    real.items = real.items.map((i) =>
      i.item_id === "train-1" ? { ...i, confirm_required: true, confirm_text: null } : i,
    );
    vi.spyOn(api, "storagePlan").mockResolvedValue(real);
    const { container } = await setup();
    expect(row(container, "train-1").querySelector("input[type=checkbox]")).toBeNull();
  });

  it("deletes a PERMANENT_LOSS item only after its name is typed", async () => {
    const { api } = await import("../src/api");
    const applySpy = vi.spyOn(api, "storageApply");
    const { container } = await setup();
    const id = "r-fail/checkpoints/epoch_0_step_10";
    fireEvent.click(within(row(container, id)).getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Xoá 1 mục đã chọn · 0\.0 GB/ }));
    const dialog = screen.getByRole("alertdialog");
    const confirm = within(dialog).getByRole("button", { name: "Xoá vĩnh viễn" });
    expect(confirm).toHaveProperty("disabled", true);
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "r-fai" } });
    expect(confirm).toHaveProperty("disabled", true);
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "r-fail" } });
    expect(confirm).toHaveProperty("disabled", false);
    fireEvent.click(confirm);
    await waitFor(() => expect(screen.getByTestId("storage-result")).toBeTruthy());
    expect(applySpy).toHaveBeenCalledWith(
      expect.objectContaining({ item_ids: [id], confirm: { [id]: "r-fail" } }),
      "vi",
    );
  });

  it("treats CLEANUP_IN_PROGRESS as a wait, and DISK-002 as an admin command, not a crash", async () => {
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    const { container } = await setup();
    fireEvent.click(within(row(container, "eval-1")).getByRole("checkbox"));
    vi.spyOn(api, "storageApply").mockRejectedValueOnce(
      new ApiError(409, { code: "CLEANUP_IN_PROGRESS", plan_id: "x" }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Xoá 1 mục đã chọn · 0\.0 GB/ }));
    fireEvent.click(screen.getByRole("button", { name: "Xoá vĩnh viễn" }));
    await waitFor(() =>
      expect(screen.getByText(/Đang có một lượt dọn dẹp khác chạy/)).toBeTruthy(),
    );

    vi.spyOn(api, "storageApply").mockRejectedValueOnce(
      new ApiError(403, {
        error: {
          code: "DISK-002",
          title: "Cần quản trị viên duyệt",
          what: "Tổng dung lượng xoá vượt ngưỡng.",
          params: {
            admin_command: "uv run python scripts/cleanup/cache.py --items 'eval-1' --apply",
          },
        },
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Xoá vĩnh viễn" }));
    await waitFor(() => expect(screen.getByText(/scripts\/cleanup\/cache.py/)).toBeTruthy());
    expect(screen.getByRole("alert").textContent).toContain("Cần quản trị viên duyệt");
  });
});
