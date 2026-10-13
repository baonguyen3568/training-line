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

  it("lists a broken eval dataset (source eval_data, G-5) as a dataset item it can delete", async () => {
    const { api } = await import("../src/api");
    const real = await api.storagePlan("vi");
    const first = real.items.find((i) => i.section === "auto") ?? real.items[0];
    if (!first) throw new Error("sample plan has no items");
    const evalItem = {
      ...first,
      item_id: "eval:ocrbench",
      source: "eval_data" as const,
      section: "auto" as const,
      group: "dataset",
      node_base: null,
      display_name: null,
      display_key: "storage.item.eval_data_unfinished",
      display_params: { dataset: "OCRBench" },
      size_gb: 1.2,
      confirm_required: false,
      protected_reason: null,
      action: "delete" as const,
    };
    const cacheItem = {
      ...evalItem,
      item_id: "evalcache:ocrbench",
      source: "eval_cache" as const,
      group: "temp_result",
      display_key: "storage.item.eval_cache_leftover",
    };
    vi.spyOn(api, "storagePlan").mockResolvedValue({
      ...real,
      items: [...real.items, evalItem, cacheItem],
    });
    const { container } = await setup();
    const c = row(container, "evalcache:ocrbench"); // outputs/eval-cache leftovers, a temporary result
    expect(c.textContent).toContain("OCRBench");
    expect(c.textContent).not.toContain("evalcache:");
    const r = row(container, "eval:ocrbench");
    expect(r).toBeTruthy();
    expect(r.querySelector("input[type=checkbox]")).not.toBeNull(); // regenerable, no typed confirmation
    expect(r.textContent).toContain("Dữ liệu"); // storage.group.dataset
    // copy or not, the row names the dataset and never shows the internal id (design)
    expect(r.textContent).toContain("OCRBench");
    expect(r.textContent).not.toContain("eval:ocrbench");
  });

  it("a recent broken eval dataset asks you (manual), shows when it was set aside and when the app cleans it", async () => {
    const { api } = await import("../src/api");
    const { t } = await import("../src/i18n");
    const real = await api.storagePlan("vi");
    const first = real.items[0];
    if (!first) throw new Error("sample plan has no items");
    const invalid = {
      ...first,
      item_id: "eval:docvqa",
      run_id: ".docvqa-92a54bd1.invalid-20261013-214000-ab12",
      source: "eval_data" as const,
      section: "manual" as const,
      group: "dataset",
      node_base: null,
      display_name: null,
      display_key: "storage.item.eval_data_invalid",
      display_params: { dataset: "DocVQA", stamp: "2026-10-13 21:40" },
      note_key: "storage.item_note.eval_data_auto_clean",
      note_params: { days: "7" },
      size_gb: 0.8,
      confirm_required: false,
      confirm_text: null,
      lock_reason: null,
      protected_reason: null,
      consequence_code: null,
      action: "delete" as const,
    };
    vi.spyOn(api, "storagePlan").mockResolvedValue({ ...real, items: [...real.items, invalid] });
    const { container } = await setup();
    const r = row(container, "eval:docvqa");
    expect(r.closest("section")?.querySelector("h2, h3")?.textContent).toBe("Cần bạn chọn");
    expect(r.textContent).toContain("DocVQA");
    expect(r.textContent).toContain("21:40 13/10"); // stamp like {until}, not the raw "2026-10-13 21:40"
    expect(r.textContent).not.toContain("2026-10-13");
    expect(r.textContent).toContain(t("storage.item_note.eval_data_auto_clean", { days: "7" }));
    expect(r.querySelector("input[type=checkbox]")).not.toBeNull();
    expect(r.querySelector("input[type=text]")).toBeNull(); // no typed confirmation
  });
});
