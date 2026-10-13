// G-5 "Tải về máy": a NET-002 (not_cached) step offers the download; the dialog needs "I agree" before starting,
// asks again for a large download, shows progress and re-validates when done. not_prepared (a bridged model the
// server prepares after the download) gets the button too; not_pinned does not.
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DownloadEvent, ValidationIssue } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

const net002 = (reason: string): ValidationIssue => ({
  node_id: "n1",
  code: "NET-002",
  severity: "error",
  params: {
    repo: "Qwen/Qwen3.5-0.8B",
    reason,
    catalog_id: "qwen3.5-0.8b",
    kind: "model",
    size_gb: 1.6,
  },
  title: "Chưa tải về máy",
  what: "…",
  action_ids: [],
});

describe("download target", () => {
  it("not_cached and not_prepared are fixed by a download, not_pinned is not", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const { downloadTargetOf } = await import("../src/lib/downloadTarget");
    const specs = useWorkflow.getState().specs;
    const target = downloadTargetOf(net002("not_cached"), specs);
    expect(target).toMatchObject({ kind: "model", catalogId: "qwen3.5-0.8b", sizeGb: 1.6 });
    expect(target?.name).not.toBe("qwen3.5-0.8b"); // the catalog label
    expect(downloadTargetOf(net002("not_prepared"), specs)?.catalogId).toBe("qwen3.5-0.8b");
    expect(downloadTargetOf(net002("not_pinned"), specs)).toBeUndefined();
    // the publisher is never the lowercase slug: catalog vendor_name, else the repo's org
    expect(target?.publisher).toBe("Qwen");
  });
});

describe("download dialog", () => {
  it("needs I agree, asks again when large, then shows progress and completes", async () => {
    const { api } = await import("../src/api");
    const { useWorkflow } = await import("../src/store/workflow");
    const validate = vi.spyOn(useWorkflow.getState(), "validate").mockResolvedValue();
    let emit: (ev: DownloadEvent) => void = () => {};
    const start = vi
      .spyOn(api, "startDownload")
      .mockResolvedValueOnce({
        download_id: null,
        needs_confirm: true,
        size_gb: 24.3,
        threshold_gb: 20,
      })
      .mockResolvedValueOnce({ download_id: "d1", status: "queued", size_gb: 24.3 });
    vi.spyOn(api, "watchDownload").mockImplementation((_id, onEvent) => {
      emit = onEvent;
      return () => {};
    });
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    render(
      <DownloadDialog
        target={{
          kind: "model",
          catalogId: "big",
          name: "Big model",
          sizeGb: 24.3,
          license: "apache-2.0",
        }}
        onClose={() => {}}
      />,
    );
    const go = screen.getByRole("button", { name: /Tải về/ });
    expect(go).toHaveProperty("disabled", true);
    expect(screen.getByText("Apache-2.0")).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(go);
    await screen.findByText(/vượt ngưỡng 20 GB/);
    fireEvent.click(screen.getByRole("button", { name: /Tải về/ }));
    await waitFor(() =>
      expect(start).toHaveBeenLastCalledWith({ kind: "model", catalog_id: "big", confirm: true }),
    );
    act(() =>
      emit({
        type: "progress",
        files_done: 1,
        files_total: 3,
        bytes_done: 8e9,
        bytes_total: 24.3e9,
      }),
    );
    expect(screen.getByText("8.0/24.3 GB · 1/3 tệp")).toBeTruthy();
    act(() => emit({ type: "status", status: "verifying" }));
    expect(screen.getByText(/Đang kiểm tra tệp vừa tải/)).toBeTruthy();
    act(() => emit({ type: "status", status: "completed" }));
    expect(screen.getByText('Đã tải "Big model" về máy.')).toBeTruthy();
    expect(validate).toHaveBeenCalled();
  });

  it("a refused start shows the failure with the server's code title, never a trace", async () => {
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    vi.spyOn(api, "startDownload").mockRejectedValue(
      new ApiError(409, { detail: { code: "DISK-001", params: {} } }),
    );
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    render(
      <DownloadDialog target={{ kind: "model", catalogId: "x2", name: "X" }} onClose={() => {}} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Tải về/ }));
    const card = await screen.findByRole("alert");
    expect(card.getAttribute("data-code")).toBe("DISK-001");
    expect(card.textContent).toContain('Chưa tải được "X".');
  });

  it("a start that lands on the busy slot (SYS-002) is tried once more; the server's name and licence win", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { api } = await import("../src/api");
      const { ApiError } = await import("../src/api/client");
      const { useDownloads } = await import("../src/store/downloads");
      const start = vi
        .spyOn(api, "startDownload")
        .mockRejectedValueOnce(new ApiError(409, { detail: { code: "SYS-002", params: {} } }))
        .mockResolvedValueOnce({
          download_id: null,
          status: "completed",
          verified: true,
          name: "Qwen3.5 0.8B",
          license: "apache-2.0",
        });
      const run = useDownloads.getState().start("model", "x3");
      await vi.advanceTimersByTimeAsync(2000);
      await run;
      expect(start).toHaveBeenCalledTimes(2);
      expect(useDownloads.getState().entries["model:x3"]).toMatchObject({
        status: "completed",
        name: "Qwen3.5 0.8B",
        license: "apache-2.0",
      });
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("node face", () => {
  it("a not-cached model step shows the badge and a download button that opens the dialog", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const { container } = render(<Editor />);
    act(() =>
      useWorkflow.setState({
        validation: {
          ok: false,
          errors: [{ ...net002("not_cached"), node_id: id }],
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
    const node = container.querySelector(`[data-id="${id}"]`);
    expect(node?.textContent).toContain("Chưa tải về máy");
    // not the user's mistake: no "Chưa đủ (1)" count, no red edge (design G-5 #1)
    expect(node?.textContent).not.toContain("Chưa đủ");
    expect(node?.classList.contains("has-error")).toBe(false);
    const btn = [...(node?.querySelectorAll("button") ?? [])].find((b) =>
      b.textContent?.includes("Tải về máy"),
    );
    if (!btn) throw new Error("no download button on the node");
    fireEvent.click(btn);
    expect(useUi.getState().downloadTarget?.catalogId).toBe("qwen3.5-0.8b");
    expect(await screen.findByRole("dialog")).toBeTruthy();
  });
});

describe("run error", () => {
  it("a step that failed with NET-002 while running offers the download in the inspector", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { Inspector } = await import("../src/components/Inspector");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const { code, params } = net002("not_cached");
    useWorkflow.setState({
      run: {
        id: "r1",
        nodes: { [id]: { status: "failed", errorCode: code, error: { code, params } } },
        done: true,
        startedAt: 0,
      },
    });
    useWorkflow.getState().select(id);
    render(<Inspector />);
    fireEvent.click(await screen.findByRole("button", { name: /Tải về máy/ }));
    expect(useUi.getState().downloadTarget?.catalogId).toBe("qwen3.5-0.8b");
  });
});

describe("download chain", () => {
  it("downloads one after another and stops at the first that fails", async () => {
    const { api } = await import("../src/api");
    const { useDownloads } = await import("../src/store/downloads");
    const { useWorkflow } = await import("../src/store/workflow");
    vi.spyOn(useWorkflow.getState(), "validate").mockResolvedValue();
    const emits: Record<string, (ev: DownloadEvent) => void> = {};
    const start = vi
      .spyOn(api, "startDownload")
      .mockImplementation(async (b) => ({ download_id: `d-${b.catalog_id}`, status: "queued" }));
    vi.spyOn(api, "watchDownload").mockImplementation((id, onEvent) => {
      emits[id] = onEvent;
      return () => {};
    });
    const chain = useDownloads.getState().downloadAll([
      { kind: "model", catalogId: "a" },
      { kind: "dataset", catalogId: "b" },
      { kind: "model", catalogId: "c" },
    ]);
    await waitFor(() => expect(emits["d-a"]).toBeDefined());
    expect(start).toHaveBeenCalledTimes(1); // b waits for a
    expect(start.mock.calls[0]?.[0]).toMatchObject({ catalog_id: "a", confirm: true });
    act(() => emits["d-a"]?.({ type: "status", status: "completed" }));
    await waitFor(() => expect(emits["d-b"]).toBeDefined());
    act(() =>
      emits["d-b"]?.({ type: "status", status: "failed", error: { code: "DISK-001", params: {} } }),
    );
    expect(await chain).toBe("dataset:b");
    expect(start).toHaveBeenCalledTimes(2); // c never started
  });
});

describe("unverified", () => {
  it("a copy already on disk but not checked offers 'Kiểm tra tệp', which re-sends with check", async () => {
    const { api } = await import("../src/api");
    const start = vi
      .spyOn(api, "startDownload")
      .mockResolvedValueOnce({ download_id: null, status: "completed", verified: false })
      .mockResolvedValueOnce({ download_id: null, status: "completed", verified: true });
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    render(
      <DownloadDialog target={{ kind: "model", catalogId: "x4", name: "X" }} onClose={() => {}} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Tải về/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Kiểm tra tệp" }));
    await screen.findByText('Đã tải "X" về máy.');
    expect(start.mock.calls[1]?.[0]).toMatchObject({ catalog_id: "x4", check: true });
  });
});
