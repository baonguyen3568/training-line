// G-5 "Bấm Chạy khi thiếu": Run with NET-002 warns opens the list instead of running; one "I agree" for all
// licences; downloads chain one after another and the run starts only when all completed; a failure stops the
// chain without running; a gated model is never chained and locks "Tải rồi chạy" until it is downloaded.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ValidationIssue } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

const warn = (catalogId: string, kind: "model" | "dataset", size: number): ValidationIssue => ({
  node_id: "n1",
  code: "NET-002",
  severity: "warn",
  params: {
    repo: `org/${catalogId}`,
    reason: "not_cached",
    catalog_id: catalogId,
    kind,
    size_gb: size,
  },
  title: "Model hoặc dữ liệu chưa có trên máy",
  what: "…",
  action_ids: [],
});

async function withWarns(issues: ValidationIssue[]) {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  useWorkflow.setState({
    validation: {
      ok: true,
      errors: issues,
      estimate: {
        ram_gb_peak: null,
        disk_gb: null,
        duration_min: null,
        gpu_exclusive: false,
        source: "unknown",
      },
    },
  });
  return useWorkflow;
}

describe("requestRun", () => {
  it("opens the list when something is missing, runs at once otherwise", async () => {
    const useWorkflow = await withWarns([warn("a", "model", 1.5), warn("a", "model", 1.5)]);
    const { useUi } = await import("../src/store/ui");
    const { requestRun, missingDownloads } = await import("../src/lib/requestRun");
    const run = vi.spyOn(useWorkflow.getState(), "startRun").mockResolvedValue();
    expect(missingDownloads()).toHaveLength(1); // one asset, even when two steps need it
    requestRun();
    expect(useUi.getState().runMissingOpen).toBe(true);
    expect(run).not.toHaveBeenCalled();
    useUi.setState({ runMissingOpen: false });
    const v = useWorkflow.getState().validation;
    if (!v) throw new Error("no validation");
    useWorkflow.setState({ validation: { ...v, errors: [] } });
    requestRun();
    expect(run).toHaveBeenCalled();
  });
});

describe("run-missing dialog", () => {
  it("downloads everything after one 'I agree', then validates and runs", async () => {
    const useWorkflow = await withWarns([warn("a", "model", 1.5), warn("b", "dataset", 0.5)]);
    const { useDownloads } = await import("../src/store/downloads");
    const all = vi.spyOn(useDownloads.getState(), "downloadAll").mockResolvedValue(null);
    const validate = vi.spyOn(useWorkflow.getState(), "validate").mockResolvedValue();
    const run = vi.spyOn(useWorkflow.getState(), "startRun").mockResolvedValue();
    const onClose = vi.fn();
    const { RunMissingDialog } = await import("../src/components/RunMissingDialog");
    render(<RunMissingDialog onClose={onClose} />);
    expect(screen.getByText(/Workflow cần 2 model hoặc bộ dữ liệu/)).toBeTruthy();
    expect(screen.getByText("Tổng khoảng 2.0 GB")).toBeTruthy();
    const go = screen.getByRole("button", { name: /Tải rồi chạy/ });
    expect(go).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(go);
    await waitFor(() => expect(run).toHaveBeenCalled());
    expect(all.mock.calls[0]?.[0].map((x) => x.catalogId)).toEqual(["a", "b"]);
    expect(validate).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("a failed download stops the chain and the workflow does not run", async () => {
    const useWorkflow = await withWarns([warn("a", "model", 1.5), warn("b", "dataset", 0.5)]);
    const { useDownloads } = await import("../src/store/downloads");
    vi.spyOn(useDownloads.getState(), "downloadAll").mockResolvedValue("model:a");
    const run = vi.spyOn(useWorkflow.getState(), "startRun").mockResolvedValue();
    const { RunMissingDialog } = await import("../src/components/RunMissingDialog");
    render(<RunMissingDialog onClose={() => {}} />);
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Tải rồi chạy/ }));
    expect((await screen.findByRole("alert")).textContent).toContain("Workflow chưa chạy.");
    expect(run).not.toHaveBeenCalled();
  });

  it("a gated model is not chained: its row opens its own download and Run waits for it", async () => {
    await withWarns([warn("a", "model", 1.5)]);
    const target = await import("../src/lib/downloadTarget");
    const real = target.downloadTargetOf;
    vi.spyOn(target, "downloadTargetOf").mockImplementation((i, s) => {
      const x = real(i, s);
      return x && { ...x, gated: true };
    });
    const { useUi } = await import("../src/store/ui");
    const onClose = vi.fn();
    const { RunMissingDialog } = await import("../src/components/RunMissingDialog");
    render(<RunMissingDialog onClose={onClose} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(screen.getByRole("button", { name: /Tải rồi chạy/ })).toHaveProperty("disabled", true);
    expect(screen.getByText(/cần bạn tự đồng ý điều khoản trên Hugging Face/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Tải về máy" }));
    expect(onClose).toHaveBeenCalled();
    expect(useUi.getState().downloadTarget?.catalogId).toBe("a");
  });
});

describe("prepare only row", () => {
  it("a not_prepared asset says it only needs preparing and adds nothing to the total", async () => {
    const prep = {
      ...warn("p", "model", 1.9),
      params: { ...warn("p", "model", 1.9).params, reason: "not_prepared" },
    };
    await withWarns([prep, warn("b", "dataset", 0.5)]);
    const { RunMissingDialog } = await import("../src/components/RunMissingDialog");
    render(<RunMissingDialog onClose={() => {}} />);
    expect(screen.getByText(/Chỉ cần chuẩn bị, không tải lại/)).toBeTruthy();
    expect(screen.getByText("Tổng khoảng 0.5 GB")).toBeTruthy();
  });
});
