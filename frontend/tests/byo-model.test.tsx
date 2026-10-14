// ADR-021 / spec H §1-3: the base model step has "Đã kiểm chứng | Từ Hugging Face"; a pasted repo is checked on
// Enter (never per keystroke), the result shows what it is with "Chưa kiểm chứng", "Dùng model này" sets user_model;
// a model that needs its own code is refused with the catalog's words and no "Dùng".
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

async function setup() {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
  useWorkflow.getState().select(id);
  const { Inspector } = await import("../src/components/Inspector");
  render(<Inspector />);
  fireEvent.click(screen.getByRole("tab", { name: /Từ Hugging Face/ }));
  return { useWorkflow, id };
}

describe("model from Hugging Face", () => {
  it("checks on Enter only, shows the unverified model, and 'Dùng model này' picks it", async () => {
    const { api } = await import("../src/api");
    const add = vi.spyOn(api, "addUserModel");
    const { useWorkflow, id } = await setup();
    const input = screen.getByPlaceholderText(/Qwen\/Qwen3.5-2B/);
    fireEvent.change(input, { target: { value: "Qwen/Qwen2.5-VL-3B-Instruct" } });
    expect(add).not.toHaveBeenCalled(); // typing alone never calls the server
    fireEvent.keyDown(input, { key: "Enter" });
    expect(await screen.findByText(/Đang đọc thông tin/)).toBeTruthy();
    const card = await screen.findByTestId("byo-model-result", {}, { timeout: 4000 });
    expect(card.textContent).toContain("Chưa kiểm chứng");
    expect(card.textContent).toContain("Qwen2.5-VL · ~3.8B tham số · tải ~7.6 GB");
    expect(card.textContent).toContain("0f3a9c1e2b7d"); // the pinned sha, 12 characters
    act(() => fireEvent.click(screen.getByRole("button", { name: "Dùng model này" })));
    // all three together (the server refuses them when they disagree)
    await waitFor(() =>
      expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params).toMatchObject(
        {
          user_model: "user--Qwen--Qwen2.5-VL-3B-Instruct--0f3a9c1e",
          user_repo: "Qwen/Qwen2.5-VL-3B-Instruct",
          user_revision: "0f3a9c1e2b7d4a6f8e9c0b1a2d3e4f5a6b7c8d9e",
        },
      ),
    );
    // back to "Đã kiểm chứng": all three cleared
    act(() => fireEvent.click(screen.getByRole("tab", { name: /Đã kiểm chứng/ })));
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params).toMatchObject({
      user_model: "",
      user_repo: "",
      user_revision: "",
    });
  });

  it("a model that runs its own code is refused with the catalog's words, without 'Dùng'", async () => {
    await setup();
    const input = screen.getByPlaceholderText(/Qwen\/Qwen3.5-2B/);
    fireEvent.change(input, { target: { value: "someone/remote-vlm" } });
    fireEvent.click(screen.getByRole("button", { name: "Kiểm tra" }));
    const err = await screen.findByRole("alert", {}, { timeout: 4000 });
    expect(err.getAttribute("data-code")).toBe("MODEL-004");
    const { tpOptional } = await import("../src/i18n");
    expect(err.textContent).toContain(tpOptional(["error", "MODEL-004", "title"]) as string); // fixcode's words
    expect(screen.queryByRole("button", { name: "Dùng model này" })).toBeNull();
  });
});

describe("refused adds", () => {
  it("the download slot busy (409) says so and 'Kiểm tra' works again", async () => {
    await setup();
    const input = screen.getByPlaceholderText(/Qwen\/Qwen3.5-2B/);
    fireEvent.change(input, { target: { value: "someone/busy-vlm" } });
    fireEvent.click(screen.getByRole("button", { name: "Kiểm tra" }));
    const err = await screen.findByRole("alert");
    expect(err.getAttribute("data-code")).toBe("SYS-002");
    const button = screen.getByRole("button", { name: "Kiểm tra" }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false)); // retry is one click away
  });

  it("a 422 schema refusal is the format error of the field it names, never SYS-000", async () => {
    const { ApiError } = await import("../src/api/client");
    const { hubError } = await import("../src/lib/hubJob");
    const body = (loc: string[]) => ({ detail: [{ loc, msg: "x", type: "value_error" }] });
    expect(hubError(new ApiError(422, body(["body", "revision"])))).toEqual({
      code: "CFG-005",
      params: { param: "revision", rule: "format" },
    });
    expect(hubError(new ApiError(422, body(["body", "repo"]))).params?.param).toBe("repo");
    expect(hubError(new Error("offline")).code).toBe("SYS-000");
  });
});

describe("download of a user model", () => {
  it("confirms the licence with the server before it starts the download", async () => {
    const { api } = await import("../src/api");
    const order: string[] = [];
    vi.spyOn(api, "confirmUserModelLicense").mockImplementation(async () => {
      order.push("confirm");
      return { model: {} as never };
    });
    vi.spyOn(api, "startDownload").mockImplementation(async () => {
      order.push("download");
      return { download_id: null, status: "completed", verified: true };
    });
    const { useDownloads } = await import("../src/store/downloads");
    await useDownloads.getState().start("model", "user--Qwen--Qwen2.5-VL-3B-Instruct--0f3a9c1e");
    expect(order).toEqual(["confirm", "download"]);
    await useDownloads.getState().start("model", "qwen3.5-0.8b"); // a catalog model: nothing to confirm
    expect(order).toEqual(["confirm", "download", "download"]);
  });
});

describe("quick fixes of ADR-021", () => {
  it("'Chạy thử nhanh' starts a trial run of that step and leaves its params alone", async () => {
    const { api } = await import("../src/api");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const train = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    const before = structuredClone(
      useWorkflow.getState().workflow?.nodes.find((n) => n.id === train)?.params,
    );
    const start = vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "trial-1" });
    vi.spyOn(api, "watchRun").mockImplementation(() => () => {});
    const { QuickFixes } = await import("../src/components/QuickFixes");
    render(
      <QuickFixes
        nodeId={train}
        issue={{
          node_id: train,
          code: "MODEL-005",
          severity: "info",
          params: {},
          title: "Model chưa kiểm chứng",
          what: "…",
          action_ids: ["run_smoke"],
          actions: [
            { id: "run_smoke", kind: "trial", confirm: false, node_id: train, target: null },
          ],
        }}
      />,
    );
    const btn = screen.getAllByRole("button")[0] as HTMLElement;
    fireEvent.click(btn);
    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[1]).toEqual({ node: train });
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === train)?.params).toEqual(
      before,
    );
  });
});

describe("unknown licence", () => {
  it("says export stays blocked, above the 'I agree' tick, and never shows the raw id", async () => {
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    const { t } = await import("../src/i18n");
    render(
      <DownloadDialog
        target={{ kind: "model", catalogId: "user--a--b--0f3a9c1e", name: "b", license: "unknown" }}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain(t("download.license_unknown"));
    expect(dialog.textContent).toContain(t("license.badge.unknown"));
    expect(dialog.textContent).not.toMatch(/\bunknown\b/);
  });
});
