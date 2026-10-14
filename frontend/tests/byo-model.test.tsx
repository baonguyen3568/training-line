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
    // one undo step: all three back together, never a mismatched pair
    const before = useWorkflow.getState().past.length;
    act(() => useWorkflow.getState().undo());
    expect(useWorkflow.getState().past.length).toBe(before - 1);
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params).toMatchObject({
      user_model: "",
      user_repo: "",
      user_revision: "",
    });
    act(() => useWorkflow.getState().redo());
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

describe("models already added (H3)", () => {
  const added = {
    id: "user--acme--doc-vlm--1a2b3c4d",
    name: "doc-vlm",
    repo: "acme/doc-vlm",
    revision: "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
    family: "Qwen2.5-VL",
    architecture: "Qwen2_5_VLForConditionalGeneration",
    params_b: 3.8,
    size_gb: 7.6,
    license: "apache-2.0",
    license_confirmed: true,
    gated: false,
    ram_gb_train: 31.5,
    ram_source: "estimated",
    verified: false as const,
  };

  it("picks one from the list without checking it again", async () => {
    const { api } = await import("../src/api");
    vi.spyOn(api, "listUserModels").mockResolvedValue({ models: [added] });
    const add = vi.spyOn(api, "addUserModel");
    const { useWorkflow, id } = await setup();
    const row = await screen.findByRole("button", { name: /acme\/doc-vlm/ });
    expect(row.getAttribute("aria-current")).toBeNull();
    const { t } = await import("../src/i18n");
    expect(screen.queryByText(t("byo.recent"), { exact: false })).toBeNull(); // the list replaces "recent" (spec §1)
    act(() => fireEvent.click(row));
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params).toMatchObject({
      user_model: added.id,
      user_repo: added.repo,
      user_revision: added.revision,
    });
    await waitFor(() => expect(row.getAttribute("aria-current")).toBe("true"));
    expect(row.textContent).toContain(t("byo.model.in_use"));
    expect(add).not.toHaveBeenCalled();
  });

  it("MODEL-006 'Thêm model này' opens the hub tab with the repo and revision typed in, not checked", async () => {
    const { api } = await import("../src/api");
    const add = vi.spyOn(api, "addUserModel");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const { Inspector } = await import("../src/components/Inspector");
    const { QuickFixes } = await import("../src/components/QuickFixes");
    render(
      <>
        <QuickFixes
          nodeId={id}
          issue={{
            node_id: id,
            code: "MODEL-006",
            severity: "error",
            params: { node: "Chọn model gốc", repo: added.repo, revision: added.revision },
            title: "…",
            what: "…",
            action_ids: ["add_user_model"],
            actions: [
              {
                id: "add_user_model",
                kind: "navigate",
                confirm: false,
                node_id: id,
                target: "add_model_dialog",
              },
            ],
          }}
        />
        <Inspector />
      </>,
    );
    const { t } = await import("../src/i18n");
    act(() => fireEvent.click(screen.getByRole("button", { name: t("action.add_user_model") })));
    const tab = await screen.findByRole("tab", { name: /Từ Hugging Face/ });
    expect(tab.getAttribute("aria-selected")).toBe("true");
    expect((screen.getByPlaceholderText(/Qwen\/Qwen3.5-2B/) as HTMLInputElement).value).toBe(
      added.repo,
    );
    expect((screen.getByLabelText(t("byo.revision")) as HTMLInputElement).value).toBe(
      added.revision,
    );
    expect(add).not.toHaveBeenCalled(); // the user presses "Kiểm tra"
  });
});

describe("unknown licence", () => {
  it("says export stays blocked, above the 'I agree' tick, and never shows the raw id", async () => {
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    const { t } = await import("../src/i18n");
    render(
      <DownloadDialog
        target={{
          kind: "model",
          catalogId: "user--a--b--0f3a9c1e",
          name: "b",
          license: "unknown",
          licenseCard: `<b>research only</b> ${"x".repeat(300)}`,
        }}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain(t("download.license_unknown"));
    expect(dialog.textContent).toContain(t("license.badge.unknown"));
    expect(dialog.textContent).not.toMatch(/\bunknown\b/);
    // the author's own words, as plain text (untrusted): the tag stays text, never markup; no copy → no line, no key
    const { cardSays } = await import("../src/lib/licenses");
    const card = `<b>research only</b> ${"x".repeat(300)}`;
    const line = dialog.querySelector(".license-card") as HTMLElement;
    expect(line.textContent).toBe(cardSays(card)); // cut at 200 with "…"
    expect(line.textContent).toContain("<b>research only</b>");
    expect(line.textContent?.endsWith("…”")).toBe(true);
    expect(line.title).toBe(card); // the whole of it on hover
    expect(dialog.textContent).not.toContain("license.card_says");
    expect(line.querySelector("b")).toBeNull();
  });

  it("a user dataset's licence words come from the added list when the issue has none", async () => {
    const { api } = await import("../src/api");
    vi.spyOn(api, "listUserDatasets").mockResolvedValue({
      datasets: [
        {
          id: "userds--acme--invoices--7f0115a4",
          name: "acme/invoices (default/train)",
          repo: "acme/invoices",
          revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
          split: "train",
          license: "unknown",
          license_card: "other (acme)",
          license_confirmed: false,
          verified: false,
        },
      ],
    });
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    const { cardSays } = await import("../src/lib/licenses");
    render(
      <DownloadDialog
        target={{
          kind: "dataset",
          catalogId: "userds--acme--invoices--7f0115a4",
          name: "acme/invoices",
          license: "unknown",
        }}
        onClose={() => {}}
      />,
    );
    await waitFor(() =>
      expect(document.querySelector(".license-card")?.textContent).toBe(cardSays("other (acme)")),
    );
  });

  it("a user model's licence class comes from the server; none means unknown, never commercial", async () => {
    const { api } = await import("../src/api");
    const { t } = await import("../src/i18n");
    const base = {
      id: "user--acme--vlm--1a2b3c4d",
      name: "vlm",
      repo: "acme/vlm",
      revision: "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
      family: "Qwen2.5-VL",
      architecture: "Qwen2_5_VLForConditionalGeneration",
      params_b: 3.8,
      size_gb: 7.6,
      license: "apache-2.0",
      license_confirmed: false,
      gated: false,
      ram_gb_train: 31.5,
      ram_source: "estimated",
      verified: false as const,
    };
    const list = vi.spyOn(api, "listUserModels").mockResolvedValue({ models: [base] });
    const { DownloadDialog } = await import("../src/components/DownloadDialog");
    const target = {
      kind: "model" as const,
      catalogId: base.id,
      name: "vlm",
      license: "apache-2.0",
    };
    const { unmount } = render(<DownloadDialog target={target} onClose={() => {}} />);
    // no class from the server: the unknown verdict, even for an id that reads like an open licence
    await waitFor(() => expect(list).toHaveBeenCalled());
    await waitFor(() =>
      expect(document.querySelector(".license-verdict")?.textContent).toContain(
        t("license.badge.unknown"),
      ),
    );
    unmount();
    list.mockResolvedValue({ models: [{ ...base, license_class: "commercial_ok" }] });
    render(<DownloadDialog target={target} onClose={() => {}} />);
    await waitFor(() =>
      expect(document.querySelector(".license-verdict")?.textContent).toContain(
        t("license.badge.commercial_ok"),
      ),
    );
  });
});

describe("MODEL-005 in the mock follows the server", () => {
  it("only LoRA and full SFT steps get the trial hint", async () => {
    const { mockValidate } = await import("../src/mocks/api");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const s = useWorkflow.getState();
    const types = [...s.specs.values()];
    const wf = (train: string) => ({
      version: 1,
      name: "t",
      nodes: [
        { id: "m", type: "model.hf_pick", params: { user_model: "user--a--b--0f3a9c1e" } },
        { id: "t", type: train, params: {} },
      ],
      edges: [{ from: ["m", "model"], to: ["t", "base_model"] }],
    });
    const codes = (train: string) =>
      // biome-ignore lint/suspicious/noExplicitAny: a minimal workflow literal for the mock validator
      mockValidate(wf(train) as any, types, s.ports).errors.map((e) => e.code);
    expect(codes("train.lora_vlm")).toContain("MODEL-005");
    expect(codes("train.sft_vlm")).toContain("MODEL-005");
    expect(codes("train.dpo_vlm")).not.toContain("MODEL-005");
    expect(codes("train.grpo_vlm")).not.toContain("MODEL-005");
  });
});
