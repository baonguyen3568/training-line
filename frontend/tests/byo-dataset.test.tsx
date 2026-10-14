// ADR-021 / spec H §4: a dataset from Hugging Face is inspected, its columns mapped in a dialog with the server's
// suggestions and a preview, a wrong column kind is said under its select, and the mapping lands in the step.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

async function setup() {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const id = useWorkflow.getState().addNode("data.hf_load", { x: 0, y: 0 }) as string;
  useWorkflow.getState().select(id);
  const { Inspector } = await import("../src/components/Inspector");
  render(<Inspector />);
  fireEvent.click(screen.getByRole("tab", { name: /Từ Hugging Face/ }));
  return { useWorkflow, id };
}

describe("dataset from Hugging Face", () => {
  it("maps columns with suggestions and a preview, refuses a wrong kind, then fills the step", async () => {
    const { useWorkflow, id } = await setup();
    const input = screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/);
    fireEvent.change(input, { target: { value: "acme/invoices-vi" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const dialog = await screen.findByRole(
      "dialog",
      { name: "Ghép cột dữ liệu" },
      { timeout: 4000 },
    );
    const image = within(dialog).getByLabelText("Cột ảnh") as HTMLSelectElement;
    expect(image.value).toBe("image"); // the server's suggestion
    const hints = () => within(dialog).queryAllByText(/· gợi ý/).length;
    const suggested = hints();
    expect(suggested).toBeGreaterThan(0); // beside the label, only while the value is the server's
    expect(dialog.querySelectorAll(".byo-preview li")).toHaveLength(3);
    // a bad sample says why (its code in words, never raw) instead of a thumbnail
    const { t } = await import("../src/i18n");
    expect(dialog.textContent).toContain(t("byo.sample_error.image_unreadable"));
    expect(dialog.textContent).not.toContain("image_unreadable");
    // a text column as the image: said under its select, and "Dùng cách ghép này" stays off
    fireEvent.change(image, { target: { value: "query" } });
    // the select is marked and read with its error (no alert while choosing)
    expect(image.getAttribute("aria-invalid")).toBe("true");
    expect(document.getElementById(image.getAttribute("aria-describedby") ?? "")?.className).toBe(
      "field-error",
    );
    expect(hints()).toBe(suggested - 1); // changed by the user: no longer "gợi ý"
    const apply = within(dialog).getByRole("button", { name: "Dùng cách ghép này" });
    expect(apply).toHaveProperty("disabled", true);
    fireEvent.change(image, { target: { value: "image" } });
    const { api } = await import("../src/api");
    const add = vi.spyOn(api, "addUserDataset");
    fireEvent.click(apply);
    // the mapping goes to the server (H5), which answers with the added dataset; the step names it by its three
    // params, set together
    await waitFor(() =>
      expect(add).toHaveBeenCalledWith({
        inspect_id: expect.any(String),
        columns: { image: "image", question: "query", answer: "ground_truth" },
      }),
    );
    await waitFor(() => {
      const p = useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params;
      expect(p).toMatchObject({
        user_dataset: "user--ds--acme--invoices-vi--7f0115a4--default--train",
        user_repo: "acme/invoices-vi",
        user_revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
      });
    });
    expect(screen.queryByRole("dialog")).toBeNull();
    // one undo step: all three back together
    act(() => useWorkflow.getState().undo());
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params).toMatchObject({
      user_dataset: "",
      user_repo: "",
      user_revision: "",
    });
  });

  it("a repo that only has a loading script is refused with the catalog's words", async () => {
    await setup();
    const input = screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/);
    fireEvent.change(input, { target: { value: "acme/needs-script" } });
    fireEvent.click(screen.getByRole("button", { name: "Kiểm tra" }));
    const err = await screen.findByRole("alert", {}, { timeout: 4000 });
    expect(err.getAttribute("data-code")).toBe("DATA-009");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("a refused mapping is said in the dialog, which stays open", async () => {
    await setup();
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    vi.spyOn(api, "addUserDataset").mockRejectedValue(
      new ApiError(400, {
        detail: {
          code: "DATA-010",
          params: { repo: "acme/x", missing: "image", available: "a, b" },
        },
      }),
    );
    const input = screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/);
    fireEvent.change(input, { target: { value: "acme/invoices-vi" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const dialog = await screen.findByRole("dialog", {}, { timeout: 4000 });
    fireEvent.click(within(dialog).getByRole("button", { name: "Dùng cách ghép này" }));
    const card = await within(dialog).findByRole("alert");
    expect(card.getAttribute("data-code")).toBe("DATA-010");
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("DATA-013 'Thêm dataset này' opens the hub tab with the repo and revision typed in", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("data.hf_load", { x: 0, y: 0 }) as string;
    const { Inspector } = await import("../src/components/Inspector");
    const { QuickFixes } = await import("../src/components/QuickFixes");
    const { tpOptional, t } = await import("../src/i18n");
    const label = tpOptional(["action", "add_user_dataset"]);
    if (!label) return; // fixcode's label not in this bundle yet: no button, never a key
    const rev = "7f0115a4b758a71d6473b8d085751692da2fef98";
    render(
      <>
        <QuickFixes
          nodeId={id}
          issue={{
            node_id: id,
            code: "DATA-013",
            severity: "error",
            params: { node: "Dataset", repo: "acme/invoices-vi", revision: rev },
            title: "…",
            what: "…",
            action_ids: ["add_user_dataset"],
            actions: [
              {
                id: "add_user_dataset",
                kind: "navigate",
                confirm: false,
                node_id: id,
                target: "add_dataset_dialog",
              },
            ],
          }}
        />
        <Inspector />
      </>,
    );
    act(() => fireEvent.click(screen.getByRole("button", { name: label })));
    const tab = await screen.findByRole("tab", { name: /Từ Hugging Face/ });
    expect(tab.getAttribute("aria-selected")).toBe("true");
    expect((screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/) as HTMLInputElement).value).toBe(
      "acme/invoices-vi",
    );
    expect((screen.getByLabelText(t("byo.revision")) as HTMLInputElement).value).toBe(rev);
  });

  it("a step that names a user dataset re-checks with that config/split (DATA-013 on another machine)", async () => {
    const { idPart, configSplitOf } = await import("../src/lib/hubJob");
    // the server's _id_part: lowercase, runs of other characters → "-", no dot/dash at the ends
    expect(idPart("Default")).toBe("default");
    expect(idPart("train_2024..v1")).toBe("train-2024.v1");
    expect(idPart("__")).toBe("x");
    expect(
      configSplitOf(
        "user--ds--a--b--7f0115a4--default--validation",
        ["default"],
        ["train", "validation"],
      ),
    ).toEqual({ config: "default", split: "validation" });
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("data.hf_load", { x: 0, y: 0 }) as string;
    useWorkflow.getState().setParams(id, {
      user_dataset: "user--ds--acme--invoices-vi--7f0115a4--default--validation",
      user_repo: "acme/invoices-vi",
      user_revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
    });
    useWorkflow.getState().select(id);
    const { api } = await import("../src/api");
    const inspect = vi.spyOn(api, "inspectDataset");
    const { Inspector } = await import("../src/components/Inspector");
    render(<Inspector />);
    const input = screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/);
    fireEvent.change(input, { target: { value: "acme/invoices-vi" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await screen.findByRole("dialog", {}, { timeout: 8000 });
    // first check takes the repo's default split; the step names "validation", so it checks that one
    expect(inspect).toHaveBeenCalledTimes(2);
    expect(inspect.mock.calls[1]?.[0]).toMatchObject({ split: "validation" });
  }, 15000);

  it("without a question column the preview shows the question every sample gets", async () => {
    await setup();
    const input = screen.getByPlaceholderText(/naver-clova-ix\/cord-v2/);
    fireEvent.change(input, { target: { value: "acme/invoices-vi" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const dialog = await screen.findByRole("dialog", {}, { timeout: 4000 });
    expect(dialog.querySelector(".byo-default-q")).toBeNull(); // a question column is mapped
    const q = within(dialog).getByLabelText(/Cột câu hỏi/) as HTMLSelectElement;
    fireEvent.change(q, { target: { value: "" } });
    const shown = dialog.querySelectorAll(".byo-default-q");
    expect(shown.length).toBeGreaterThan(0);
    expect(shown[0]?.textContent).toBe("Describe this image."); // the server's, never typed in the UI
  });
});
