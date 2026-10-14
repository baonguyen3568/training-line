// ADR-021 / spec H §4: a dataset from Hugging Face is inspected, its columns mapped in a dialog with the server's
// suggestions and a preview, a wrong column kind is said under its select, and the mapping lands in the step.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

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
    expect(within(dialog).getByRole("alert")).toBeTruthy();
    expect(hints()).toBe(suggested - 1); // changed by the user: no longer "gợi ý"
    const apply = within(dialog).getByRole("button", { name: "Dùng cách ghép này" });
    expect(apply).toHaveProperty("disabled", true);
    fireEvent.change(image, { target: { value: "image" } });
    fireEvent.click(apply);
    await waitFor(() => {
      const p = useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params;
      expect(p).toMatchObject({
        repo: "acme/invoices-vi",
        revision: "7f0115a4b758a71d6473b8d085751692da2fef98",
        split: "train",
        columns: { image: "image", question: "query", answer: "ground_truth" },
      });
    });
    // the whole mapping is one undo step: one Ctrl+Z puts the step back as it was
    act(() => useWorkflow.getState().undo());
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params.repo).not.toBe(
      "acme/invoices-vi",
    );
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
});
