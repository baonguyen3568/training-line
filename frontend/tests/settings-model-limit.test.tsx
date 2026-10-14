// ADR-021 A1: the size cap of one user-added model is set in Settings (1–1000 GB, default 40); a value out of the
// server's range is said with that range and never sent; NET-004 "raise the limit" opens Settings on that input.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

describe("model size cap", () => {
  it("checks the range before saving, then saves on 'Lưu'", async () => {
    const { api } = await import("../src/api");
    const put = vi.spyOn(api, "setUserAssetLimits");
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const { t } = await import("../src/i18n");
    render(<SettingsPanel />);
    const input = (await screen.findByLabelText(
      t("settings.row_max_model_gb"),
    )) as HTMLInputElement;
    expect(input.value).toBe("40");
    const section = input.closest("section") as HTMLElement;
    const save = () =>
      Array.from(section.querySelectorAll("button")).find(
        (b) => b.textContent === t("settings.save"),
      ) as HTMLButtonElement;
    fireEvent.change(input, { target: { value: "2000" } });
    fireEvent.click(save());
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      t("settings.max_model_gb_range", { min: 1, max: 1000 }),
    );
    expect(put).not.toHaveBeenCalled(); // out of range: never sent
    fireEvent.change(input, { target: { value: "60" } });
    fireEvent.click(save());
    await waitFor(() => expect(put).toHaveBeenCalledWith({ max_model_gb: 60 }));
    await waitFor(() => expect(save().disabled).toBe(true)); // saved: nothing left to save
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("NET-004 'raise the limit' opens Settings, as a quick fix and on the hub panel's error card", async () => {
    const seen: string[] = [];
    const on = (e: Event) => seen.push((e as CustomEvent<string>).detail);
    window.addEventListener("vlm:navigate", on);
    const { t } = await import("../src/i18n");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const { QuickFixes } = await import("../src/components/QuickFixes");
    const { unmount } = render(
      <QuickFixes
        nodeId={id}
        issue={{
          node_id: id,
          code: "NET-004",
          severity: "error",
          params: { size_gb: 72, quota_gb: 40, reason: "model_cap" },
          title: "…",
          what: "…",
          action_ids: ["raise_model_limit"],
          actions: [
            {
              id: "raise_model_limit",
              kind: "navigate",
              confirm: false,
              node_id: id,
              target: "settings_limits",
            },
          ],
        }}
      />,
    );
    act(() => fireEvent.click(screen.getByRole("button", { name: t("action.raise_model_limit") })));
    expect(seen).toEqual(["settings"]);
    unmount();
    // adding a model above the cap: the job's NET-004 card carries the same button
    useWorkflow.getState().select(id);
    const { Inspector } = await import("../src/components/Inspector");
    render(<Inspector />);
    fireEvent.click(screen.getByRole("tab", { name: /Từ Hugging Face/ }));
    const input = screen.getByPlaceholderText(/Qwen\/Qwen3.5-2B/);
    fireEvent.change(input, { target: { value: "someone/huge-vlm" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const card = await screen.findByRole("alert", {}, { timeout: 4000 });
    expect(card.getAttribute("data-code")).toBe("NET-004");
    act(() =>
      fireEvent.click(within(card).getByRole("button", { name: t("action.raise_model_limit") })),
    );
    expect(seen).toEqual(["settings", "settings"]);
    window.removeEventListener("vlm:navigate", on);
  });
});
