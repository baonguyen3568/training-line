// D-7 settings: a secret is write-only (input empty, cleared after save, only "…last4" shown); policy is read-only.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

describe("SettingsPanel", () => {
  it("never shows the token back: the field clears and only the last 4 characters appear", async () => {
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const { container } = render(<SettingsPanel />);
    const input = (await screen.findByLabelText("Token Hugging Face")) as HTMLInputElement;
    expect(input.type).toBe("password");
    expect(input.value).toBe("");
    // fake token built at run time, never a literal token shape in the repo (security, as test_scan_tokens.py)
    const fake = ["hf", "fakevalue1234"].join("_");
    fireEvent.change(input, { target: { value: fake } });
    fireEvent.click(screen.getAllByRole("button", { name: "Lưu" })[0] as HTMLElement);
    expect(input.value).toBe(""); // cleared before the response even arrives
    await waitFor(() => expect(screen.getByText(/Đã đặt …1234/)).toBeTruthy());
    expect(container.textContent).not.toContain(fake.slice(0, -4));
  });

  it("explains server errors and shows policy read-only, with the broken-key card when key_ok is false", async () => {
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    const real = await api.settings();
    vi.spyOn(api, "settings").mockResolvedValue({ ...real, key_ok: false });
    vi.spyOn(api, "putSecret").mockRejectedValue(
      new ApiError(422, {
        error: {
          code: "settings_invalid",
          params: { key: "NGC_API_KEY", reason: "not_printable" },
        },
      }),
    );
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const { container } = render(<SettingsPanel />);
    await screen.findByText("Không dùng được khoá mã hoá của app");
    const input = screen.getByLabelText("Khoá NGC");
    fireEvent.change(input, { target: { value: " x " } });
    fireEvent.click(screen.getAllByRole("button", { name: "Lưu" })[1] as HTMLElement);
    await screen.findByText(/dán thừa xuống dòng/);
    expect(screen.getAllByText("Mặc định của app").length).toBeGreaterThan(0); // source "default": no fake value
    expect(container.querySelectorAll(".set-row.locked input").length).toBe(0);
  });
});

describe("SettingsPanel password managers", () => {
  it("asks the browser and password managers not to store the token", async () => {
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const { container } = render(<SettingsPanel />);
    const input = (await screen.findByLabelText("Token Hugging Face")) as HTMLInputElement;
    expect(input.getAttribute("autocomplete")).toBe("new-password");
    expect(input.hasAttribute("data-1p-ignore")).toBe(true);
    expect(input.getAttribute("data-lpignore")).toBe("true");
    expect(container.querySelector("form")).toBeNull();
  });

  it("shows a policy warning under its row only when the app has copy for it (never the raw code)", async () => {
    const { api } = await import("../src/api");
    const real = await api.settings();
    const pol = real.admin_policy["storage.approval_policy"];
    if (!pol) throw new Error("no approval policy in the mock");
    vi.spyOn(api, "settings").mockResolvedValue({
      ...real,
      admin_policy: {
        ...real.admin_policy,
        "storage.approval_policy": { ...pol, warning: "NO_SUCH_CODE" },
      },
    });
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const { container } = render(<SettingsPanel />);
    await screen.findByRole("heading", { level: 1 });
    expect(container.querySelector(".set-warning")).toBeNull();
    expect(container.textContent).not.toContain("NO_SUCH_CODE");
  });
});
