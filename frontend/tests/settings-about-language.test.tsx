// A5 Settings: "Ứng dụng" shows the version + commit and copies exactly the bug report's version and GPU fields,
// and switches language only once the open workflow is safe as a restorable draft; "Truy cập từ máy khác" gives the
// LAN addresses and the CA with its fingerprint, on the app's own machine only.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => {
  vi.restoreAllMocks();
  try {
    localStorage.clear();
  } catch {
    /* jsdom */
  }
});

describe("Ứng dụng › Phiên bản", () => {
  it("shows version and the short commit, and copies only version + GPU", async () => {
    const writeText = vi.fn(async (_text: string) => {});
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    render(<SettingsPanel />);
    const heading = await screen.findByRole("heading", { name: "Ứng dụng" });
    const section = heading.closest("section") as HTMLElement;
    await waitFor(() => expect(section.querySelector("code")?.textContent).toBe("4b04214c0ffe"));
    expect(section.textContent).toContain("1.0.0.dev0 (4b04214c0ffe)");
    expect(section.textContent).toContain("Ngôn ngữ"); // both rows in one section
    fireEvent.click(screen.getByRole("button", { name: /Sao chép thông tin phiên bản/ }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    const copied = writeText.mock.calls[0]?.[0] ?? "";
    expect(copied).toContain("1.0.0.dev0+4b04214c0ffe");
    expect(copied).toContain("GB10");
    expect(copied.split("\n")).toHaveLength(2); // nothing beyond the bug report's two fields
  });
});

describe("Ngôn ngữ", () => {
  it("names each language in itself and marks the current one", async () => {
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    render(<SettingsPanel />);
    const group = await screen.findByRole("radiogroup");
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["Tiếng Việt", "English"]);
    expect(radios[0]?.getAttribute("aria-checked")).toBe("true");
    expect(radios[1]?.getAttribute("lang")).toBe("en");
    expect(group).toBeTruthy();
  });

  it("keeps an open workflow as a draft before switching, and refuses when it cannot", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 });
    const wf = useWorkflow.getState().workflow;
    const { switchLocale } = await import("../src/lib/localePref");
    const { loadDraft } = await import("../src/lib/draft");
    // storage refuses writes: the draft would be lost, so the language does not change
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota");
    });
    expect(switchLocale("en", wf)).toBe(false);
    setItem.mockRestore();
    // storage works: the draft is saved and restorable, the choice is stored, then the page reloads
    const errors = vi.spyOn(console, "error").mockImplementation(() => {}); // jsdom: navigation not implemented
    expect(switchLocale("en", wf)).toBe(true);
    expect(loadDraft()?.workflow.nodes).toHaveLength(1);
    expect(localStorage.getItem("vlm.locale.v1")).toBe("en");
    errors.mockRestore();
  });

  it("applies the stored language before the app renders", async () => {
    localStorage.setItem("vlm.locale.v1", "en");
    const { applyStoredLocale } = await import("../src/lib/localePref");
    const { getLocale, setLocale } = await import("../src/i18n");
    applyStoredLocale();
    expect(getLocale()).toBe("en");
    expect(document.documentElement.lang).toBe("en");
    setLocale("vi"); // the rest of the suite runs in Vietnamese
    localStorage.setItem("vlm.locale.v1", "nope");
    applyStoredLocale();
    expect(getLocale()).toBe("vi"); // an unknown value is ignored
  });
});

describe("Truy cập từ máy khác", () => {
  it("lists each address and the CA with its fingerprint, the download going through the app prefix", async () => {
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    render(<SettingsPanel />);
    const heading = await screen.findByRole("heading", { name: "Truy cập từ máy khác" });
    const section = heading.closest("section") as HTMLElement;
    await waitFor(() => expect(section.textContent).toContain("https://192.168.2.178:4000/"));
    const dl = screen.getByRole("link", { name: /Tải chứng chỉ/ });
    expect(dl.getAttribute("href")).toMatch(/\/api\/lan\/ca\.crt$/);
    expect(dl.hasAttribute("download")).toBe(true);
    const fp = section.querySelector(".set-fingerprint code")?.textContent ?? "";
    expect(fp).toMatch(/^([0-9A-F]{2}:){31}[0-9A-F]{2}$/);
    expect(section.textContent).toContain("so vân tay SHA-256");
    expect(screen.getByRole("button", { name: "Sao chép vân tay" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sao chép địa chỉ" })).toBeTruthy();
    // order: Kết nối, Phiên đăng nhập, Truy cập từ máy khác, Ứng dụng, Do quản trị viên đặt
    const order = [...document.querySelectorAll("section h2")].map((h) => h.textContent?.trim());
    expect(order).toEqual([
      "Kết nối",
      "Phiên đăng nhập",
      "Truy cập từ máy khác",
      "Ứng dụng",
      "Do quản trị viên đặt",
    ]);
  });

  it("says LAN is off, says when the CA is missing, and is absent for a LAN browser", async () => {
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    const lan = vi.spyOn(api, "lan").mockResolvedValue({ enabled: false, urls: [], ca: null });
    const a = render(<SettingsPanel />);
    expect(await screen.findByText(/Đang tắt\. Quản trị viên bật/)).toBeTruthy();
    a.unmount();
    lan.mockResolvedValue({ enabled: true, urls: ["https://192.168.2.178:4000/"], ca: null });
    const b = render(<SettingsPanel />);
    expect(await screen.findByText(/Chưa có chứng chỉ cho truy cập qua LAN/)).toBeTruthy();
    expect(screen.queryByRole("link", { name: /Tải chứng chỉ/ })).toBeNull();
    b.unmount();
    lan.mockRejectedValue(new ApiError(403, { code: "local_only" }));
    render(<SettingsPanel />);
    await screen.findByRole("heading", { name: "Ứng dụng" });
    expect(screen.queryByRole("heading", { name: "Truy cập từ máy khác" })).toBeNull();
  });
});
