// D-8 sign-in session: Settings shows where this browser signs in from, signs out here or everywhere (after a
// confirmation), and on the app's machine lifts the LAN lock. A socket closed with 4401 or an API 401 means
// "signed out", never "connection lost", and the app reloads once instead of looping.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

class FakeSocket {
  static last: FakeSocket | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  constructor(public url: string) {
    FakeSocket.last = this;
  }
  send() {}
  close() {}
}

describe("client", () => {
  it("a socket closed with 4401 reports the session ended, not a lost connection", async () => {
    vi.stubGlobal("WebSocket", FakeSocket);
    const { httpApi, onSessionEnded } = await import("../src/api/client");
    const ended = vi.fn();
    onSessionEnded(ended);
    const events: unknown[] = [];
    httpApi.watchDownload("d1", (ev) => events.push(ev));
    FakeSocket.last?.onclose?.({ code: 4401 });
    expect(ended).toHaveBeenCalledTimes(1);
    expect(events).toEqual([]);
    httpApi.watchRun("r1", (ev) => events.push(ev));
    FakeSocket.last?.onclose?.({ code: 1006 }); // a real drop is still "connection lost"
    expect(events).toEqual([{ type: "connection_lost" }]);
    onSessionEnded(null);
  });

  it("an API 401 unauthenticated reports the session ended once", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ code: "unauthenticated" }), {
            status: 401,
            headers: { "content-type": "application/json" },
          }),
      ),
    );
    const { httpApi, onSessionEnded } = await import("../src/api/client");
    const ended = vi.fn();
    onSessionEnded(ended);
    await expect(httpApi.runs()).rejects.toBeTruthy();
    await expect(httpApi.system()).rejects.toBeTruthy();
    expect(ended).toHaveBeenCalledTimes(1);
    onSessionEnded(null);
  });

  it("a reload that just happened shows the 'open the link' card instead of reloading again", async () => {
    const { handleSessionEnded } = await import("../src/lib/session");
    sessionStorage.setItem("vlm-session-reload-at", String(Date.now()));
    const show = vi.fn();
    handleSessionEnded(show);
    expect(show).toHaveBeenCalled();
    sessionStorage.removeItem("vlm-session-reload-at");
  });
});

describe("Settings › sign-in session", () => {
  it("shows the LAN lock on the local machine and lifts it", async () => {
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    render(<SettingsPanel />);
    expect(await screen.findByText("Bạn đang dùng app ngay trên máy chủ.")).toBeTruthy();
    expect(screen.getByTestId("lan-lock").textContent).toMatch(
      /bị khoá tới \d{2}:\d{2} vì đã nhập sai 5 lần/,
    );
    fireEvent.click(screen.getByRole("button", { name: "Mở khoá" }));
    expect(await screen.findByText("Đã mở khoá đăng nhập qua LAN.")).toBeTruthy();
    expect(screen.queryByTestId("lan-lock")).toBeNull();
  });

  it("signs out everywhere only after the confirmation; a LAN browser sees no lock", async () => {
    const { api } = await import("../src/api");
    vi.spyOn(api, "session").mockResolvedValue({ origin_class: "lan" });
    const revoke = vi.spyOn(api, "revokeAllSessions").mockRejectedValue(new Error("offline"));
    const { SettingsPanel } = await import("../src/components/SettingsPanel");
    render(<SettingsPanel />);
    expect(await screen.findByText(/qua mạng nội bộ \(LAN\)/)).toBeTruthy();
    expect(screen.queryByTestId("lan-lock")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Đăng xuất mọi thiết bị" }));
    expect(revoke).not.toHaveBeenCalled();
    expect(screen.getByRole("alertdialog").textContent).toContain("kể cả trình duyệt này");
    const buttons = screen.getAllByRole("button", { name: "Đăng xuất mọi thiết bị" });
    fireEvent.click(buttons[buttons.length - 1] as HTMLElement);
    await waitFor(() => expect(revoke).toHaveBeenCalled());
    expect(await screen.findByRole("alert")).toBeTruthy(); // the failure is said, not swallowed
  });
});
