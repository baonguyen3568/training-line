import { afterEach, describe, expect, it, vi } from "vitest";
import { httpApi, resetCsrfForTests } from "./client";
import type { Workflow } from "./types";

const wf: Workflow = { format: "vlm-workflow", schema_version: 1, name: "x", nodes: [], edges: [] };
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("CSRF handling (P003 D13)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    resetCsrfForTests();
    // biome-ignore lint/suspicious/noDocumentCookie: test sets a hostile cookie on purpose.
    document.cookie = "vlm_csrf=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
  });

  it("takes the token from /api/session and ignores any readable cookie", async () => {
    // biome-ignore lint/suspicious/noDocumentCookie: simulate cookie tossing by another local app.
    document.cookie = "vlm_csrf=attacker";
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(json({ csrf_token: "tok-1" }))
      .mockResolvedValueOnce(json({ ok: true, errors: [], estimate: {} }));
    await httpApi.validate(wf, "vi");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/session");
    const init = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-VLM-CSRF"]).toBe("tok-1");
  });

  it("refreshes the token once after a server restart (403 csrf_invalid)", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(json({ csrf_token: "old" }))
      .mockResolvedValueOnce(json({ code: "csrf_invalid" }, 403))
      .mockResolvedValueOnce(json({ csrf_token: "new" }))
      .mockResolvedValueOnce(json({ ok: true, errors: [], estimate: {} }));
    await httpApi.validate(wf, "vi");
    const last = fetchMock.mock.calls[3]?.[1] as RequestInit;
    expect((last.headers as Record<string, string>)["X-VLM-CSRF"]).toBe("new");
  });
});
