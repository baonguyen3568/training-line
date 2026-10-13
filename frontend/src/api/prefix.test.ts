import { afterEach, describe, expect, it } from "vitest";
import { appPrefix, uploadUrl } from "./client";

const at = (path: string) => window.history.replaceState(null, "", path);
afterEach(() => at("/"));

describe("app prefix (nginx, D-8)", () => {
  it("is empty at the site root and the mount path under a prefix", () => {
    at("/");
    expect(appPrefix()).toBe("");
    expect(uploadUrl("u1")).toBe("/api/uploads/u1");
    at("/app/?ui=d4");
    expect(appPrefix()).toBe("/app");
    expect(uploadUrl("u1")).toBe("/app/api/uploads/u1");
    at("/app/index.html");
    expect(appPrefix()).toBe("/app");
  });
});
