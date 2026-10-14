// lib/licenses: a licence id is data from a repo card; names come from copy or the label table, never from the
// object prototype ("constructor", "__proto__" read as themselves).
import { describe, expect, it } from "vitest";
import { licenseName } from "../src/lib/licenses";

describe("licenseName", () => {
  it.each(["constructor", "__proto__", "toString", "hasOwnProperty"])(
    "%s reads as itself",
    (id) => {
      expect(licenseName(id)).toBe(id);
    },
  );
  it("a known id reads as its name", () => {
    expect(licenseName("apache-2.0")).toBe("Apache-2.0");
  });
});
