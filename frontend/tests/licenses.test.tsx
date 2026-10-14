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
  it("a known id reads as design's name, a dotted id as one key", async () => {
    const { t } = await import("../src/i18n");
    expect(licenseName("apache-2.0")).toBe("Apache-2.0");
    expect(licenseName("llama3.1")).not.toBe("llama3.1"); // "license.name.llama3.1", not license.name.llama3 → 1
    expect(licenseName("lfm1.0")).toContain("LFM");
    expect(licenseName("no-such-licence")).toBe("no-such-licence");
    expect(t("license.badge.unknown")).not.toBe("license.badge.unknown");
  });
});

describe("Icon", () => {
  it("an unknown or prototype name renders nothing instead of throwing", async () => {
    const { render } = await import("@testing-library/react");
    const { Icon } = await import("../src/components/Icon");
    for (const name of ["constructor", "__proto__", "no-such-icon"]) {
      const { container, unmount } = render(<Icon name={name} />);
      expect(container.innerHTML).toBe("");
      unmount();
    }
  });
});
