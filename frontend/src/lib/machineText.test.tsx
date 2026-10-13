import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { withCode } from "./machineText";

describe("withCode", () => {
  it("sets file names and paths as code, in sentence order", () => {
    const { container } = render(
      <p>
        {withCode('Bản xuất thiếu config.json, *.safetensors trong "outputs/x".', {
          files: "config.json, *.safetensors",
          node: "Xuất",
          path: "outputs/x",
        })}
      </p>,
    );
    expect([...container.querySelectorAll("code")].map((c) => c.textContent)).toEqual([
      "config.json, *.safetensors",
      "outputs/x",
    ]);
    expect(container.textContent).toBe(
      'Bản xuất thiếu config.json, *.safetensors trong "outputs/x".',
    );
  });

  it("leaves plain words alone", () => {
    expect(withCode("Bước Dạy model bị lỗi.", { node: "Dạy model" })).toBe(
      "Bước Dạy model bị lỗi.",
    );
  });
});
