import { describe, expect, it } from "vitest";
import type { PortType } from "../api/types";
import { isCompatible, tokenVar } from "./ports";

const ports: PortType[] = [
  { id: "Model", parent: null, color_token: "color.port.model", label_key: "port.model" },
  {
    id: "QuantizedModel",
    parent: "Model",
    color_token: "color.port.model",
    label_key: "port.quantized_model",
  },
  { id: "Dataset", parent: null, color_token: "color.port.dataset", label_key: "port.dataset" },
];

describe("isCompatible", () => {
  it("accepts subtype into supertype, not the reverse", () => {
    expect(isCompatible("QuantizedModel", "Model", ports)).toBe(true);
    expect(isCompatible("Model", "QuantizedModel", ports)).toBe(false);
  });
  it("rejects unrelated types", () => {
    expect(isCompatible("Dataset", "Model", ports)).toBe(false);
  });
  it("maps tokens to CSS variables", () => {
    expect(tokenVar("color.port.model")).toBe("var(--color-port-model)");
  });
});
