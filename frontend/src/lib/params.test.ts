import { expect, it } from "vitest";
import { mockNodeTypes } from "../mocks/api";
import { customizedAdvanced } from "./params";

it("lists only advanced params changed from their default", () => {
  const lora = mockNodeTypes.find((s) => s.type_id === "train.lora_vlm");
  expect(customizedAdvanced(lora, { epochs: 5 })).toEqual([]); // epochs is not advanced
  expect(customizedAdvanced(lora, { learning_rate: 0.0001 })).toEqual([]); // equals default
  expect(customizedAdvanced(lora, { learning_rate: 0.01 })).toEqual(["learning_rate"]);
});
