import { expect, it } from "vitest";
import { mockNodeTypes } from "../mocks/api";

it("mock model.hf_pick mirrors the real catalog param (model + option_meta)", () => {
  const spec = mockNodeTypes.find((s) => s.type_id === "model.hf_pick");
  const p = spec?.params_schema.properties.model;
  expect(p?.default).toBe("nemotron-parse-v1.1");
  expect(p?.["x-ui"]?.option_meta?.["nemotron-parse-2.0"]?.tier).toBe("rnd");
});
