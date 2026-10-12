import { expect, it } from "vitest";
import { expandQuery, fold, matches } from "./search";

it("folds Vietnamese accents", () => {
  expect(fold("Nén model Đo điểm")).toBe("nen model do diem");
});

it("matches synonyms: 'nén' finds a Quantize node, 'benchmark' finds 'Đo điểm'", () => {
  expect(matches("Nén model quantize.modelopt_ptq", expandQuery("nén"))).toBe(true);
  expect(matches("Đo điểm benchmark eval.benchmarks", expandQuery("chấm điểm"))).toBe(true);
  expect(matches("Chọn model gốc", expandQuery("lora"))).toBe(false);
});
