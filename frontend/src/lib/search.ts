/** Accent-insensitive, case-insensitive text for search ("nén" matches "nen"; đ → d). */
export function fold(s: string): string {
  return s
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase();
}

// Small synonym table (P010 §5): users type either the Vietnamese task or the technical term.
const SYNONYMS: string[][] = [
  ["nen", "quantize", "fp8", "nvfp4", "int4", "awq", "nho"],
  ["do diem", "cham diem", "benchmark", "eval", "danh gia"],
  ["fine-tune", "finetune", "lora", "sft", "day model", "huan luyen", "train"],
  ["du lieu", "dataset", "data", "anh"],
  ["model", "mo hinh"],
  ["xuat", "export", "luu"],
  ["chay thu", "chat", "serve", "vllm"],
];

export function expandQuery(q: string): string[] {
  const f = fold(q.trim());
  if (!f) return [];
  const group = SYNONYMS.find((g) => g.some((w) => w.startsWith(f) || f.startsWith(w)));
  return group ? [f, ...group] : [f];
}

export function matches(haystack: string, terms: string[]): boolean {
  if (terms.length === 0) return true;
  const h = fold(haystack);
  return terms.some((t) => h.includes(t));
}
