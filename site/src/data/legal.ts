// Policy pages are IMPORTED from docs/legal/ (owner: bao-security, source of truth), never copied into site/.
// File stem in docs/legal -> URL /chinh-sach/<slug>/. Order = footer order.
export const legalPages = [
  { file: "chinh-sach-quyen-rieng-tu", slug: "quyen-rieng-tu", label: "Quyền riêng tư" },
  { file: "dieu-khoan-su-dung", slug: "dieu-khoan", label: "Điều khoản sử dụng" },
  { file: "chinh-sach-cookie", slug: "cookie", label: "Cookie" },
  { file: "bao-mat-va-bao-cao-lo-hong", slug: "bao-mat", label: "Bảo mật & báo cáo lỗ hổng" },
  { file: "giay-phep-va-ghi-cong", slug: "giay-phep", label: "Giấy phép & ghi công" },
] as const;

/** URL slug of a docs/legal file; an unmapped file fails the build instead of getting a guessed URL. */
export function legalSlug(file: string): string {
  const stem = file.replace(/\.md$/, "");
  const page = legalPages.find((p) => p.file === stem);
  if (!page) throw new Error(`docs/legal/${file}: add it to site/src/data/legal.ts`);
  return page.slug;
}
