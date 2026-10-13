import { legalPages } from "./legal";

// Site chrome copy (vi). Page copy lives in the pages themselves; bao-design owns the wording.
export const nav = [
  { href: "/tinh-nang/", label: "Tính năng" },
  { href: "/mau/", label: "Mẫu workflow" },
  { href: "/docs/bat-dau/", label: "Tài liệu" },
  { href: "/ma-nguon-mo/", label: "Miễn phí & mã nguồn mở" },
] as const;

export const footer = [
  ...legalPages.map((p) => ({ href: `/chinh-sach/${p.slug}/`, label: p.label })),
  { href: "/lien-he/", label: "Liên hệ" },
] as const;

export { productName } from "./product.mjs";
