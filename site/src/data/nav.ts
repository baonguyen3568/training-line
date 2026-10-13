import { legalPages } from "./legal";

// Site chrome copy (vi), IA from design/m2/sitemap.md (M2-2). Page copy lives in the pages; bao-design owns wording.
// Only pages that exist are linked: check-links fails the build on a dead link.
export const nav = [
  { href: "/tinh-nang/", label: "Tính năng" },
  { href: "/template/", label: "Template" },
  { href: "/docs/bat-dau/", label: "Tài liệu" },
  { href: "/gioi-thieu/", label: "Giới thiệu" },
  { href: "/ma-nguon-mo/", label: "Mã nguồn" },
] as const;

export const installHref = "/docs/cai-dat/";

export const footer = [
  {
    title: "Sản phẩm",
    links: [
      { href: "/tinh-nang/", label: "Tính năng" },
      { href: "/template/", label: "Template" },
      { href: "/truong-hop-su-dung/", label: "Trường hợp sử dụng" },
      { href: "/lo-trinh/", label: "Lộ trình" },
      { href: "/thay-doi/", label: "Thay đổi" },
    ],
  },
  {
    title: "Tài liệu",
    links: [
      { href: "/docs/bat-dau/", label: "Bắt đầu" },
      { href: "/docs/cai-dat/", label: "Cài đặt" },
      { href: "/docs/faq-ky-thuat/", label: "FAQ" },
    ],
  },
  {
    title: "Dự án",
    links: [
      { href: "/gioi-thieu/", label: "Giới thiệu" },
      { href: "/ma-nguon-mo/", label: "Miễn phí & mã nguồn mở" },
      { href: "/lien-he/", label: "Liên hệ" },
    ],
  },
  {
    title: "Pháp lý",
    links: [
      ...legalPages.map((p) => ({ href: `/chinh-sach/${p.slug}/`, label: p.label })),
      { href: "/chinh-sach/ben-thu-ba/", label: "Phần mềm bên thứ ba" },
    ],
  },
] as const;

export { productName } from "./product.mjs";
