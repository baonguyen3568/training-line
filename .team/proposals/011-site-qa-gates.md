# Proposal 011 — QA gates cho site SaaS (M2-7)

Tác giả: bao-tester · 2026-10-13 · Trạng thái: DRAFT, chờ frontend / cleancode / security / PM review

## Mục tiêu
Site (Astro 7.3 + Starlight 0.42, `site/`) chỉ được publish khi qua 6 gate. Mọi gate chạy offline trên bản build tĩnh
(`site/dist`), không gọi ra mạng, không cần GPU. Gate rẻ chạy trong `npm run check`; gate nặng (visual, Lighthouse)
chạy trong `scripts/check.sh --site` trước khi merge.

| # | Gate | Công cụ | Ngưỡng fail | Trạng thái |
|---|---|---|---|---|
| G1 | Không bịa số | `scripts/qa/check_site_numbers.py` (tester) + `check-sources` (cleancode: path trong `sources` phải tồn tại) | Mọi con số trong prose phải xuất hiện trong ít nhất 1 file `sources` của trang; whitelist chỉ qua frontmatter `numbers_without_source` | **XONG (report-only)**; bật `--strict` khi đủ trang |
| G2 | Link | crawler trên `dist/` (`site/scripts/check-links.mjs`, đã có trong `npm run check`, file chưa có) | Link nội bộ 404, anchor `#id` không tồn tại, link ra ngoài không nằm trong allowlist | cần thống nhất ai viết (đề xuất: tester viết, frontend review) |
| G3 | a11y WCAG 2.2 AA | Playwright + `@axe-core/playwright` (tags wcag2a, wcag2aa, wcag21aa, wcag22aa), mọi trang | Có vi phạm `serious`/`critical`; `moderate` thì báo cáo | cần dep mới, chờ security duyệt (ADR-004) |
| G4 | Visual regression | Playwright `toHaveScreenshot`, 3 viewport 375×812 / 768×1024 / 1440×900, light + dark, font và ảnh tự host | `maxDiffPixelRatio` 0.01; snapshot chụp trên đúng máy GB10 (aarch64), không trộn với snapshot chụp ở máy khác | dùng `@playwright/test` 1.63.0 sẵn có của frontend |
| G5 | Lighthouse | `lighthouse` CLI + Chromium của Playwright, preset desktop + mobile, 5 trang đại diện | Performance ≥ 90, Accessibility = 100, Best Practices ≥ 95, SEO ≥ 90 (median của 3 lần chạy) | dep nặng, chờ security; nếu không duyệt thì đo bằng Playwright + web-vitals |
| G6 | prefers-reduced-motion | Playwright `emulateMedia({reducedMotion: "reduce"})` | Khi reduce: không còn animation/transition nào > 0.01s (đọc `getAnimations()`), không autoplay video; nội dung vẫn hiện đầy đủ (không bị kẹt `opacity: 0`) | dùng dep sẵn có |

## G1 chi tiết (đã làm)
- Lấy số trong prose: bỏ frontmatter, import, code, đích của link, thuộc tính tag, mã lỗi (`CFG-010`).
  Chuẩn hoá: `0,94` → `0.94`; dấu tách nghìn (`1.024` → `1024`); `42.0` trong JSON khớp `42`.
  Số 0–10 bỏ qua (số thứ tự, "2 bước").
- Giới hạn đã biết: chỉ khớp chuỗi số, không khớp ngữ nghĩa. Nếu file nguồn là `models.yaml` dài thì một số bịa có
  thể trùng ngẫu nhiên với số trong file. Vì vậy cần thêm review của người; script chỉ là lớp chặn đầu tiên.
  Quy ước: con số kết quả đo lấy từ `benchmarks/`, nên liệt kê đúng file benchmark cụ thể, không liệt kê cả thư mục.
- Trạng thái 2026-10-13: 5 trang, 0 vấn đề. Lượt chạy đầu báo 4 cảnh báo, cả 4 đều là báo nhầm do phần chuẩn hoá:
  "128 GB" (nguồn viết liền "128GB"), "42" (nguồn ghi "42.0"), "95" (nguồn ghi "CI95") và "010" (thuộc mã lỗi
  CFG-010). Đã sửa phần chuẩn hoá. 6 unit test chứng minh script vẫn bắt được số bịa.

## Cần từ đồng đội
- **bao-frontend**: tên route cố định và `data-testid` cho các vùng có animation; trang nào có chuyển động; cấu trúc
  `dist/`. Ai viết `check-links.mjs`.
- **bao-cleancode**: `check-sources` nằm ở đâu; gộp G1 vào cùng bước để chỉ chạy một lần.
- **bao-security**: duyệt dep `@axe-core/playwright` (MPL-2.0) và `lighthouse` (Apache-2.0), cài bản pin vào
  devDependencies của `site/`.
- **bao-design**: chốt viewport và chế độ màu cho visual regression; nội dung động cần che khi chụp (ngày tháng, số
  thay đổi theo lần chạy).
