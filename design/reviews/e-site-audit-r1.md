# Luồng E — audit a11y site, vòng 1
bao-design · 2026-10-13. Chạy probe `design/reviews/d7-probe/probe.js` qua CDP trên bản build hiện tại (preview 7877) cho 10 trang:
/, /tinh-nang/, /template/, /lo-trinh/, /lien-he/, /docs/bat-dau/, /docs/ma-loi/, /chinh-sach/giay-phep/, /thay-doi/ và 404.

## Kết quả
- **Tên truy cập và ngôn ngữ:** cả 10 trang đều có `lang="vi"`, đúng 1 `h1`, có `main`, có `nav`. Không có control nào thiếu tên truy cập.
- **Contrast:** 9/10 trang không có lỗi. Trang chủ có 2 chỗ: ký tự "✕" (`span.si.bad`) trong EditorPreview chỉ đạt 3.62:1 và 3.90:1 ở cỡ 12–13px.
  - Ký tự này đang đóng vai icon trạng thái. Đo như chữ thường thì trượt AA (cần 4.5:1); nếu đổi thành icon thì chỉ cần 3:1 (WCAG 1.4.11).
  - Sửa (P2): thay ✕/✓ bằng icon Lucide `x` / `check` (aria-hidden), dùng màu `status_ink`. Cách này cũng khớp luật không dùng glyph làm icon đã áp cho app.
- **Vùng bấm:** link ở footer chỉ cao 18px, xếp dọc.
  - Mức này đạt WCAG 2.5.8 nhờ ngoại lệ khoảng cách, nhưng khó bấm trên màn cảm ứng.
  - Sửa (P3): `padding-block: 3px` cho link footer, để vùng bấm đạt ≥24px.
  - Link skip-to-content cao 23px: được chấp nhận.

## Chưa kiểm
- Viewport điện thoại (390px).
- Lighthouse (máy không có Chrome đủ sandbox).
- Theme sáng của site.
- Trình đọc màn hình thật.
Tester (M2-7) giữ phần visual regression 3 viewport và Lighthouse.
