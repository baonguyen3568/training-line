# Release 1.0 — kế hoạch luồng D (Dashboard) và E (Site)
bao-design · 2026-10-13. Đồng dẫn với bao-frontend. Nguồn: BOARD "RELEASE 1.0", DESIGN.md, design/m2/sitemap.md.
Nguyên tắc chung: chỉ nói về tính năng đã chạy; số có nguồn hoặc "chưa có số đo"; giải thích phụ nằm sau "?".

## Luồng E — Site: sitemap chốt (30 trang)
Chi tiết, lý do và chủ trang nằm trong `design/m2/sitemap.md`, mục "Release 1.0 — bản chốt".

Thứ tự làm: (1) copy và dữ liệu trang còn thiếu → (2) minh hoạ → (3) vòng cuối, gồm impeccable `critique` + `audit` theo nhóm trang
(P: landing, tính năng, template; R: docs, chính sách) và `polish` một lượt.

Mỗi trang đạt DoD khi:
- [ ] `check_site_numbers.py --strict` xanh (mọi số có `sources:`)
- [ ] Thuật ngữ nằm trong `<Hint>`, thân bài không còn chú thích nhỏ
- [ ] Tiếng Việt có dấu đầy đủ; tiêu đề trả lời câu hỏi của người đọc
- [ ] a11y AA (contrast theo token, focus thấy được, heading đúng cấp); `prefers-reduced-motion` tắt animation
- [ ] Không ảnh hoặc câu hỏi benchmark; không logo NVIDIA; "Built on NVIDIA Cosmos" chỉ ở khối Cosmos và trang giấy phép
- [ ] Ảnh minh hoạ là ảnh chụp app thật (port 7878 hoặc app thật), dữ liệu CORD-v2 có ghi công, hoặc ảnh tự vẽ

## Luồng D — Dashboard
| # | Việc | Làm | Design giao | Điều kiện xong |
|---|---|---|---|---|
| D1 | D4 thành mặc định; giao diện cũ chỉ còn qua `?ui=legacy` tới khi F (QA) xanh rồi xoá | frontend | review ảnh 3 màn: canvas, gallery, cài đặt | E2E mọi template trên D4 |
| D2 | Subgraph bung/gập, lỗi bên trong group | frontend (chờ P016 của cleancode) | spec P4-4 + key `canvas.group.*` | design APPROVE ảnh chụp |
| D3 | G-3 "Sửa nhanh" trên thẻ lỗi | frontend + fixcode | nhãn action đã đủ 52/52 trong i18n; review vị trí nút | tester: mỗi action có test |
| D4 | G-4 "Đặt lại bước" | cleanup (API) + frontend | copy dialog dry-run + PERMANENT_LOSS khi có API | design APPROVE copy |
| D5 | Trạng thái "Chờ bộ nhớ trống" trên node | frontend | copy đã có (9273f3a) | review ảnh |
| D6 | Chạy thử (`run_smoke`) | frontend + backend | review vị trí trên toolbar | — |
| D7 | Icon-first: mọi nút toolbar/strip có tooltip, không chữ thừa | frontend | impeccable `audit` + `critique` trên 7878 | không còn finding P0/P1 |
| D8 | Copy cho node/lỗi mới của luồng B | người thêm node | mình viết/duyệt trong ngày | gate i18n xanh |

Quy tắc review: người làm gửi ảnh chụp (cdp-shot) hoặc URL 7878 kèm bước tái hiện. Mình trả lời APPROVE hoặc liệt kê việc cần sửa.
Chỉ chụp lại khi thay đổi về hình; thay đổi chỉ về copy thì chạy gate i18n là đủ.
