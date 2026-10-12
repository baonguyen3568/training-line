# Review UI — S-5 vertical slice, vòng 1 (bao-design, 2026-10-12)

Cách xem: dev server `127.0.0.1:7875` với `VITE_MOCK_API=1`. Ảnh chụp bằng Chromium headless (driver CDP tự viết, 1440×900, theme sáng):
`design/reviews/s5-r1/gallery.png`, `design/reviews/s5-r1/canvas.png`.
Verdict: **CHANGES REQUESTED** — có 2 lỗi chặn (B1, B2), còn lại không chặn. Slice chạy được, phần khung rất tốt: số thứ tự bước, thanh kiểm tra, ước lượng có dải, header GPU/RAM theo MemTotal thật.

## Chặn
**B1. Nền làn che node — 4/7 làn không thấy node.**
Các node ở làn ①③⑤⑦ (Nguồn, Huấn luyện, Nén, Xuất) bị che, chỉ còn dây lộ ra; xem `canvas.png` ở x≈485–635.
- Nguyên nhân: `<ViewportPortal>` render *sau* lớp `.react-flow__nodes`. Làn lẻ có nền đục (`.lane:nth-child(odd)` dùng `canvas.lane_band`), nên phủ lên node trong làn đó.
- Sửa (chọn 1):
  - (a) đặt lớp portal của làn xuống dưới node, vd `.react-flow__viewport-portal { z-index: -1 }`, hoặc tách làn ra một layer riêng nằm dưới;
  - (b) vẽ làn bằng background tuỳ biến (SVG pattern trong `<Background>`).
- Nên thêm 1 test e2e: mọi node của template phải có `elementFromPoint(tâm node)` thuộc chính node đó.

**B2. Tương phản nút chính và link không đạt WCAG.**
- Nút `.primary` dùng chữ trắng trên `accent.focus`: chỉ 3.9:1 ở light (#5a8f00) và 2.4:1 ở dark (#76b900), cần ≥ 4.5:1.
- `.link` dùng chữ `accent.focus` trên nền surface: 3.8:1.
- Mình đã thêm token `color.action.*`:
  - `primary_bg #76b900`, `primary_fg #0b0b0b` (8.2:1, đúng quy ước NVIDIA là chữ đen trên nền xanh), `primary_bg_hover`;
  - `selected_bg`/`selected_fg` cho segmented `.on`;
  - `link` = text.primary + gạch chân.
- `accent.focus` chỉ dùng cho focus ring và viền selection, không dùng làm nền chữ.

## Không chặn (xếp theo ưu tiên)
1. **Chữ trên canvas quá nhỏ.** fitView với 7 làn × 300px kéo zoom xuống ~0.5, chữ node còn ~6px.
   Đề xuất `fitViewOptions={{ maxZoom: 1, minZoom: 0.75 }}` và cho cuộn ngang. Cách khác: thu `LANE_WIDTH` còn 260, vì node rộng 240.
   Người no-code cần đọc được ngay khi mở template, không phải zoom.
2. **Nhãn làn nằm giữa màn hình** (top 2010px trong lane cao 6000px). Nên dính ở mép trên viewport (sticky theo trục y), cỡ chữ ≥ 12px *sau* zoom.
3. **MiniMap là khối trắng trống** ở góc phải dưới. Hoặc style theo tokens (`nodeColor` theo `node_state`, nền `canvas.background`), hoặc ẩn ở v1. Graph 5–10 node chưa cần minimap.
4. **Panel phải chỉ có chữ "Thư viện bước", không có gì bên trong.** Khi chưa chọn node, panel nên hiện thư viện bước thật: nhóm theo làn, ô tìm, kéo vào canvas. Hoặc tạm thời hiện câu hướng dẫn "Chọn một bước để xem thiết lập". Panel trống trông như lỗi.
5. **Thẻ template** (P010 §8): thiếu sơ đồ thu nhỏ (thumbnail SVG từ graph như bạn đã định), thiếu 2 nút **[Chạy thử nhanh ~N phút] · [Dùng mẫu này]**. Key copy đã có: `app.run_smoke`, `gallery.use_template`.
6. **Chip "💾 Dung lượng 2000 GB"** dễ hiểu nhầm là tổng dung lượng. Nên ghi "còn trống 2000 GB", hoặc dùng `header.disk` + số kèm chữ "trống". Mình sẽ thêm key `header.disk_free`.
7. **Dòng "Dự kiến…" ở thanh kiểm tra**: tốt. Khi `source=declared/unknown` nhớ áp style ở P010 §6.
8. `error.SYS-001.title` đang dùng sai nghĩa cho lỗi "không tải được catalog" (SYS-001 là "máy đang bận"). Đã thêm `app.catalog_load_failed` + `app.retry`.

## Key đã thêm cho slice (i18n 407 key, vi/en vẫn khớp)
- `template.lora_finetune|quantize_bench.title|desc`
- `node.<type>.param.<name>.label|help` cho repo, path, val_fraction, preset, epochs, learning_rate, format, groups
- `node.<type>.port.<name>` cho mọi port trong mock
- `port.checkpoint`: mock có kiểu `Checkpoint` (export.merge_lora). Kiểu này không có trong 8 kiểu của 009 → cần **backend xác nhận** là kiểu riêng hay là `Model`/`ExportArtifact`
- `app.catalog_load_failed`, `app.retry`

Nhãn option của enum: đề xuất NodeSpec `x-ui.option_keys: "train_preset"` (prefix copy dùng chung), để UI tra `train_preset.<value>` mà không hard-code theo tên param.

## Vòng sau
Sau khi sửa B1/B2 và mục 1–4, mình chụp lại cả light/dark, rồi đi qua 5 nhiệm vụ của `design/research/usability-test-plan.md` trên slice trước khi báo PM.
