# D7 — Audit dashboard D4, vòng 1 (impeccable audit + phần critique ngắn)
bao-design · 2026-10-13 · đối tượng: frontend sau D1 (8b5570c + c19db2e), mock 7878, 1440×900, dark.

Bằng chứng gồm 3 nguồn:
- Detector `impeccable detect frontend/src`. Detector URL không chạy được vì máy thiếu sandbox Chrome.
- Probe runtime qua CDP (headless_shell) trên 3 màn: thư viện, canvas "Đọc hoá đơn", cài đặt. Script lưu ở `design/reviews/d7-probe/` để chạy lại.
- Đọc CSS/TSX.

**Chưa kiểm:**
- Viewport hẹp và cảm ứng.
- Light theme: D4 bị ép dark.
- Hiệu năng runtime (profiling).
- Trình đọc màn hình thật.

## Điểm
| # | Mảng | Điểm | Ý chính |
|---|---|---|---|
| 1 | Accessibility | 3 | Không có lỗi contrast chữ, mọi control đều có tên. Nhưng **focus bàn phím trên node bị mất** (P1). |
| 2 | Performance | 3 | Bundle chính ~483 KB chưa nén, chấp nhận được với app chạy local; có `prefers-reduced-motion` ở 4 chỗ. Chưa profile. |
| 3 | Theming | 4 | CSS dùng token gần như toàn bộ (~490 `var(--…)`); chỉ còn 3 giá trị hex lẻ. |
| 4 | Responsive | 2 | Thiết kế cho desktop là có chủ đích (Operate, ≥1280). Nhưng dưới ngưỡng đó chưa có màn hình thay thế, và chưa kiểm. |
| 5 | Implementation integrity | 3 | Detector chỉ bắt được kiểu viền dày một bên ("side-tab") ở thẻ lỗi; phần còn lại nhất quán với DESIGN.md. |
| **Tổng** | | **15/20** | **Good** |

**Verdict tính nhất quán: PASS.** Hệ thống có bản sắc riêng: hình cổng + nhãn, status ink, icon-first, inspector có nhãn, popover "?". Tóm tắt số lỗi: 0 P0, 2 P1, 4 P2, 3 P3.

## Phát hiện

**[P1] Node trên canvas không hiện viền focus khi dùng bàn phím**
- Vị trí: `@xyflow/react/dist/style.css:450` đặt `.react-flow__node.selectable:focus-visible { outline: none }`, mà app không thêm style focus nào thay thế.
- Probe: cả 5 node có tabindex nhưng không có outline hay box-shadow khi focus. Viền dây (edge) chỉ đổi màu stroke.
- Tác động: người dùng bàn phím không biết node nào đang được focus. Vi phạm WCAG 2.4.7 Focus Visible (AA).
- Sửa: `.react-flow__node:focus-visible .vlm-node { outline: 2px solid var(--color-accent-focus); outline-offset: 2px; }`. Với edge, tăng stroke-width khi `:focus-visible`, không chỉ đổi màu (luật Never-Colour-Alone).

**[P1] Ô tìm pipeline mất viền focus**
- Vị trí: `d4.css:562` đặt `.rp-search input { outline: none }`, mà wrapper không có `:focus-within`.
- Tác động: giống P1 ở trên (WCAG 2.4.7).
- Sửa: `.rp-search:focus-within { outline: 2px solid var(--color-accent-focus); outline-offset: 2px; }`.

**[P2] Thẻ lỗi dùng viền trái 3px ("side-tab")**
- Vị trí: `app.css:709` (`.toolbar-error`) và `app.css:841` (`.error-card`).
- Tác động: đây là dấu hiệu UI rập khuôn mà detector bắt được. Nó cũng lệch luật Hairline-First, và chỉ dựa vào màu để báo mức độ.
- Sửa: viền 1px `border.subtle` bao quanh cả thẻ, cộng icon `triangle-alert` (`status_ink.critical`) đặt ở đầu tiêu đề. Màu nằm ở icon, chữ giữ text token.

**[P2] Màn canvas không có landmark `main` và không có `h1`**
- Probe: canvas có `main=0`, chỉ có `h2` "Đọc hoá đơn → JSON". Các màn thư viện và cài đặt thì có cả hai.
- Sửa: bọc vùng editor bằng `<main>`; tên workflow dùng `h1`, giữ nguyên style hiện tại.

**[P2] Nút "?" điều khoản Hugging Face chỉ rộng 14×20 px**
- Vị trí: thẻ Cosmos trong thư viện, `aria-label` "Cần tự đồng ý điều kiện…".
- Tác động: vùng bấm quá nhỏ và dính vào chữ bên cạnh, không đạt WCAG 2.5.8 (24×24, ngoại lệ về khoảng cách cũng không áp dụng được).
- Sửa: vùng bấm tối thiểu 24×24. Nếu muốn giữ icon 14px thì dùng padding hoặc pseudo-element để nới vùng bấm.

**[P2] Thời gian dự kiến không khớp giữa thư viện và canvas**
- Thẻ "Đọc hoá đơn" ở thư viện ghi "~1 giờ 56 phút · RAM đỉnh 23.7 GB". Mở cùng template ra canvas thì inspector (WorkflowSummary) lại ghi "chưa có số đo" cho cả 3 dòng.
- Tác động: người dùng thấy hai con số khác nhau cho cùng một workflow và mất tin vào số đo.
- Sửa: cả hai chỗ dùng chung một thứ tự nguồn: số đo trong data file → ước lượng của server → "chưa có số đo". Nếu inspector cố ý chỉ hiện số đo trên chính máy này thì nhãn phải nói rõ điều đó (mình sẽ viết copy).

**[P3] Nút "?" 20×20 px**
- Đạt WCAG 2.5.8 nhờ ngoại lệ về khoảng cách, vì không có control nào khác nằm trong vòng 24px.
- Nếu tiện thì nới vùng bấm lên 24px cho đồng bộ với P2 ở trên.

**[P3] Còn màu hex cứng**
- Vị trí: `ShellD4.tsx:184` và `ToolbarD4.tsx:46` (`stroke="#fff"` của logo); `d4.css:233` (`box-shadow … #000`).
- Sửa: chuyển sang token tương ứng (`text.on-accent`, `shadow`), hoặc `currentColor` cho logo.

**[P3] Chưa có màn hình cho viewport hẹp**
- Màn dưới 1024px hiện chưa được thiết kế, cũng chưa kiểm.
- Đề xuất: hiện một màn chiếm cả trang với câu "Màn hình này hẹp quá để chỉnh workflow. Hãy mở trên máy tính." Bảng Dung lượng và trang Cài đặt vẫn dùng được. Mình sẽ viết key i18n khi bạn làm.

**False positive đã loại:**
- `broken-image` trong `workflowFile.test.ts`: là fixture test.
- 4 nút toolbar Hoàn tác / Làm lại / Nhân bản / Xoá "không có viền focus": các nút này đang disabled nên không nhận focus.
- Attribution React Flow 10px: đã chốt giữ.

## Điểm tốt nên giữ
- **Contrast:** 0 lỗi contrast chữ trên 3 màn (probe tính alpha + nền chồng lớp).
- **Tên truy cập:** 78/78 control đều có tên. Trang có `lang="vi"`.
- **Focus toàn cục:** có `:focus-visible` toàn cục dùng `accent.focus`.
- **Chuyển động:** có `prefers-reduced-motion` cho flash node, dock và chuyển cảnh.
- **Token:** dùng token rất đều.
- **Thư viện:** heading hợp lý (1 `h1` + `h2` cho từng thẻ).

## Việc tiếp theo (theo thứ tự)
1. **[P1] polish:** viền focus cho node, edge và ô tìm kiếm.
2. **[P2] harden:** đồng bộ nguồn số đo giữa thư viện và inspector; nới vùng bấm nút "?" điều khoản.
3. **[P2] quieter:** bỏ viền dày một bên ở thẻ lỗi, chuyển sang hairline + icon.
4. **[P2] harden:** thêm landmark `main` và `h1` cho canvas.
5. **[P3]:** màn viewport hẹp, chuyển màu hex sang token.

Nghiệm thu: sau khi sửa, mình chạy lại cùng probe. Điều kiện: 0 P1, đủ landmark, và mọi node hiện viền focus khi nhấn Tab.

## Vòng 2 (nghiệm thu f39ffab, cùng probe)
- **P1 node focus:** ĐẠT. `.vlm-node` có outline 2px khi node được focus, đã chụp ảnh xác nhận. Probe chung vẫn báo node/edge "không có ring" vì nó đo trên wrapper; đây là false positive đã biết.
- **P1 ô tìm:** ĐẠT. `.rp-search` có `:focus-within` với outline 2px.
- **P2 landmark:** ĐẠT. Canvas đã có `main` + `h1`. Còn P3: inspector lặp lại tiêu đề bằng `h2` cùng chữ; có thể hạ xuống `p` đậm.
- **P2 side-tab / glyph:** ĐẠT. Thẻ lỗi dùng hairline + icon; glyph đã đổi sang Lucide. Ngoại lệ: "⚠" trong `<option>` của chọn model (Inspector:69). Sửa bằng cách bỏ glyph, vì chữ `model_picker.needs_free_ram` đã tự nói rõ.
- **P2 nút "?" điều khoản:** còn 15×24. Đạt chiều cao, chiều rộng vẫn hẹp; hạ xuống P3 vì đã có khoảng cách quanh nút.
- **P2 số đo lệch:** chấp nhận theo giải thích của frontend. Inspector tính theo tham số hiện tại (`estimate.duration_basis` đã nói rõ nguồn), còn thư viện hiện số research đo với cấu hình mặc định. Trên app thật cả hai đều có số. Mock chỉ có số của thư viện, nên trên mock mới trông lệch.
- **Tổng:** 0 P0, 0 P1, 0 P2 mở; còn 4 P3. D7 coi như xong cho 1.0.
