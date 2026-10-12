# Review UI — S-5 vertical slice, vòng 2 (bao-design, 2026-10-12)
Ảnh: `design/reviews/s5-r2/` (light: r2_*, dark: r2d_*). Công cụ: `design/tools/cdp-shot.mjs` (Chromium headless của Playwright cache, CDP, không cần thêm dependency).
Verdict: **APPROVE có điều kiện**. B1 và B2 đã sửa. Còn 2 lỗi cần sửa trước buổi thử với người dùng (C1, C2); các mục khác không chặn.

## Đã sửa, xác nhận
- B1: node không còn bị nền làn che. Xem `r2_canvas.png`.
- B2: nút "▶ Chạy" dùng chữ đen trên nền xanh, segmented và link đúng token.
- Chữ trên canvas đọc được; nhãn làn dính mép trên; panel phải có câu hướng dẫn; chip dung lượng ghi "còn trống"; MiniMap đã ẩn.
- Inspector khi chọn node (`r2_select.png`) rõ ràng: tiêu đề, mô tả, nhãn tham số từ i18n.

## Cần sửa trước buổi thử
**C1. Mở template thì làn ① Nguồn bị cắt ngoài mép trái, node cuối bị cắt ở mép phải.**
- Đây là điều đầu tiên người dùng thấy, và node "👉 Thay bằng dữ liệu của bạn" nằm đúng ở làn ①.
- Đề xuất: khi mở template, căn viewport theo **góc trên-trái** (x của làn ①, y của node cao nhất − 24px) ở zoom 1, thay vì căn giữa. Phần bên phải xem bằng cuộn ngang.
- Tuỳ chọn thêm: thanh mục lục làn ①–⑦ ở mép trên, bấm vào để cuộn tới làn đó.

**C2. Node "Gộp phần học thêm vào model" (export.merge_lora) nằm ở làn ⑦ Xuất nhưng chạy *trước* bước Đo điểm ⑥.**
- Hệ quả: dây đi ngược từ phải sang trái và vắt ngang canvas. Số thứ tự bước (5 ở bên phải 6) cũng mâu thuẫn với vị trí.
- Gộp LoRA là bước tạo model hoàn chỉnh sau khi học, không phải bước xuất.
- Đề xuất: `export.merge_lora.category = "train"` (làn ③), hoặc làn tự đẩy node sang phải khi nó cấp dữ liệu cho node ở làn trước. Cần backend chốt category.

## Không chặn
1. Cổng output `Checkpoint` của merge đang dùng `label_key: port.model` → nên là `port.checkpoint` ("Model hoàn chỉnh").
2. Cổng tuỳ chọn hiện "Dữ liệu ·" (dấu chấm lửng). Dùng copy port của node, vd `node["train.lora_vlm"].port.val_data` = "Dữ liệu kiểm tra (tuỳ chọn)", hoặc thêm chip "tuỳ chọn".
3. Tiêu đề node bị cắt ("Chia dữ liệu học / kiểm…"): thêm `title` tooltip khi bị cắt, hoặc cho tiêu đề xuống 2 dòng.
4. Theme tối: nút zoom (Controls) và nhãn "React Flow" là khối trắng. Style theo `color.surface`/`text` (attribution React Flow bỏ được với `proOptions.hideAttribution`; license MIT cho phép, nhưng nên giữ ghi công ở trang About).
5. Kiểm tra tự động: `design/tools/cdp-shot.mjs` + biểu thức `elementFromPoint` chỉ đúng cho node *trong* viewport. Test e2e của tester nên cuộn tới từng node (fitView theo node) trước khi kiểm.
6. Copy: dòng ước lượng đổi "Disk" → "Ổ đĩa" (đã sửa trong i18n).

## Tiếp theo
Sửa xong C1 và C2 thì slice đủ cho buổi thử T1–T3. T4 và T5 cần lỗi giả lập và dialog cài; có thể dùng fixture của tester và prototype tĩnh. Mình sẽ báo PM để hỏi user tìm người tham gia.
