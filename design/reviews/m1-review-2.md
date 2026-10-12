# Review M1-4 vòng 2: sửa K1–K3 + trạng thái chạy (bao-design, 2026-10-12)
Mock 7876. Kịch bản thật: Workflow trống → thêm 4 node → nối bằng "Nhận từ" → ▶ Chạy → chờ xong → reload trang.
Verdict: **APPROVE** cho luồng M1 "tạo → nối → chạy". Còn 4 chỉnh nhỏ về câu chữ và trạng thái; xem cùng lúc với các việc frontend đang làm.

## Đã đạt
- K1: reload → hiện "Khôi phục workflow chưa lưu từ … ? [Khôi phục] [Bỏ]".
- K2: không còn tự nối vào chính mình.
- K3: canvas trống hiện "Thêm bước để bắt đầu".
- Mặt node hiện "NVIDIA Nemotron Parse v1.1".
- Khi chạy, node lần lượt: ✓ Xong → ◐ Đang chạy 40% → ⧗ Đang chờ. Nút đổi thành "■ Dừng", có dòng tiến độ; xong thì về "▶ Chạy". Dây đúng màu theo kiểu.

## Cần chỉnh (không chặn)
1. **Số bước đang chạy bị lệch.** Hiện ghi "Bước 2/4 · Fine-tune toàn bộ" trong khi Fine-tune là bước ③ (đã xong 2). Người dùng sẽ hiểu Fine-tune là bước 2.
   Đã đổi copy `run.progress` thành "Đang chạy bước {current_step}/{total_count} · {node}". Truyền `current_step` = số thứ tự của node đang chạy (khớp số ① ② ③ trên node).
2. **Thanh dưới vừa ghi "✓ Mọi thứ đã sẵn sàng" vừa hiện tiến độ.** Khi đang chạy, thay hẳn phần kiểm tra bằng tiến độ.
   Khi xong, dùng `run.finished` "Chạy xong sau {duration}" (hoặc `run.finished_with_errors`) + nút "Xem kết quả", thay cho chữ "Xong" trơn.
3. **Thời gian theo định dạng Mỹ** ("10/12/2026, 5:35:42 PM") trong giao diện tiếng Việt. Format theo locale đang chọn: vi → "12/10/2026 17:35", en → như cũ. Áp cho mọi chỗ hiện giờ: banner khôi phục, Lịch sử chạy, "dùng lại kết quả lúc".
4. Banner khôi phục hiện trên gallery: tốt. Bấm "Khôi phục" nên mở thẳng canvas với workflow đó, còn bấm "Bỏ" thì xoá bản nháp.

## Việc frontend đang làm tiếp (đã thống nhất)
Cuộn tới node mới thêm → thẻ lỗi trong inspector → gợi ý cổng bắt buộc chưa nối (CFG-003) → rút gọn mục thư viện.
