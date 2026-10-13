---
title: Chính sách lưu giữ dữ liệu
description: App giữ checkpoint, kết quả và lịch sử chạy trong bao lâu, và những gì không bao giờ bị xoá.
sources: [docs/adr/ADR-005-storage-retention.md, .team/proposals/005-storage-retention-policy.md, src/vlm_pipeline/artifacts/retention.py]
---

# Chính sách lưu giữ dữ liệu

Chính sách này áp dụng cho dữ liệu app tạo ra **trên máy của bạn**. App không gửi dữ liệu đi đâu; mọi thứ nằm trong
thư mục dự án.

## Nguyên tắc

1. **Xem trước, rồi mới xoá.** Mọi công cụ dọn dẹp mặc định chỉ liệt kê; phải yêu cầu rõ mới xoá.
2. **Chỉ trong thư mục dự án.** Không động vào dữ liệu, container hay cache của ứng dụng khác trên máy.
3. **Lịch sử không bao giờ bị xoá.** Thông tin lần chạy, nhật ký, số đo và sự kiện luôn được giữ.
4. **Model bạn đã train không bao giờ bị tự động xoá.**
5. **Mọi lần xoá đều được ghi lại**: cái gì, bao nhiêu, lúc nào, ai làm, ai duyệt.

## Luôn được giữ

- `run.json` (thông tin lần chạy), các file `*.jsonl` (số đo, sự kiện), thư mục `logs/`
- thư mục `export/` (model bạn đã xuất)
- mọi lần chạy hoặc checkpoint có file `KEEP`
- lần chạy đang chạy, đang chờ, bị tạm dừng, hoặc đang giữ GPU

## Checkpoint khi train

Trong lúc train, app giữ **2 checkpoint gần nhất** (có thể chỉnh 1–5) và bản model hoàn chỉnh cuối cùng.

Sau khi lần chạy **hoàn thành**:

| Checkpoint | Giữ |
|------------|-----|
| Checkpoint mới nhất | đầy đủ trong **7 ngày** (để train tiếp được), sau đó bỏ trạng thái optimizer, giữ model |
| Checkpoint tốt nhất (sai số kiểm định thấp nhất) | giữ model, bỏ trạng thái optimizer ngay |
| Checkpoint lưu dở (bị ngắt khi đang ghi) | xoá |
| Các checkpoint khác | xoá sau **7 ngày** |

Sau khi lần chạy **thất bại hoặc bị huỷ**:

- **3 ngày** đầu: không xoá gì, để bạn kiểm tra hoặc chạy tiếp.
- Sau 3 ngày: xoá checkpoint lưu dở và các checkpoint không phải mới nhất.
- Sau **7 ngày**: xoá cả checkpoint mới nhất.

<!-- popover: "Trạng thái optimizer là gì?" -->
Ngoài trọng số model, mỗi checkpoint còn lưu trạng thái của thuật toán tối ưu để có thể train tiếp đúng chỗ dừng.
Phần này khá lớn: đo thực tế với Nemotron-Parse v1.1 là 2,7 GB trên 8,1 GB mỗi checkpoint. Bỏ
nó đi thì model vẫn dùng, nén và đánh giá được, chỉ không train tiếp từ điểm đó được nữa.

Khi một model chưa hỗ trợ train tiếp mà bạn yêu cầu chạy tiếp, app cất checkpoint và số đo cũ sang một bên rồi
train lại từ đầu. Phần đã cất này bị xoá sau **3 ngày**.

Thời gian được tính từ hoạt động cuối của lần chạy; nếu có nghi ngờ, app chọn cách **giữ lâu hơn**.

## Kết quả node trong workflow

Xem [Quản lý dung lượng](./quan-ly-dung-luong#bo-nho-dem-ket-qua-node): kết quả tạm bị xoá theo thứ tự dùng lâu nhất
khi vượt giới hạn; model đã train chỉ bị xoá khi bạn tự chọn và gõ xác nhận.

## Image container

Image của dự án mang nhãn `project=vlm-training` hoặc tên bắt đầu bằng `vlm-`. App không bao giờ chạy lệnh dọn
toàn bộ Docker của máy. Image gốc của NVIDIA chỉ bị xoá theo quy trình 2 bước có quản trị viên xác nhận.

## Mức cảnh báo dung lượng

| Tổng dữ liệu dự án | Mức |
|--------------------|-----|
| dưới 500 GB | bình thường |
| từ 500 GB | cảnh báo |
| từ 800 GB | nguy hiểm |

## Nhật ký xoá

Mỗi lần xoá ghi một dòng vào `outputs/system/cleanup.jsonl` và vào nhật ký sự kiện của lần chạy liên quan, gồm loại
sự kiện, đường dẫn, dung lượng, thời điểm, người thực hiện và người duyệt (nếu có).
