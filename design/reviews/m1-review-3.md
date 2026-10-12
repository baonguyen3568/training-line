# Review M1-4 vòng cuối (bao-design, 2026-10-12)
Ảnh: `design/reviews/m1-r3/`. Verdict: **APPROVE M1 về UX.** Đủ để mời user thử lại.

## Đạt
- Thêm node bằng thư viện: canvas cuộn tới node mới, node được chọn và nháy viền.
- Inspector có thẻ lỗi ngay đầu (tiêu đề + 1 câu + "Cách sửa →" focus đúng ô).
- Cổng bắt buộc chưa nối có viền nhấp nháy + gợi ý.
- Thư viện gọn: mỗi mục 1 dòng mô tả, thấy được ~12 bước không cần cuộn.

## Chỉnh nhỏ
1. Gợi ý cổng "← Nối Model vào đây" xuống 2 dòng làm node cao lên, mà nhãn cổng ngay trên đã ghi "Model gốc". Copy `canvas.hint.connect_port` đã rút còn "← nối vào đây", không còn placeholder.
2. Thẻ lỗi CFG-003 hiện id thô ("base_model", "train_data"). Backend/fixcode đang sửa: server truyền label cổng (`node.<type>.port.<name>`), không truyền tên tham số kỹ thuật.
3. **Mock bị lệch registry**: node "Đo điểm" trong mock vẫn có param `groups` (checkbox) với nhãn thô `node.eval.benchmarks.param.groups.label`, trong khi registry thật dùng `suite` (chọn một). Đề xuất sinh dữ liệu mock node-types từ snapshot `REGISTRY.to_api()` (JSON commit trong repo, test so khớp), để mock không lệch nữa. Buổi thử với user sẽ chạy trên app thật hoặc mock này, nên phải khớp.
