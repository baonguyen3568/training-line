# Review M1-4 vòng 1: thư viện, canvas trống, nối dây (bao-design, 2026-10-12)
Chạy trên mock (`VITE_MOCK_API=1`, port 7876 riêng của design). Ảnh: `design/reviews/m1-r1/`.
Thao tác thật bằng CDP: Workflow trống → double-click thêm Dataset HF, Chọn model, Fine-tune toàn bộ, Đo điểm → nối bằng "Nhận từ" → Chạy.
Bước chạy chưa chụp được: trang reload do HMR khi frontend đang sửa code, nên mỗi lần reload là mất workflow (xem K1). Phần trạng thái chạy sẽ review ở vòng 2.

Verdict: **CHANGES REQUESTED.** Hướng đi rất tốt; tạo pipeline từ trống giờ làm được. Có 3 lỗi chặn.

## Chặn
**K1. Reload trang (F5, mất mạng, HMR) là mất toàn bộ workflow chưa lưu.** Với người no-code, đây là mất việc.
Sửa: tự lưu bản nháp (graph + vị trí + tham số) vào localStorage mỗi lần đổi, debounce ~1s. Mở app có bản nháp thì hỏi 1 dòng "Khôi phục workflow chưa lưu từ {time}? [Khôi phục] [Bỏ]". Cảnh báo beforeunload vẫn giữ.

**K2. "Nhận từ" cho phép nối vào chính mình và tạo vòng.** Ở node Fine-tune, ô "Nhận từ — Model gốc" liệt kê cả "Fine-tune toàn bộ · Model hoàn chỉnh", tức là output của chính node đó.
Sửa: loại chính node đó và mọi node hạ nguồn (nối vào sẽ tạo vòng) khỏi danh sách. Kéo dây cũng phải chặn tương tự (CFG-002).

**K3. Canvas trống mà thanh kiểm tra lại ghi "✓ Mọi thứ đã sẵn sàng".** Workflow rỗng không sẵn sàng.
Sửa: khi 0 node thì hiện "Thêm bước để bắt đầu" (key `preflight.empty`, đã thêm), nút Chạy vẫn tắt.

## Không chặn
1. **Mặt node hiện mã catalog** "nemotron-parse-v1.1". Phải hiện `option_meta.label` ("NVIDIA Nemotron Parse v1.1") như ô chọn model.
2. **Thêm node bằng double-click** khi node rơi ngoài vùng đang nhìn (vd Đo điểm ở làn ⑥): người dùng không thấy gì xảy ra. Phải cuộn tới node mới, chọn sẵn và nháy viền 600ms (spec §2).
3. **Inspector không liệt kê lỗi của node**, dù node đang báo "✕ Lỗi (2)". Người dùng phải tự đoán. Đầu inspector cần hiện thẻ lỗi: tiêu đề + 1 câu + action. Với CFG-003 action là "Chọn nguồn cho đầu vào này", focus đúng ô "Nhận từ".
4. Ô tìm kiếm bị cắt placeholder ("…LoRA, be"). Đã rút copy còn "Tìm bước… (vd: nén)".
5. Mục thư viện cao 70–80px vì desc dài 2–3 dòng. Spec là 1 dòng, cắt "…", full trong tooltip. Mục gọn hơn thì thấy được nhiều node hơn. Thêm `role="option"` trong `listbox` (a11y).
6. Mock vẫn dùng `epochs` dạng ô số "2", trong khi registry thật là 1/3/5 segmented (`epoch_count`). Không lỗi UI, chỉ cần mock đồng bộ registry.
7. Mình đồng ý cắt khỏi M1: **sparkline** và **hạt sáng trên dây** (animated mặc định là đủ), và **kéo dây ra khoảng trống mở search** (v1.1).
   Riêng **gợi ý từng bước** sau ô canvas trống, mình đề nghị giữ trong M1, ít nhất là gợi ý 3 (cổng bắt buộc chưa nối thì nhấp nháy + "Nối {port_type} vào đây"). Đây là chỗ user vừa vấp: "chưa nối hay tạo pipeline gì hết". Logic chỉ là đọc lỗi CFG-003 đang có.

## Đạt
Thư viện bên trái nhóm theo làn, toolbar (Workflow trống / Mở / Lưu / Hoàn tác / Làm lại / Nhân bản / Xoá), ô "Kéo Dữ liệu vào đây", banner "Chế độ demo", node vào đúng làn, nối bằng "Nhận từ" đã phân biệt cổng, thanh kiểm tra đếm lỗi.
