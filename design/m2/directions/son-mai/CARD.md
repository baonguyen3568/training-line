# D1 — Sơn mài (hướng được bốc thăm, seed 00ab95ce)

**Luận điểm.** Train model giống làm tranh sơn mài: phủ từng lớp, rồi *mài* để lộ hình bên dưới. Site không đi theo khuôn mẫu chung của ngành (hero tối + gradient phát sáng + hàng card).

**Bảng màu** (mọi màu trung tính đều ngả về son/nâu, không dùng xám):
- nền: sơn đen #140e0b / #0c0807 / panel #1c1410
- son: #a8261a (nút chính, mảng tranh) · đậm #6e1810 · chữ son #ec7058
- vàng quỳ #c9a14a · vỏ trứng #efe6d2 / #c3b398 / #9c8a71
- trạng thái giữ nguyên nghĩa, luôn có icon + chữ: tốt #4cbf6a · cảnh báo #f2b63d · lỗi #ff7a6b
- cổng dữ liệu ■ #1baf7a · model ● #3987e5 · kết quả ◆ #e58a4a
- Không dùng #76b900.

**Chất liệu.** Lớp sơn son quét cọ, rắc bụi vàng, có vệt bóng. Vỏ trứng rạn được vẽ bằng voronoi procedural, không dùng ảnh stock. Viền vàng quỳ; khung tranh là sơn đen.

**Chữ.**
- Display: **Phudu** (Dương Trần, OFL-1.1, đủ dấu tiếng Việt, wght 300–900).
- UI/thân bài: **Be Vietnam Pro** (OFL-1.1).
- Cả hai self-host trong `design/m2/fonts/` kèm file OFL.

**Logo.** "NTC" đặt trên ô son có viền vàng quỳ, kèm "Training" làm phụ. Dùng cùng một SVG cho nav 36px, toolbar 32px và favicon.

**Màn đầu (1440×900).**
- Tiêu đề Phudu cỡ lớn, "hoá đơn" bằng vàng quỳ.
- Dưới tiêu đề là một bức tranh sơn son đóng khung, chiếm hết chiều ngang. Bên dưới lớp sơn là pipeline 7 làn khảm vỏ trứng, nối bằng dây vàng.
- Dưới cùng là "biển ghi chú tranh" chứa hành động chính: nút **Tải về & cài đặt** + lệnh cài (minh hoạ), "Miễn phí · Apache-2.0", "Chạy trên NVIDIA NeMo AutoModel · Tối ưu cho DGX Spark".

**Tương tác đặc trưng.**
- Rê chuột hoặc vuốt để *mài* lớp son (mask trên canvas), lộ dần pipeline thật.
- Khi tải trang, một nhát mài tự chạy qua cả 7 làn để thấy ngay cơ chế. Cuộn trang thì mài thêm 2 nhát.
- Có nút "Mài hết" cho người dùng bàn phím. Khi `prefers-reduced-motion` thì pipeline lộ sẵn.

**Mở rộng sang trang khác.**
- Landing: dùng trọn thế giới sơn mài.
- Docs/chính sách: nền sơn đen, chữ vỏ trứng, tiêu đề Phudu, "?" mở popover viền vàng, số liệu kèm đường khâu dẫn nguồn.
- Dashboard: editor dạng dock kiểu Unreal, sạch. Sơn mài chỉ còn ở chi tiết: màu trung tính ngả son, viền vàng mảnh khi chọn, son ở nút Chạy/Dừng. Không có vân rạn hay hoạ tiết trong panel làm việc.

**Nâng cấp lấy từ các ứng viên bị loại** (mỗi ứng viên góp một kỷ luật):
- *Từ Ikeda datamatics (dồn hết vào một bảng màu, chuyển động sinh từ dữ liệu thật):* chỉ dùng son–đen–vàng–vỏ trứng. Vân cọ được seed từ số đo thật trong `benchmarks/perf/s3-lora-qwen3.5-0.8b-20261012.json` (390–664 tok/s, 3,2–4,2 GiB, 6,5/12,6 GB, 308 s). Đây là minh hoạ, không phải biểu đồ.
- *Từ tấm chăn chắp vá ký ức (đường khâu nguồn gốc):* mỗi con số có một đường khâu vàng nét đứt dẫn tới "?" ghi file nguồn. Ví dụ: 35 GB → `benchmarks/perf/p1-parse-v1.1-full-26.08-2-20261012.json`; 9 GB được ghi rõ là minh hoạ.
- *Từ menu console trong khoảng tối (ngữ pháp chọn):* trên dashboard, node đang chọn nổi lên (bóng thật + viền vàng), các thao tác nhanh nằm quanh nó thành icon vệ tinh.

**Rủi ro thật.**
- Màu son của thương hiệu gần với đỏ báo lỗi. Hiện tách bằng icon + chữ và đỏ lỗi sáng hơn (#ff7a6b), nhưng người dùng có thể vẫn đọc nhầm nút Chạy màu son là "nguy hiểm".
- Mài canvas tốn CPU trên máy yếu: cần giới hạn DPR (đã giới hạn ở 2).
- Phudu chỉ có chữ hoa nên không dùng được cho tiêu đề dài trong docs; ở đó dùng Be Vietnam Pro Bold.
- Thế giới sơn mài có thể bị coi là "trang trí văn hoá" nếu landing bỏ mất phần trình diễn sản phẩm thật.
