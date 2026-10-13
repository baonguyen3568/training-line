# D2 · Bản đồ tuyến (IMPECCABLE'S PICK)

**Luận điểm.** Workflow là một tuyến metro: mỗi bước là một ga, dây nối là đường tàu, và bước nhận nhiều đầu vào là ga đổi tuyến. Người không biết code vẫn đọc được bản đồ metro (lên ở đâu, đổi ở đâu, tới đâu). Hướng này tránh cả hero SaaS chung chung lẫn kiểu graph rối dây.

**Bảng màu**
- Landing (ban ngày, giấy men):
  - nền `#f1f0ec`, phụ `#e7e6e0`;
  - chữ `#15171a` / `#4a4c50` / `#626469` (≥ 4.5:1);
  - viền `#cfcdc5`.
- Màu thương hiệu: vàng biển báo `#ffd21f`, luôn đi với chữ mực `#15171a` (13:1). Không dùng xanh NVIDIA.
- Màu tuyến = họ cổng của app:
  - Dữ liệu ■ `#1b9669`, Model ● `#2a78d6`, Điểm ◆ `#ca5a2e`, Chạy thử ▲ `#52514e`;
  - landing dùng bậc fp8 để đạt ≥ 3:1 trên nền giấy.
- Dashboard (tối, như bảng sơ đồ phát sáng):
  - nền `#121417`, panel `#1a1d21`, `#20242a`, viền `#2d3238`;
  - chữ `#ecebe6` / `#b9bbbf` / `#8a8d93`;
  - màu tuyến lấy token dark `#199e70` `#3987e5` `#d95926` `#c3c2b7`. validate_palette dark: ALL PASS, CVD ΔE ≥ 19.6, contrast ≥ 4.35 trên panel.

**Chất liệu.**
- Đường tàu dày 12px, chỉ đi 0°/45°.
- Ga là chấm trắng viền mực.
- Ga đổi tuyến là viên con nhộng trắng.
- Biển vàng có viền mực.
- Lưới chấm 20px trên canvas.
- Không gradient, không glass.

**Chữ.** Be Vietnam Pro (SIL OFL-1.1, tự host trong `design/m2/fonts/`, kèm `OFL-BeVietnamPro.txt`).
- Bold làm giọng biển báo, regular cho thân bài.
- Đủ dấu tiếng Việt.
- Số dạng tabular.

**Icon.** Lucide (ISC), nhúng sẵn SVG từ `icons/`, kèm LICENSE.

**Logo.** Wordmark "NTC" là một biển tuyến (viên vàng, hai đầu nối màu Model/Dữ liệu) + chữ "Training". Bản 32px trong toolbar dashboard.

**Màn đầu (landing 1440).**
- Thanh nav mực.
- Tiêu đề lớn bên trái: "Dạy AI đọc hoá đơn của bạn — kéo, nối, bấm Chạy."
- Hai dòng sự thật: Apache-2.0; "Chạy trên NVIDIA NeMo AutoModel · Tối ưu cho DGX Spark".
- Ngay dưới: bản đồ tuyến trải hết chiều ngang. "Ga xuất phát" bên trái chính là CTA "Tải về & cài đặt", kèm lệnh cài ghi rõ *minh hoạ*.

**Tương tác đặc trưng.**
- Đoàn tàu vàng chạy từng ga trên tuyến Model, dừng ở mỗi ga.
- Rê chuột hoặc Tab vào một ga thì mở popover "?" với mô tả lấy nguyên văn từ `vi.yaml` (node.*.desc) và các cổng vào/ra.
- Nhánh chấm chấm là đường LoRA thay thế.
- Khi `prefers-reduced-motion`: tàu đứng yên ở ga Xuất.

**Mục 2.** "Chặn lỗi trước khi chạy": dây ◆ cắm sai vào cổng ● bị từ chối (đúng copy `canvas.wire_rejected`), kèm thẻ MEM-001 (số 14/9 GB là minh hoạ).

**Dashboard.**
- Tối, kiểu dock của Unreal.
- Toolbar chỉ có icon + tooltip; nhãn chữ duy nhất là nút chính Chạy/Dừng.
- Rail icon cho thư viện 7 làn, mỗi làn có vạch màu họ cổng.
- Canvas dây đi 0°/45° như đường tàu.
- Inspector bên phải: phần "Dự kiến" ghi "chưa có số đo", không bịa số.
- Thanh kiểm tra ở đáy.

**Mở rộng sang bề mặt khác.**
- Docs: mỗi trang là một ga, mục lục là sơ đồ tuyến.
- Changelog: "các ga mới mở".
- Roadmap: tuyến đang xây (nét đứt).
- 404: "Ga không tồn tại".

**Rủi ro thật.**
- Ẩn dụ bản đồ metro cho pipeline khá **quen**: nhiều công cụ dữ liệu đã dùng, nên không gây bất ngờ.
- Bản đồ cần cuộn ngang trên điện thoại.
- Vàng biển báo dễ bị đọc nhầm thành "cảnh báo". Vì vậy cảnh báo thật luôn có icon và chữ, còn vàng chỉ dùng cho nút chính và đoàn tàu.
