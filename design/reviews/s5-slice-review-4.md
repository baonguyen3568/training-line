# Review UI — S-5 vòng 4 (bao-design, 2026-10-12)
Ảnh: `design/reviews/s5-r4/`. Verdict: **CHANGES REQUESTED**, 2 lỗi chặn (M1 chưa sửa xong, E1 mới).

## Đã đạt
Tile bước "20", chip "✓ Xong", loss full-width + lưới 2×2, chart bộ nhớ từ 0, gallery có "Đọc hoá đơn → JSON" đứng đầu và estimate `unknown` hiện "chưa có số đo".

## Chặn
**M1 (còn). Trục y learning rate vẫn hiện "0, 0, 0"** (`r4_monitor.png`).
- `formatNumber` mới gắn cho `series[].value` (legend), **chưa gắn cho trục**.
- Trong `LineChart.tsx`, trục y `{ ...axis, size: 60 }` cần thêm `values: (_u, ticks) => ticks.map((v) => formatNumber(v))`. Nên áp cho cả trục x để nhất quán.
- Test hiện chỉ kiểm hàm, chưa kiểm trục; nên thêm test render hoặc kiểm trực tiếp option của axes.

**E1. Ước lượng không nhất quán giữa gallery và thanh kiểm tra.**
- Thẻ template ghi "chưa có số đo", nhưng mở canvas thì thanh kiểm tra lại hiện "Dự kiến ~20–45 phút · RAM ~18–26 GB · Ổ đĩa ~4–8 GB". Đó là số của mock cũ, và là đúng loại "số bịa" mà P010 §6 cấm.
- Validate mock của template invoice_to_json phải trả `source: unknown`, và thanh kiểm tra hiện "chưa có số đo".

## Không chặn
1. Legend loss vẫn hiện "Bước học: -- · Gốc: — · Đã làm mượt: —". `setLegend({idx: last})` có vẻ chạy trước khi dữ liệu sẵn sàng, hoặc bị ghi đè khi build lại. Mục tiêu: trước khi rê chuột thì hiện giá trị bước cuối.
2. Gallery: "⏱ chưa có số đo · chưa có số đo" bị lặp. Khi cả thời gian và RAM đều unknown thì chỉ hiện một lần: "⏱ chưa có số đo".
3. Gallery vẫn có thẻ "Dạy model đọc tài liệu của bạn (LoRA)". Template thứ 2 user đã chốt là **"Hiểu ảnh và video" (Cosmos-Reason2-2B, gated)**, key `template.image_video_understanding.*`, chip "Cần tài khoản Hugging Face" + "Built on NVIDIA Cosmos". Nên thay thẻ LoRA bằng thẻ này, để buổi thử thấy đúng sản phẩm.
4. Canvas: node train.sft_vlm hiện 2 cổng "Dữ liệu / Dữ liệu". Đã thêm copy `node["train.sft_vlm"].port.*` (Dữ liệu học / Dữ liệu kiểm tra (tuỳ chọn)) và `node["data.hf_load"].param.repo`; cần build lại i18n.
5. **Mặt node nên hiện giá trị chính đang chọn**, vd "Chọn model gốc" kèm dòng phụ "Nemotron-Parse v1.1", "Dataset Hugging Face" kèm "CORD-v2". Người no-code nhìn canvas phải biết đang dùng model và dữ liệu nào mà không cần mở inspector. Đề xuất `x-ui.summary: true` cho param chính.
6. Run mẫu hiện tại là "S-3 smoke LoRA Qwen3.5-0.8B (MedPix, spike only)". MedPix là CONDITIONAL, chỉ dùng cho spike, nên không để run này trong instance dùng cho buổi thử với người dùng (dùng fixture riêng).
