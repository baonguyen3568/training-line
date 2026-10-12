# Review UI — S-5 vòng 3: canvas + Run monitor (bao-design, 2026-10-12)
Ảnh: `design/reviews/s5-r3/` (light r3_*, dark r3d_*). Verdict: **Canvas APPROVE**. **Run monitor: CHANGES REQUESTED**, có 1 lỗi chặn (M1).

## Canvas — đã đạt
Mở template thấy ngay làn ① ở mép trái, zoom 1, chữ đọc tốt. Nhãn cổng lấy từ node (vd "Dữ liệu kiểm tra (tuỳ chọn)"), Checkpoint hiện "Model hoàn chỉnh", merge đã nằm ở làn ③. Dark mode đúng token.
Ghi chú nhỏ, không chặn:
- Hai node phụ thuộc nhau mà cùng làn (Fine-tune → Gộp) thì dây phải vòng lại, và dây n1→n5 đi lẫn sau node ③. Có thể xếp node phụ thuộc trong cùng làn lệch sang phải nửa cột, hoặc chấp nhận ở v1.
- **Mock template vẫn là "LoRA Qwen"**, trong khi user đã chốt template mặc định là **"Đọc hoá đơn → JSON" (Nemotron-Parse v1.1, fine-tune toàn bộ, CORD-v2)**. Buổi thử T1 cần đúng template này: key `template.invoice_to_json.*`, node `train.sft_vlm`.

## Run monitor
**M1 (chặn). Trục y của learning rate hiện "0, 0, 0".** Giá trị cỡ 1e-4 bị làm tròn, nên biểu đồ vô nghĩa và trông như lỗi.
Dùng định dạng số khoa học hoặc rút gọn cho giá trị < 0.01 (`1e-4`, `2.0e-4`). Áp cho cả tooltip và bảng.

Không chặn:
1. Tile "Bước học 19": run có 20 bước (đánh số từ 0), nên người dùng sẽ hiểu là còn thiếu 1 bước. Nên hiện **"20 / 20"** (số bước đã xong / tổng); run đang chạy thì kèm ETA.
2. Legend của loss hiện "Bước học: -- Gốc: -- Đã làm mượt: --" khi chưa rê chuột → nhiễu. Ẩn giá trị khi không có con trỏ, hoặc hiện giá trị cuối.
3. **Bộ nhớ của job: trục y nên bắt đầu từ 0** (đại lượng độ lớn). Hiện trục bắt đầu ~2.5, khiến mức dao động 3–4 GB trông như biến động lớn. Có thể thêm đường ngang tham chiếu = ngân sách RAM của job, nếu backend có.
4. Điểm tok/s ở step 0 = 0 là warmup/JIT. Đánh dấu "khởi động" (marker rỗng) hoặc bỏ khỏi trục, để không kéo mắt người xem. Tile đã dùng trung vị, tốt.
5. Trạng thái "Xong" cạnh tên run chỉ là chữ trơn. Dùng chip **màu + icon + chữ** theo `node_state`/`status` (✓ Xong, ✕ Lỗi, ⏸ Tạm dừng).
6. Run đã xong mà vẫn có marker tròn trên mọi điểm: ổn với 20 điểm. Run dài (hàng nghìn bước) thì tắt marker, chỉ giữ ✕ cho NaN/inf.
7. Lưới 3+2 để trống 1 ô. Có thể cho tốc độ và bộ nhớ chiếm cả hàng 2, hoặc đặt bảng/tóm tắt vào ô trống.
8. Tốt: mỗi chart 1 trục y, có tiêu đề kèm chiều tốt ("càng thấp càng tốt") và đơn vị, có "Xem dạng bảng", tile "bộ nhớ đỉnh của job" tách khỏi RAM máy.

## Copy
Đã thêm `app.nav_*`, `run_monitor.*` (gồm chart.* và status.* theo enum RunStatus); i18n 499 key, vi/en khớp.
