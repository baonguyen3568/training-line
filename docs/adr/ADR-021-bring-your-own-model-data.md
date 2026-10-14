# ADR-021 — Mang model/data của bạn (model VLM bất kỳ + dataset bất kỳ từ Hugging Face)

- Trạng thái: **ACCEPTED (USER)** — 2026-10-14
- Bối cảnh: user hỏi "mỗi pipeline chỉ áp dụng cho 1 model à? … app cho user tự setup và training cho mọi model - data miễn về VLM".
  Hiện trạng: `model.hf_pick` chỉ chọn trong catalog 11 model; `data.hf_dataset` chỉ có `naver-clova-ix/cord-v2`
  (`HubDataset = Literal[...]`); `data.local_folder` đã nhận mọi data ở 3 định dạng (llava/sharegpt/hf_messages).
- Chốt bởi: user (qua PM)

## Quyết định
1. **Vào Release 1.0** (user chọn, chấp nhận lùi ~2–3 ngày).
2. **Model bất kỳ từ Hugging Face**: user dán repo id (+ revision tuỳ chọn). App:
   - đọc `config.json` (chỉ parse JSON, KHÔNG import/chạy code) → nhận `architectures` / `model_type`;
   - chỉ nhận kiến trúc mà engine (NeMo AutoModel / transformers trong image train mặc định) hỗ trợ **native**; map sang
     recipe chung theo họ (family) → sinh template train (chat template, freeze, LoRA target, `qlora_skip_modules`);
   - ghim revision thành commit sha 40 ký tự lúc thêm; kiểm dung lượng (budget tải như G-5); hiện license từ model card,
     user xác nhận license trước khi tải; repo gated → cần token user lưu trong Cài đặt;
   - nhãn **"Chưa kiểm chứng"**: RAM/thời gian là ước lượng theo số tham số (cận trên), có "Chạy thử nhanh" (vài bước) trước
     khi train thật; mọi giới hạn ≤ 80% RAM trống (ADR-005 A1) vẫn áp dụng.
3. **Model cần `trust_remote_code`: CHẶN** khi user tự thêm (user chọn). Chỉ model trong catalog (team kiểm + ghim hash từng
   file `.py`) mới được chạy remote code. Thông báo dễ hiểu: vì sao chặn, có thể đề nghị đưa vào catalog.
4. **Dataset bất kỳ từ Hugging Face**: user dán repo id (+ config/split). App đọc schema/cột (không chạy loading script;
   chỉ parquet/arrow/json/imagefolder), gợi ý ghép cột ảnh / câu hỏi / câu trả lời (hoặc messages), xem trước vài mẫu,
   chạy cùng bộ kiểm chất lượng như data local (dedupe, leak train/test, kích thước ảnh, độ dài token).
5. **Catalog giữ nguyên**, hiển thị thành mục **"Đã kiểm chứng"** (có số đo thật trên GB10). Pipeline tham chiếu vẫn ghi model
   đã đo, nhưng UI nói rõ: đổi model/data ở node nguồn là dùng được với model/data khác.
6. check_support là nguồn sự thật duy nhất cho cả model catalog lẫn model tự thêm (validate == worker).

## Phân công (luồng H của Release 1.0)
- research: bảng họ kiến trúc VLM native trong image 26.08-7 + recipe chung mỗi họ + công thức ước lượng RAM; danh sách
  kiến trúc thử nghiệm (≥3 repo ngoài catalog) để test.
- backend: node/params model tự thêm, resolver config.json → family → template, tải qua G-5, dataset HF tự do + ánh xạ cột,
  check_support.
- security: review parse config.json, chặn remote code, ghim revision, budget tải, license/gated, không chạy dataset script.
- fixcode: mã lỗi mới (kiến trúc chưa hỗ trợ, cần remote code, thiếu cột, license chưa xác nhận…) câu dễ hiểu.
- frontend + design: ô dán repo id, thẻ "Chưa kiểm chứng" / "Đã kiểm chứng", dialog license, màn ghép cột + xem trước.
- cleancode: registry/spec node, review chéo.
- tester: E2E (model ngoài catalog chạy thử nhanh thật trên GB10; dataset HF ngoài catalog; ca bị chặn remote code).

## Hệ quả
- 1.0 lùi khoảng 2–3 ngày. Site/docs cập nhật: "mọi VLM kiến trúc chuẩn", giới hạn đã biết ghi rõ.
