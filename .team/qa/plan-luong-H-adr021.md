# Kế hoạch test luồng H — Mang model/data của bạn (ADR-021)

Owner: bao-tester · 2026-10-14 · Release 1.0 checklist mục 8 (H1–H5)
Nguồn: docs/adr/ADR-021-bring-your-own-model-data.md. Mọi ca đều chạy qua app (API + UI), không chạy script riêng
thay cho đường của user. "Đạt" nghĩa là có bằng chứng: run dir / report / output E2E, được ghi vào checklist.

## Nguyên tắc
- **check_support là nguồn sự thật duy nhất** (ADR-021 §6): với mỗi ca, kết quả `POST /api/graphs/validate` phải trùng
  với kết quả khi chạy thật (worker). Validate xanh mà job chạy lỗi = bug; validate chặn mà lẽ ra chạy được = bug.
- **Ghim trước, tải sau**: revision người dùng dán (branch/tag/không ghi) luôn được đổi thành sha 40 ký tự và hiển thị
  trước khi tải; test kiểm sha trong workflow đã lưu, run.json và cache.
- **Không chạy code của repo lạ** ở bất kỳ bước nào trước khi train (đọc config, đọc schema dataset, xem trước mẫu).
- Số liệu "Chưa kiểm chứng" (RAM / thời gian) là cận trên: ca chạy thử nhanh phải cho số đo thật ≤ số ước lượng.

## H1 — ≥ 3 model VLM ngoài catalog chạy thử nhanh thật trên GB10 (ADR-021 §2)
Research chọn 3 repo khác họ kiến trúc (vd 1 Qwen-VL khác size, 1 họ LLaVA/Idefics/SmolVLM, 1 họ Gemma/InternVL native),
mỗi repo ≤ 10 GB, không gated, không remote code. Với mỗi repo:
1. Dán repo id (không revision) → app hiện: tên, kiến trúc → family, sha ghim, dung lượng, license; nhãn "Chưa kiểm chứng".
2. Xác nhận license → G-5 tải qua egress-proxy (bằng chứng S14 như catalog: chỉ host HF, không token trong log).
3. Ghép vào template "Fine-tune VLM bằng LoRA" thay model nguồn → validate xanh (không lỗi chặn).
4. "Chạy thử nhanh" (vài bước, GPU lock + preflight) → completed; có loss hữu hạn, checkpoint/adapter đúng định dạng.
5. Đo thật: peak RAM, thời gian/bước ≤ ước lượng hiển thị (cận trên). Nếu vượt → bug ước lượng.
6. Chạy thử (Trial) trên adapter / bản merge → trả lời được một ảnh CORD.
Đạt khi cả 3 repo qua 1–6. Ghi: repo@sha, family, ước lượng vs số đo, run id.

## H2 — 1 dataset HF ngoài catalog: ánh xạ cột → train (ADR-021 §4)
Research chọn 1 dataset parquet có ảnh (vd VQA/OCR nhỏ, ≤ 2 GB, license cho phép train). Kiểm:
1. Dán repo id (+ config/split) → app đọc schema mà KHÔNG chạy loading script; hiện danh sách cột + kiểu.
2. Gợi ý ghép cột ảnh / câu hỏi / câu trả lời (hoặc messages) đúng; người dùng sửa được; xem trước ≥ 3 mẫu khớp dữ liệu gốc.
3. Bộ kiểm chất lượng như data local chạy và báo: trùng lặp, rò rỉ train/test (DATA-007), kích thước ảnh, độ dài token.
4. Train LoRA ngắn trên 1 model catalog với dataset này → completed; mẫu trong batch đầu khớp ánh xạ (so 3 mẫu).
Ca âm: ghép sai kiểu (cột text vào ô ảnh) → lỗi dễ hiểu trước khi chạy; thiếu cột bắt buộc → chặn.

## H3 — Các ca bị chặn (lỗi dễ hiểu, không stack trace, KHÔNG tải gì / không chạy gì)
| Ca | Repo mẫu | Kỳ vọng |
|---|---|---|
| Model cần remote code (`auto_map` / `trust_remote_code`) | một repo VLM có modeling_*.py ngoài catalog | chặn ngay khi dán; câu giải thích + gợi ý đề nghị đưa vào catalog; 0 byte tải, 0 file .py được đọc |
| Kiến trúc chưa hỗ trợ native | `architectures` không thuộc bảng family của research | chặn, nói rõ kiến trúc nào, không tải |
| Model không phải VLM (text-only) | một LLM text | chặn hoặc cảnh báo theo luật của backend (ghi kỳ vọng khi backend chốt) |
| Gated, chưa lưu token | một repo gated | báo cần token + hướng dẫn; không lộ token; khi lưu token → tải được (ca này đóng luôn S14(6)) |
| Repo sai | repo không tồn tại / gõ sai / revision không tồn tại / repo private | lỗi dễ hiểu theo từng trường hợp; không SYS-000 |
| Dataset có loading script (.py) | một dataset dạng script | chặn, không chạy script |
| config.json bất thường | config quá lớn, không phải JSON, symlink, `architectures` rỗng | từ chối an toàn (đọc giới hạn byte, không follow link) — security viết ca cùng mình |
| Vượt budget dung lượng / RAM | model > ngưỡng hoặc ước lượng > 80% RAM trống | chặn hoặc yêu cầu xác nhận như G-5, ADR-005 A1 |
Mỗi ca: kiểm cả validate (API) lẫn UI (thông báo hiện đúng chỗ, theo nhãn UI chứ không phải id) và E2E tier mock/app.

## H4 — Regression: catalog sinh lại từ recipe chung khớp config cũ (ADR-021 §2, §5)
1. Với MỌI model trong catalog: template train sinh từ recipe chung của family == template hiện có
   (configs/train/templates/*), so sau khi chuẩn hoá (YAML → dict, sắp khoá). Khác biệt nào cũng phải được liệt kê
   và giải thích từng dòng, hoặc là bug.
2. Recipe AutoModel render ra (build_recipe) cho params mặc định + 3 bộ params khác (LoRA, QLoRA nf4, full SFT nếu có)
   trùng byte với bản trước refactor (snapshot chụp TRƯỚC khi backend đổi code).
3. Cache key của executor không đổi với workflow cũ (không ép train lại những gì đã cache), trừ khi có lý do ghi rõ.
4. Bộ số đã công bố không bị ảnh hưởng: chạy lại 1 eval ngắn (R2 validation[0:30]) cho cùng checkpoint → cùng điểm.
Mình chụp snapshot (2) ngay bây giờ, trước khi code luồng H vào main.

## H5 — E2E
- Tier mock/app (CI): dán repo hợp lệ / remote code / repo sai / dataset ghép cột: UI đúng, không request ra ngoài origin.
- Tier external (compose, PM chạy): 1 model ngoài catalog từ dán repo → tải → validate → chạy thử nhanh.

## Phụ thuộc
research: danh sách repo H1/H2/H3 + bảng family · backend: API + check_support · fixcode: mã lỗi và câu chữ ·
security: ca config.json bất thường + review · frontend: UI dán repo / ghép cột.

## Repo đã chọn (research, 2026-10-14, sha từ HF API)
- H1:
  - Qwen/Qwen3-VL-4B-Instruct @ebb281ec70b0 (qwen3_vl, apache-2.0, 8.89 GB)
  - HuggingFaceTB/SmolVLM-Instruct @81cd9a775a4d (idefics3, apache-2.0, 4.5 GB)
  - Qwen/Qwen2-VL-2B-Instruct @895c3a49bc3f (qwen2_vl, apache-2.0, 4.43 GB)
  - Thêm (họ không phải Qwen + bộ lọc chỉ tải safetensors): llava-hf/llava-onevision-qwen2-0.5b-ov-hf @74dd0bf8
    (repo 14 GB vì có onnx; phần cần tải khoảng 1.8 GB). Mình chạy cả 4 nếu GPU cho phép, tối thiểu 3 khác họ.
- H2: katanaml-org/invoices-donut-data-v1 @d2cde298e79c (MIT, 0.2 GB, cột image + ground_truth JSON kiểu Donut).
  Ca phụ WebDataset: pixparse/docvqa-wds @23106d41, chỉ 1–2 shard, theo điều kiện của security (không train-rồi-báo-điểm,
  không đưa ảnh lên site/docs, E2E kiểm nhãn "license card chưa duyệt").
- H3:
  - remote code: OpenGVLab/InternVL3-2B @89915501
  - kiến trúc chưa hỗ trợ: Salesforce/blip2-opt-2.7b @59a1ef6c1e51
  - text-only: Qwen/Qwen3-0.6B @c1899de289a0
  - gated: google/gemma-3-4b-it @093f9f388b31
  - dataset có script: nlphuji/flickr30k @2b239befc81b (dự phòng HuggingFaceM4/VQAv2 @4b98c864262e)
- Bảng họ kiến trúc: .team/knowledge/vlm-families-26.08-7.md.
