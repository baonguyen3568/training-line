# ADR-001 — Amendment 1: model NVIDIA cho template

- Trạng thái: **ACCEPTED** — 2026-10-12 (bao-pm chốt). Quyết định của user đã có (2026-10-12 22:10, qua PM).
- Nguồn: `.team/proposals/001a-nvidia-default-model.md` (bao-research). Vet: `.team/security-reviews/2026-10-12-nvidia-template-candidates.md` (bao-security). Kiểm class native: bao-backend.
- Thay thế mục "Catalog model → Template mặc định" của ADR-001. Các phần khác của ADR-001 (image, serve, eval, driver) giữ nguyên.

## Bối cảnh
User yêu cầu dùng model của NVIDIA. Security chỉ ra rằng repo gated buộc mọi end-user no-code phải có tài khoản HF và accept trước khi chạy template đầu tiên.

## Quyết định
| Vai trò | Model@SHA | Ghi chú |
|---|---|---|
| **Template mặc định** "Trích xuất hoá đơn → JSON" | `nvidia/NVIDIA-Nemotron-Parse-v1.1@fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4` + `naver-clova-ix/cord-v2@7f0115a4…` | Không gated. NVIDIA Open Model License (tokenizer CC-BY-4.0). Recipe chính chủ `nemotron_parse_v1_1.yaml`, class native AutoModel. **Điều kiện**: C-RADIOv2-H@0d8f4c18 pin local + offline (bước prepare) |
| Template 2 "Hiểu ảnh/video" | `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561` | Gated. NVIDIA Open Model License, "Built on NVIDIA Cosmos". Qwen3-VL native. Wizard tài khoản HF + token chỉ đọc. **Không mirror weights**. Duyệt cuối khi đọc được config + gate N-4 PASS |
| Baseline / G1-harness | `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0, APPROVED |
| Catalog phụ | Qwen3.5-0.8B/2B/4B/9B, Gemma 4 E2B/E4B | như ADR-001 |
| R&D / không làm template | Nemotron-Parse-2.0 và v1.2 (AutoModel chưa hỗ trợ: tied embeddings, đổi tên cờ chuẩn hoá ảnh → chuẩn hoá 2 lần), Nano-12B-v2-VL (bắc cầu C-RADIO, không có engine 1 GPU, RAM), Llama-Nemotron-Nano-VL-8B (cũ), Omni (RAM), Eagle (non-commercial) | |

## Hệ quả
- Smoke ưu tiên: P-0 (prepare/pin C-RADIO) → P-1 (fine-tune 50 step) → P-2 (eval field-F1 trên CORD test). Cosmos N-2/N-4 hạ ưu tiên.
- Metric template mặc định: field-F1 (chính), JSON-exact, nTED, parse-fail rate. Đưa vào registry P002.
- Template mặc định tạm **không có bước quantize** (ModelOpt chưa kiểm chứng Nemotron Parse). Serve qua vLLM ❓, nếu không được thì dùng HF generate.
- Vet model bắt buộc quét **đệ quy** `auto_map` và `_name_or_path` ở mọi sub-config (`scripts/research/hf_model_meta.py`).
- Theo dõi upstream AutoModel để lên Parse-2.0 khi class hỗ trợ.

## Ghi chú của PM khi chốt
- Template mặc định không có node quantize. Tầm nhìn sản phẩm có "nén model", nên **template 2 (Cosmos, kiến trúc Qwen3-VL, ModelOpt FP8 hỗ trợ chính thức) sẽ là template trình diễn đủ chuỗi train → quantize → benchmark**. Research + backend thử ModelOpt FP8 PTQ trên Parse v1.1 như một spike R&D (không chặn template).
- Gate trước khi Parse v1.1 vào gallery: test build-from-config offline pass (security) + P-2 field-F1 có số thật.

## Bổ sung 2026-10-12 — Gate tiền xử lý (áp dụng cho MỌI template)
Bối cảnh: Parse v1.1 bị chuẩn hoá ảnh 2 lần trong class native của AutoModel (cờ bị rơi khi merge với RADIOConfig). Hậu quả là lệch giữa train và serve.
- Template chỉ vào gallery khi có: (a) test số học ảnh trắng tổng hợp PASS (normalize đúng 1 lần); (b) test **train/serve parity**: pixel_values ở đường train (AutoModel) và đường chạy thử/serve (vLLM hoặc HF) khớp nhau. **Dung sai so với kỳ vọng đã làm tròn theo dtype thực tế** (vd bf16 lệch tới ~0.005 so với fp32), không dùng atol 1e-3 cố định. Bằng chứng lỗi: `benchmarks/sanity/parse-v1.1-normalize-26.08-1.json` (tester).
- Parse chạy trên `vlm-automodel:26.08-2` (đã vá). Mọi số chất lượng đo trên 26.08-1 là INVALID.
- Bài học (đưa vào skill research/tester): kiểm tiền xử lý bằng test số, không suy ra từ việc đọc config.
