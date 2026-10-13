# M2-2 — Kế hoạch nội dung kỹ thuật cho docs (owner nội dung: bao-research)

Nguyên tắc (PM, BOARD M2):
- Chỉ dùng số đo thật, mỗi số có nguồn (`sources:` trong frontmatter).
- Số chưa đo thì ghi "chưa đo", không ước đoán.
- Phần giải thích phụ để trong `<Hint>` (popover "?").
- Tiếng Việt có dấu. Thuật ngữ gốc (LoRA, FP8, F1…) giữ nguyên, giải thích ở Hint.

## Danh sách trang docs (đề xuất 11 trang)
| # | slug | Nội dung chính | Nguồn sự thật |
|---|---|---|---|
| 1 | `bat-dau-nhanh` | Mở app → chọn template "Hoá đơn → JSON" → Kiểm tra → Chạy → xem kết quả, chạy thử | configs/workflows, ADR-001 am.1, app M1 |
| 2 | `node-va-cong` | Node là gì; 12 loại node theo nhóm (data/model/train/eval/quantize/serve/export/bench); cổng có kiểu (Dataset, Model, Adapter, Checkpoint…); vì sao dây bị từ chối | src/vlm_pipeline/nodes/*.py |
| 3 | `kiem-tra-truoc-khi-chay` | Preflight: RAM (ram_gb_train + 10 GB dự phòng so với RAM trống), disk, GPU lock, mã lỗi CFG/MEM dễ hiểu | ADR-005, error_catalog |
| 4 | `catalog-model` | Bảng 11 model: hãng, license, thương mại, RAM train (đo/ước tính), cách train (SFT/LoRA), FP8, gated | configs/catalog/models.yaml |
| 5 | `template-hoa-don` | Parse v1.1 + CORD v2: dữ liệu, thời lượng, RAM đo được, decoding, điểm (khi có số chấm P-3b) | 001a, benchmarks/ |
| 6 | `template-anh-video` | Cosmos Reason2 2B (gated, phải chấp nhận điều khoản trên HF), Qwen3-VL làm baseline; trạng thái | 001, catalog |
| 7 | `template-nen-benchmark` | Quantize FP8 (ModelOpt) + benchmark; FP8 official/experimental/unsupported theo model; trạng thái thật | 001 §, catalog quant_fp8 |
| 8 | `doc-diem-benchmark` | CORD: field F1 (chính), nTED, JSON exact, parse_fail, truncated; benchmark chung (MMBench, OCRBench, ChartQA eval-only…); cách so với số công bố (protocol, seed, split) | P002, scripts eval |
| 9 | `phan-cung-va-bo-nho` | DGX Spark GB10, aarch64, 128 GB unified memory dùng chung; vì sao RAM thật trống ít hơn; driver/image | knowledge/research.md, ADR |
| 10 | `gioi-han-da-biet` | Danh sách giới hạn hiện tại, kèm trạng thái | 001a §10.x, bugs/ |
| 11 | `faq-ky-thuat` | Câu hỏi kỹ thuật thường gặp | tổng hợp |

## Số liệu đã đo có thể dùng (cập nhật khi có thêm)
- Parse v1.1 full-SFT trên GB10: RAM đỉnh 18.9 GB (P-3, bf16) / 19.8 GB (P-3b-A, FusedAdam master) → catalog 20 GB. ~20 s/step GBS 8.
- P-3b-A: val_loss 0.084 sau 1 epoch (800 ảnh). **Chưa có điểm F1**: chờ gate tester, rồi số test 100 ảnh.
- Checkpoint Parse full: ~8.1 GB/ckpt (model 5.4 + optim 2.7) — cleanup đo.
- Tốc độ chấm CORD (HF remote, batch 1): ~7 ảnh/phút.
- S-3 Qwen3.5-0.8B LoRA: benchmarks/perf/s3-lora-qwen3.5-0.8b-20261012.json.

## Giới hạn đã biết (cho trang 10) — kiểm lại trạng thái trước khi publish
- Parse: dừng rồi chạy tiếp (resume) chưa dùng được (bug checkpoint shard).
- Parse: nén FP8 chưa chạy thật (P-5 chưa làm); serve vLLM chưa kiểm (P-4).
- Parse "Chạy thử": tốc độ ~8–9 s/ảnh (không dùng vLLM).
- Parse chỉ fine-tune toàn phần (SFT), không LoRA (CFG-010 đang làm).
- Cosmos: gated, cần user chấp nhận điều khoản; template chưa chạy thật.
- Gemma 4: không hỗ trợ FP8.
- Không resume giữa chừng cho run đã xong (optim bị strip theo D3).
- MMBench v1.1 chưa tải được (host timeout); G1 hoãn.

## Việc cần cập nhật docs khi tính năng xong (theo dõi)
- [ ] Backend: export kèm LICENSE + NOTICE/ATTRIBUTION. Khi xong, sửa `faq-ky-thuat` (mục Giấy phép) và `catalog-model` từ "user phải tự giữ kèm" thành mô tả đúng.
- [ ] Parse resume sửa xong: sửa `gioi-han-da-biet` + `faq-ky-thuat`.
- [ ] Điểm P-3b (val gate → test 100 ảnh): viết `template-hoa-don`, cập nhật `gioi-han-da-biet`.
- [ ] Kết luận repetition_penalty trên validation: cập nhật FAQ.
- [ ] Template còn lại chuyển sang FusedAdam: cập nhật `phan-cung-va-bo-nho`.
- [ ] OML §3.2: trang docs/site nói về template Cosmos phải có câu "Built on NVIDIA Cosmos" (chờ security chốt cách hiển thị, M2-8).
