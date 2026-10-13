# ADR-001 — Amendment 2: template 2 chuyển sang Qwen3.5, Cosmos-Reason2 thành pipeline tham chiếu

- Trạng thái: **ACCEPTED** — 2026-10-13 (bao-pm chốt; user chọn template 2 = Qwen3.5)
- Nguồn: `.team/proposals/P4-1-reference-pipelines.md` §2 và §5 (khảo sát Cosmos-Reason2 / Cosmos 3), `.team/proposals/001-stack-model-catalog.md` (probe FP8 B-4).
- Sửa dòng "Template 2" trong bảng Quyết định của Amendment 1. Các phần khác giữ nguyên.

## Bối cảnh
- Repo `nvidia-cosmos/cosmos-reason2` ghi: "no longer under active development and will receive only limited maintenance updates". Repo trỏ người dùng sang Cosmos 3. Bảng Tested Platforms chỉ liệt kê DGX Spark cho **inference**; post-training (cosmos-rl SFT) cần 4×80 GB.
- Cosmos 3 (OpenMDW-1.1) là world model omnimodal. Recipe post-training chỉ có SFT trên 8×H100; card ghi "Only BF16 precision is tested". transformers 5.12 trong image chưa có `cosmos3_edge`. ⇒ Cosmos 3 không thay trực tiếp được template 2.
- Template 2 phải trình diễn đủ chuỗi train → nén → benchmark (ghi chú PM ở Amendment 1). Qwen3.5 có recipe AutoModel, FP8 là định dạng được ModelOpt hỗ trợ chính thức, và probe B-4 đã nén + xuất được Qwen3.5-0.8B bằng ModelOpt 0.44 có sẵn trong image.

## Quyết định
| Vai trò | Model@SHA | Ghi chú |
|---|---|---|
| **Template 2 "Hiểu ảnh/video"** | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` (catalog `tier: template`) | Apache-2.0, không gated, ảnh + video, LoRA (`train.lora_vlm`), FP8 chính thức. Bản 4B (`qwen3.5-4b`) là tuỳ chọn "lớn hơn" trong Advanced (RAM ước tính 24 GB, chưa đo) |
| Pipeline tham chiếu (P4-1, nhãn NVIDIA) | `nvidia/Cosmos-Reason2-2B@9ce19a19…` (catalog `tier: catalog`) | Gated, NVIDIA OML. "Built on NVIDIA Cosmos" chỉ hiện ở trang pipeline đó và trang Giấy phép (M2-8 §4.4). Không phải template |
| R&D | `nvidia/Cosmos3-Edge@344d602b…` (4B, OpenMDW-1.1) | Chỉ probe inference / "Chạy thử" trên GB10. Chưa vào catalog cho tới khi transformers/vLLM load được và security vet xong |

## Hệ quả
- Gate gallery cho template 2 (ADR-001 am.1, áp cho mọi template), cần chạy trên Qwen3.5-2B:
  - parity `pixel_values` và số image token / `grid_thw` train ↔ serve;
  - export giữ generation_config của base;
  - load report của export = load report của base (cẩn thận với các key `mtp.*`: merge LoRA bù từ base, `src/vlm_pipeline/train/merge_lora.py`).
- Smoke cần có trước khi vào gallery:
  - LoRA 1 epoch có validation (đang chờ giải quyết MEM-002 của smoke validation Qwen3.5-0.8B);
  - FP8 PTQ → vLLM 0.31 load được (fixcode, B-7);
  - benchmark trước/sau nén theo `configs/eval/gates.yaml`.
- RAM `qwen3.5-2b` trong catalog đang là **ước tính** (16 GB): phải đo trước khi hiện con số này trên card template.
- Docs/site: trang template ảnh/video viết theo Qwen3.5-2B. Trang "Giới hạn đã biết" đã ghi Cosmos Reason2 ở mức ứng viên.
