# Proposal 024 — Thư viện 10 phương pháp: nguồn, dữ liệu, cấu hình và DoD phần research (ADR-022 luồng I)

- Tác giả: bao-research · 2026-10-14 · Trạng thái: DRAFT (PM chốt lịch GPU ở `.team/schedule/gpu-2026-10-14.md`)
- Dữ liệu gallery: `configs/reference_pipelines.yaml` schema v2 (10 method, 1 pipeline mỗi method). Spec node: P023 (cleancode).
- Nguồn đã kiểm tiêu đề trên arXiv ngày 2026-10-14.

## 1. Danh mục + nguồn

| # | method | pipeline id | nguồn recipe | bài báo | trạng thái hôm nay |
|---|---|---|---|---|---|
| 1 | full_sft | invoice-to-json | AutoModel `examples/vlm_finetune/nemotron` | — | đo thật (P-3b, F1 70.97) |
| 2 | lora | image-video-understanding | AutoModel `examples/vlm_finetune/qwen3_5` | LoRA 2106.09685 | đo thật (R2, F1 86.42) |
| 3 | qlora | qlora-4bit-qwen35-2b | AutoModel Spark QLoRA recipe | QLoRA 2305.14314 | đo 40 bước (RAM, val_loss), chưa có điểm |
| 4 | two_stage | two-stage-qwen35-2b | github.com/haotian-liu/LLaVA | LLaVA-1.5 2310.03744 (Eagle 2408.15998) | chờ 26.08-8 (patch 0007) |
| 5 | vision_unfrozen | vision-unfrozen-qwen35-2b | QwenLM/Qwen2.5-VL `qwen-vl-finetune` | Qwen2.5-VL 2502.13923, Qwen2-VL 2409.12191 | chờ node train_blocks + 26.08-8 |
| 6 | distill | distill-synthetic-qwen35 | ShareGPT4V | 2311.12793 | fixcode |
| 7 | dpo | dpo-qwen35 | NeMo-RL DPO | DPO 2305.18290; RLAIF-V 2405.17220; MPO 2411.10442 | cần vá DPO đa phương thức |
| 8 | grpo | grpo-geometry | NeMo-RL `examples/run_vlm_grpo.py` | GRPO (DeepSeekMath) 2402.03300; R1 2501.12948; VLM-R1 2504.07615 | probe 1 GPU đang chạy |
| 9 | ptq | quantize-fp8-benchmark | ModelOpt `examples/hf_ptq` | — | FP8 đo thật; NVFP4 slot #3 |
| 10 | qat | qat-qwen35 | ModelOpt `examples/llm_qat` | — | fixcode |

## 2. Lựa chọn dữ liệu (license sạch, không tải thêm khi có thể)
- **#4 hai giai đoạn: CORD v2 cho cả 2 GĐ** (CC BY 4.0, đã duyệt, có thước đo field F1 + R2 để so). GĐ1 = chỉ bộ nối trên cặp ảnh → toàn bộ
  nội dung hoá đơn (vai trò "mô tả dày" của LLaVA GĐ1); GĐ2 = bộ nối + LoRA trên đúng prompt của R2. Kết quả so trực tiếp với R2 (chỉ LoRA).
  Không dùng LLaVA-Pretrain (license "other", ảnh LAION/CC/SBU) hay ShareGPT4V (cc-by-nc-4.0). Ứng viên ảnh tài liệu mới nếu cần:
  allenai/pixmo-docs (ODC-BY, subset diagrams 3.1 GB) — cần security duyệt.
- **#5 mở vision: CORD v2** (so với R2 / P-3b). Pixel cap giữ 200704–1003520 cho lần đo đầu; "độ phân giải cao" = nâng max_pixels sau
  khi có số RAM.
- **#7 DPO: cặp dựng từ CORD** — chosen = nhãn thật, rejected = câu trả lời SAI của chính model (base hoặc sau LoRA) sinh bằng vLLM trên
  ảnh train. Đúng tinh thần RLAIF-V (cặp on-policy) mà không cần dataset preference có license NC. Ứng viên ngoài: OpenGVLab/MMPR-v1.2
  (MIT, 54 GB — chỉ khi cần), zhiqings/LLaVA-Human-Preference-10K (apache-2.0, ảnh COCO tải riêng). Loại: RLAIF-V-Dataset (cc-by-nc-4.0).
- **#8 GRPO: Geometry3K** (MIT, đã có trong catalog datasets, reward `math_expr` + `format` có sẵn trong NeMo-RL).

## 3. Cấu hình + rủi ro đã biết
- #8: NeMo-RL v0.7.0 chỉ có recipe VLM ≥ 2 GPU. 1 GPU = vLLM colocated (`gpu_memory_utilization` 0.15, sleep mode) + `cluster.gpus_per_node=1`.
  Đã vá/đặt: NVML không có bộ nhớ trên GB10 (patch NeMo-RL 0001, image -3); `policy.refit_buffer_size_gb` cố định (cudaMemGetInfo trên UMA
  báo free ~0.5 GB vì không tính page cache); `PYTORCH_CUDA_ALLOC_CONF` phải bỏ (CuMemAllocator từ chối expandable_segments).
- #7: NeMo-RL DPO chỉ text; AutoModel không có DPO. Đề xuất: loss DPO trên recipe VLM của AutoModel (policy = LoRA, reference = cùng model
  tắt adapter → không tốn thêm RAM), β mặc định 0.1, `sft_weight` tuỳ chọn (DPO+SFT). Chỉ gọi MPO nếu thêm số hạng quality (BCO).
- #5: `param_group_overrides` + `lr_scheduler {lr_decay_style: constant, lr_warmup_steps: 0}` (lr_mult chỉ có hiệu lực khi có scheduler).
- #9: ModelOpt NVFP4_DEFAULT_CFG (ptq.py `--format nvfp4`, parity chỉ đo, chưa có ngưỡng); câu hỏi chính: vLLM 0.31 có kernel CUTLASS FP4
  cho sm_121 không (chỉ biết khi chạy trên GPU).

## 4. DoD phần research (bổ sung ADR-022)
1. Mỗi pipeline research chạy thật ≥ 1 lần trên GB10, số vào `benchmarks/` (run.json peak, phút, điểm trên tập test cố định khi có).
2. #4 và #5 so cùng CORD test 100 ảnh với R2 (field F1, CI95 bootstrap) — kết luận "có lợi / không lợi" trung thực.
3. #8: reward trung bình tăng qua các bước + điểm Geometry3K test trước/sau (ít nhất 100 câu), RAM ≤ 80% RAM trống.
4. #7: tỉ lệ thắng chosen > rejected trên tập giữ lại + F1 CORD không giảm quá 1 điểm so với model trước DPO.
5. #9: NVFP4 load + chạy được trên vLLM GB10 (hoặc lý do cụ thể nếu không), delta điểm so với BF16/FP8 trên bộ documents.
