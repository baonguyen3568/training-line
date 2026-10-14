# ADR-022 — Thư viện 10 pipeline theo PHƯƠNG PHÁP train

- Trạng thái: **ACCEPTED (USER)** — 2026-10-14
- Bối cảnh: user: "tôi tưởng ta sẽ có tầm 10 pipeline mỗi pipeline là 1 phương pháp + setup + config + kiến trúc train gì đó mới nổi
  gần đây hoặc được dùng rộng rãi, chứ của bạn nó toàn hiểu ảnh rồi v.v.v". Gallery hiện (configs/reference_pipelines.yaml)
  chia theo bài toán/model; chỉ có 5 phương pháp thật (full SFT, LoRA, QLoRA, FP8 PTQ, làm sạch data).
- Thay thế phần "5–10 pipeline tham chiếu" trong CLAUDE.md về cách tổ chức: trục chính là PHƯƠNG PHÁP; model/data là tham số
  đổi được ở bước nguồn (ADR-021).

## Quyết định — đủ 10 trong Release 1.0
| # | Phương pháp | Kiến trúc train (đóng băng / train) | Nguồn tham chiếu |
|---|---|---|---|
| 1 | Full SFT | vision đóng băng, LM train toàn bộ | NeMo AutoModel, Nemotron |
| 2 | LoRA SFT | LoRA trên LM, vision đóng băng | HF PEFT |
| 3 | QLoRA 4-bit | base NF4 + LoRA, vision/lm_head bf16 | QLoRA (Dettmers 2023), patch 0006 |
| 4 | Train 2 giai đoạn kiểu LLaVA | GĐ1: chỉ projector trên caption; GĐ2: projector + LM (hoặc LoRA) trên instruction | LLaVA-1.5, NVIDIA Eagle |
| 5 | Mở vision encoder + LR theo phần + ảnh độ phân giải cao | ViT LR thấp, projector/LM LR cao | Qwen2.5-VL / Qwen3-VL fine-tune |
| 6 | Distillation bằng data tổng hợp | teacher VLM lớn (vLLM) sinh QA/caption → SFT/LoRA student nhỏ | ShareGPT4V-style |
| 7 | DPO / MPO (preference) | policy + reference, cặp chosen/rejected | InternVL MPO, RLAIF-V, NeMo-RL DPO |
| 8 | GRPO / RL reward kiểm chứng được | policy + rollout (vLLM) cùng 1 GPU | DeepSeek-R1 → VLM-R1, DGX Spark playbook, NeMo-RL |
| 9 | PTQ FP8 / NVFP4 + đo lại điểm | không train; ModelOpt | NVIDIA ModelOpt |
| 10 | QAT (train có lượng tử) | fine-tune ngắn với fake-quant để lấy lại điểm | NVIDIA ModelOpt QAT |

Làm sạch dữ liệu (lọc, bỏ trùng, kiểm rò train/test) KHÔNG còn là pipeline riêng: thành **nhóm bước dựng sẵn** dùng chung (P4-4).

## Yêu cầu mỗi pipeline (Definition of Done)
1. Dựng bằng node của app; mở ra chạy được; đổi model/data ở bước nguồn.
2. Thẻ gallery: tên phương pháp, 1 câu "khi nào dùng", sơ đồ kiến trúc train (khối nào đóng băng / train), nguồn + license, độ khó, thời gian + RAM đo thật.
3. Chạy thật ≥1 lần trên GB10 (≤80% RAM trống), có số đo; nếu nguồn có số công bố tái lập được thì so.
4. check_support chặn đúng tổ hợp không hỗ trợ (validate == worker), câu lỗi dễ hiểu.
5. GRPO và DPO/MPO: KHÔNG còn phương án "sắp có" cho 1.0. Engine thiếu cấu hình 1 GPU thì team tự cấu hình (vLLM colocated / chia RAM),
   tự vá nếu cần (quy trình patch bên thứ 3), vẫn qua review security.

## Phân công (luồng I)
- research (lead): danh mục + nguồn + DoD; #4, #5, #8 (tiếp probe GRPO, bỏ time box); kiểm NVFP4 trên GB10 (#9).
- fixcode: #6 distillation (vLLM teacher → caption_vlm/QA → SFT student); #10 QAT (ModelOpt).
- research + backend (khi rảnh H): #7 DPO/MPO trên NeMo-RL (dùng chung hạ tầng với #8).
- cleancode: spec node mới (stage/freeze theo khối, LR theo khối, train.dpo, quantize.qat, distill) + tổ chức gallery theo phương pháp.
- design + frontend: thẻ gallery theo phương pháp + sơ đồ kiến trúc train (icon-first, popover ?).
- tester: mỗi pipeline có test validate + 1 lượt chạy thật; số đo ghi benchmarks/.
- security: review image/deps mới (NeMo-RL, ModelOpt QAT), patch nếu có.

## Hệ quả
- 1.0 lùi thêm khoảng 2–3 ngày, chạy song song luồng H (ADR-021). GPU xếp hàng qua lock; PM điều phối thứ tự.
- Site/docs: trang pipeline tham chiếu viết lại theo phương pháp.
