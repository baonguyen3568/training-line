# P4-1 — Thư viện pipeline tham chiếu (phần 1: khảo sát + đề xuất chọn)

- **Owner:** bao-research · **Ngày:** 2026-10-13 · **Trạng thái:** DRAFT phần 1 (chờ PM/user chọn). Phần 2 = P4-2 (chuẩn mực thành dữ liệu).
- **Cách làm:** khảo sát 12 ứng viên từ nguồn chính chủ (repo, docs, model card; chỉ đọc). Research tự kiểm lại các điểm then chốt (đánh dấu ✔). Điểm nào chưa kiểm được ghi **UNVERIFIED**.
- **Tiêu chí xếp hạng:**
  1. chạy được trên 1 GB10 với ~35 GB trống;
  2. nguồn chính chủ hoặc đáng tin;
  3. có số công bố để tái lập;
  4. license rõ ràng, cho dùng thương mại;
  5. engine thuộc stack NVIDIA ta đã có.

## 1. Đề xuất 8 pipeline (theo thứ tự ưu tiên dựng)
| # | Pipeline (nguồn) | Engine | Model / data mặc định (license) | GB10? | Số công bố để tái lập | Node cần thêm |
|---|---|---|---|---|---|---|
| R0 | **Hoá đơn → JSON** = recipe NVIDIA `examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml` (AutoModel v0.6.0, Apache-2.0) | AutoModel | Nemotron-Parse v1.1 (NVIDIA OML) / CORD-v2 (CC-BY-4.0) | ✔ đã chạy: 19.8 GB, ~20 s/step | Không có số CORD công bố cho Parse. Mốc là số của ta (P-3b) + Donut (R6) | đã có. Thiếu `eval.benchmarks` thật (B-5) |
| R1 | **Gemma-4-E4B LoRA**, recipe AutoModel `gemma4/gemma4_4b_peft.yaml` (MedPix) + chạy lại trên CORD theo guide Gemma 4 | AutoModel | gemma-4-E4B-it (Apache-2.0) / CORD-v2 hoặc MedPix-VQA (thẻ HF ghi Apache-2.0, **dữ liệu y tế: cần security vet**) | Có (catalog: 28 GB ước tính) | ✔ Guide Gemma 4 dùng **31B** trên 8×A100: NED 0.885 → 0.0601, field acc 92.6%, 20 ảnh test. **Không có số cho E4B** ⇒ chỉ tái lập được *cách đo*; số 4B do ta tự đo | metric NED + field accuracy (thêm vào eval) |
| R2 | **Qwen3.5-4B SFT** `qwen3_5/qwen3_5_4b.yaml` (MedPix), có thể đổi sang LoRA | AutoModel | Qwen3.5-4B (Apache-2.0) / MedPix-VQA | LoRA có (24 GB ước tính). Full SFT 4B **không** | Không | — |
| R3 | **Nén FP8 + so điểm**: ModelOpt `examples/hf_ptq` với `--vlm` (0.47.0, Apache-2.0), áp lên model của R1/R2 | ModelOpt | — | Có (B-4 đang probe) | Support matrix: Qwen3-VL/Qwen3.5 và Gemma 3 **chỉ FP8**; Nemotron VL có FP8 + NVFP4. Bảng tham chiếu FP4 vs BF16: model card Nemotron-Nano-12B-v2-VL-NVFP4-QAD (AI2D 87.1/87.1, OCRBench 85.1/85.6, ChartQA 90.0/89.7, DocVQA 94.0/94.4). Tái lập được **phần eval** 2 checkpoint đó trên GB10, phần QAD thì không | `quantize.modelopt_ptq` (B-4), `eval.benchmarks` |
| R4 | **GRPO hình học**: NeMo-RL recipe `vlm_grpo-gemma4-e4b-geo3k` (v0.7.0, Apache-2.0) | NeMo-RL + vLLM | Gemma-4-E4B (Apache-2.0) / geometry3k (MIT) | **Chật**: recipe cho 1 node × 8 GPU; full-weight GRPO + vLLM vượt 35 GB ⇒ cần LoRA (NeMo-RL VLM LoRA: UNVERIFIED) | Ngưỡng CI: mean reward > 0.4 sau 20 step | `train.grpo_vlm`, node reward function (format / exact / math) |
| R5 | **DGX Spark playbook VLM fine-tuning** (build.nvidia.com/spark/vlm-finetuning): GRPO + LoRA | (không nêu tên) + vLLM | Qwen2.5-VL-7B (Apache-2.0) / Kaggle wildfire (**license UNVERIFIED**) | Playbook giả định có cả 128 GB và bảo `drop_caches`, không chấp nhận được trên máy dùng chung ⇒ cần thu nhỏ | Không có số accuracy. Thời gian: ~2 h / 100 step | như R4 + chat A/B base vs fine-tune |
| R6 | **Donut / CORD** (clovaai/donut, MIT; model donut-base MIT) | HF `VisionEncoderDecoderModel` | donut-base ~200M / CORD-v2 | Rất dễ | README: "CORD … 91.3" cho donut-base-finetuned-cord-v2 (**metric là TED acc hay F1: UNVERIFIED**, bài báo báo F1 ~84) | template "smoke nhanh" + mốc so sánh cho R0 |
| R7 | **NeMo Curator image pipeline** (v1.4.0, Apache-2.0): CLIP embed → aesthetic / NSFW filter → semantic dedup | Curator (Ray) | — | **arm64/GB10: UNVERIFIED** (README ghi GPU install "Linux x86_64") ⇒ cần chạy stack-verify trước | Không | `data.filter`, `data.dedupe`, `data.embed` |

## 2. Không chọn (và vì sao)
- **Qwen3-VL official `qwen-vl-finetune`**: pin torch 2.6 / transformers 4.57 / flash-attn 2.7.4, xung đột với image (torch 2.13, tf 5.12); flash-attn trên aarch64/sm_121 chưa rõ. Chỉ dùng làm tham chiếu định dạng data.
- **Cosmos-Reason2 post-training** ✔: repo "no longer under active development … limited maintenance", chuyển sang Cosmos 3 (github.com/NVIDIA/Cosmos). Bảng Tested Platforms chỉ có DGX Spark cho **inference**; cosmos-rl SFT cần 4×80 GB.
  - ⚠ **Ảnh hưởng tới template 2 (Cosmos-Reason2-2B)**: NVIDIA không hỗ trợ chính thức post-training trên Spark, và dòng model đã chuyển sang bảo trì hạn chế. Đề xuất PM/user cân nhắc lại template 2: giữ (ta fine-tune qua AutoModel LoRA, không qua cosmos-rl), hoặc chuyển sang Cosmos 3 khi đã kiểm license/kích thước.
- **HF TRL** (v1.15.0): portable nhưng không phải stack NVIDIA. Mặc định dùng Qwen2.5-VL-3B (license **qwen-research, phi thương mại**) và RLAIF-V (**CC-BY-NC**). Chỉ giữ làm engine dự phòng.
- **LLaMA-Factory**: tham chiếu UX (LlamaBoard). Data DPO demo (RLHF-V) là CC-BY-NC.
- **Unsloth, ms-swift**: chưa kiểm trên GB10.

## 3. License cần cờ đỏ (cho security)
Qwen2.5-VL-3B (qwen-research, phi thương mại); RLHF-V / RLAIF-V (CC-BY-NC-4.0); rdr-items, CLEVR-CoGenT bản HF, Kaggle wildfire: không ghi license; MedPix-VQA: Apache-2.0 theo thẻ HF nhưng là ảnh y khoa, cần xem nguồn gốc gốc.

## 4. Đề xuất thứ tự dựng (P4-5)
1. R0 (đã có) + R6 (mốc nhanh, cùng CORD) ⇒ có ngay cặp "model NVIDIA vs baseline cổ điển".
2. R3 (gắn với B-4) trên model R0/R1.
3. R1 (Gemma-4-E4B LoRA CORD, so với guide 31B về cách đo).
4. R2 (Qwen3.5-4B LoRA).
5. R4/R5 (GRPO) sau khi có node GRPO + reward và đo RAM thực tế.
6. R7 sau khi stack-verify Curator trên arm64.

## Nguồn (chính)
- AutoModel: https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune ; guide Gemma 4 ✔ https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/vlm/gemma4.md
- DGX Spark playbook: https://build.nvidia.com/spark/vlm-finetuning
- NeMo-RL: https://github.com/NVIDIA-NeMo/RL
- Cosmos-Reason2 ✔: https://github.com/nvidia-cosmos/cosmos-reason2
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- TRL: https://github.com/huggingface/trl/tree/main/examples ; LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- Curator: https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/image
- ModelOpt: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq ; NVFP4-QAD card: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD
- Donut: https://github.com/clovaai/donut
