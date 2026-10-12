# Knowledge — Research (ML/VLM) · bao-research
Cập nhật: 2026-10-12. Nguồn ở cuối file. Ký hiệu: ✅ đã đo/xác minh trên máy này · 📄 xác minh từ nguồn chính chủ · ❓ chưa xác minh.

## 0. TL;DR cho cả team
1. **Ràng buộc số 1 là driver, không phải GPU**: host chạy driver **580.142 (CUDA 13.0)**. Image NGC mới nhất (26.06+) build cho CUDA 13.2–13.4 (driver 595–615). ✅ CUDA 13.0 chạy native, ✅ CUDA 13.1 chạy qua "Forward Compatibility mode", ❓ CUDA ≥13.2 có báo cáo **không chạy** trên 580 (forum #379168). Mọi tag phải smoke-test trước khi pin.
2. **Ràng buộc số 2 là RAM thực ~40GB** (UMA 121GB, ~77GB đang bị service khác dùng, swap đã dùng). Không phải 128GB.
3. Train mặc định: **NeMo AutoModel** (HF-native, recipe VLM có sẵn cho Qwen3.5/3.6/3.8, Gemma 4, Nemotron Omni). Megatron-Bridge chỉ là fallback.
4. Base model ứng viên: dev **Qwen3.5-2B** (CI: 0.8B), main **Qwen3.5-9B** (LoRA BF16 vừa 40GB), stretch **Qwen3.8-27B / Qwen3.6-27B** (QLoRA hoặc khi máy rảnh), NVIDIA-native **Nemotron 3 Nano Omni 30B-A3B** (~70GB BF16 → chỉ khi máy rảnh).
5. Quantize: **ModelOpt 0.47** `examples/hf_ptq/hf_ptq.py --vlm`; vision tower giữ BF16 mặc định. NVFP4 cho Qwen3.5/3-VL **chưa nằm trong ma trận chính thức** (chỉ FP8) → rủi ro.
6. Serve/eval: vLLM (OpenAI-compatible) là đường chung; NeMo Evaluator VLMEvalKit container **không có arm64** (26.03) → dùng lmms-eval 0.7.3 / VLMEvalKit pin commit trực tiếp.

## 1. Phần cứng & môi trường (✅ đo 2026-10-12)
| Mục | Giá trị |
|---|---|
| GPU | NVIDIA GB10, compute capability **(12,1)** = sm_121 |
| Driver / CUDA driver API | 580.142 / 13.0 |
| RAM | 121GB tổng, ~77GB used, ~44GB available, swap 8/15GB dùng |
| nvidia-smi memory | "Not Supported" (bình thường với iGPU) → đo bằng `/proc/meminfo` MemAvailable, `free -g` |
| Băng thông bộ nhớ | ~273GB/s (spec) → decode bị giới hạn băng thông → lợi thế cho MoE ít active + FP8/NVFP4 |
| Disk | 3.7TB, còn ~2TB (dùng chung) |
| Image có sẵn liên quan | pytorch:25.11-py3 (CUDA 13.0), pytorch:25.06, vllm:25.12.post1-py3 (13.1), vllm/vllm-openai:v0.27.1 (đang chạy cho service khác), nhiều NIM (không phải của dự án — không đụng) |
| HF cache global | Có sẵn `Qwen3.8-27B-NVFP4`, `Nemotron-3.5-Lightning-30B-A3B-NVFP4` (của người khác; chỉ đọc) |

### Ma trận image NGC arm64 ↔ CUDA ↔ driver baseline (✅ đọc config blob từ nvcr.io bằng `scripts/research/ngc_image_env.sh`)
| Image | CUDA | Driver build | Torch | Chạy trên 580? |
|---|---|---|---|---|
| nemo-automodel:25.11.00 | 13.0.1 | 580.82 | 2.9 | ✅ dự kiến native |
| **nemo-automodel:26.02** | 13.0.2 | 580.95 | 2.10 | ✅ dự kiến native |
| nemo-automodel:26.06 | 13.2.1 | 595.58 | 2.12 | ❓ rủi ro cao |
| nemo-automodel:26.08 (= v0.6.0) | 13.3.0 | 610.43 | 2.13 | ❓ rủi ro cao |
| nemo:26.08 | 13.3.0 | 610.43 | 2.13 | ❓ rủi ro cao |
| nemo-rl:v0.7.0 | 13.2.0 | 595.45 | — | ❓ rủi ro cao |
| pytorch:25.11-py3 | 13.0.2 | 580.95 | 2.10 | ✅ đã chạy (matmul+SDPA; Triton 3.5.0 @jit + torch.compile PASS — bao-fixcode) |
| pytorch:26.01-py3 | 13.1.1 | 590.48 | 2.10 | ✅ dự kiến (forward-compat) |
| vllm:25.12.post1-py3 | 13.1.0 | 590.44 | 2.10 | ✅ đã chạy (forward-compat) |
| vllm:26.01-py3 (vLLM 0.13) | 13.1.1 | 590.48 | — | ✅ dự kiến |
| vllm:26.09-py3 (vLLM 0.29) | 13.4.1 | 615.71 | — | ❓ rủi ro cao |
| tensorrt-llm/release:1.2.1 | 13.1.0 | 590.44 | 2.10 | ✅ dự kiến |
| tensorrt-llm/release:1.3.0rc13 (playbook) | 13.1.1 | 590.48 | 2.11 | ✅ dự kiến |
| tensorrt-llm/release:spark-single-gpu-dev | 12.9.1 | 575 | 2.8 | ✅ (cũ) |
| eval-factory/vlmevalkit:26.03 | — | — | — | ❌ **không có arm64** |

## 2. Stack NVIDIA — trạng thái (📄 trừ khi ghi khác)
### NeMo AutoModel (train chính)
- v0.6.0 (26/08/2026) ≈ container 26.08; v0.5.0 (02/07) thêm **QLoRA cho VLM**; v0.4.0 (28/04).
- Recipe VLM ✅ xác minh trên GitHub `examples/vlm_finetune/`: `qwen3_5/{qwen3_5_4b,qwen3_5_9b,qwen3_6_27b,qwen3_6_27b_lora}.yaml`, `qwen3_8/{qwen3_8_27b,qwen3_8_27b_lora}.yaml`, `nemotron_omni/{nemotron_omni_cord_v2,_peft}.yaml`, `gemma4/`, `internvl/`, `nemotron_3_5_super_vl/`.
- Recipe dùng `recipe: FinetuneRecipeForVLM`, model `NeMoAutoModelForImageTextToText.from_pretrained`, FSDP2Manager (tp=cp=pp=1 OK cho 1 GPU), `freeze_config` (vision tower/LM), `peft: PeftConfig(dim, alpha, exclude_modules=[*vision*...])`, loss `FusedLinearCrossEntropy` (tránh OOM logits), `attn: sdpa`, `activation_checkpointing: true`.
- Lệnh: `automodel <yaml> --nproc-per-node 1` (hoặc `torchrun ... examples/vlm_finetune/finetune.py -c <yaml>`). Extras: `vlm`, `vlm-media`.
- Dataset: list messages `{"role":"user","content":[{"type":"image","image":...},{"type":"text","text":...}]}` → `apply_chat_template`; custom dataset qua `_target_: /abs/path.py:build_fn`; collate riêng theo model (`collate_fns.py`).
- **Không có** recipe VLM `*_spark.yaml`; chỉ LLM có (`llm_finetune/*_rtx_spark.yaml`) → ta tự viết recipe Spark cho VLM (activation ckpt, local_batch 1, max_pixels thấp).
- Playbook Spark nemo-fine-tune: container 26.08 + clone repo tại commit `3b5663a6fdf0fee03abbe82e431ca3a7ba6dd8e2`, `docker run --gpus all --ipc=host --memory=64g ...`. Số đo (bao-backend dẫn): Qwen3-8B LoRA ≈ 20.8GiB, Qwen3-14B LoRA ≈ 43GiB, 8B full SFT ≈ 78GiB.
### Megatron-Bridge (fallback)
- v0.6.2 (18/09). VLM: Nemotron 3 Nano Omni, Gemma 4, Qwen3-Omni-MoE, Qwen3-VL... Nemotron Nano 12B v2 VL **sắp bị gỡ ở v0.7.0**, recipe chỉ multi-GPU (TP2/TP4). Không có container riêng — nằm trong `nemo:26.xx` (CUDA 13.3 → rủi ro driver).
### NeMo-RL (GRPO)
- v0.7.0 (29/07), image arm64 nhưng CUDA 13.2. VLM GRPO có từ v0.4 (`examples/run_vlm_grpo.py`). 1-GPU GB10 cho VLM: ❓. VLM DPO: ❓.
- Fallback RL: TRL (GRPO/DPO) trong pytorch:25.11 — playbook vlm-finetuning của NVIDIA dùng container riêng + LoRA + GRPO cho Qwen2.5-VL-7B (~2h/100 step).
### ModelOpt (quantize)
- 0.47.0 (23/09); `examples/vlm_ptq` đã bị gộp vào `examples/hf_ptq/hf_ptq.py --vlm --calib_with_images --calib_size 512`.
- Ma trận VLM: Qwen2/2.5-VL FP8/INT4-AWQ/W4A8/NVFP4; Nemotron VL FP8/NVFP4; **Qwen3-VL/Qwen3.5 chỉ FP8** (+ recipe FP8 vision tùy chọn); Gemma3 FP8.
- Export unified HF checkpoint → vLLM / TRT-LLM (PyTorch backend) / SGLang. QAT/QAD có (0.47 có ví dụ VLM QAD).
- Lưu ý: NVIDIA đã phát hành sẵn `nvidia/Qwen3.6-27B-NVFP4`, `nvidia/Qwen3.8-27B-NVFP4`, `nvidia/Gemma-4-31B-IT-NVFP4` → dùng làm **reference** để so NVFP4 tự làm.
### TensorRT-LLM
- Stable 1.2.1 (CUDA 13.1). Spark single-node: beta. Multimodal list: Qwen2/2.5-VL, Gemma3, Mistral 3.1, Phi-4-mm, InternVL2, LLaVA-OV; Qwen3.5 class có trong supported-models ❓ cho 3.6/3.8.
### vLLM
- vLLM upstream 0.31; NGC 26.09 = 0.29 (CUDA 13.4 ❓driver), 26.01 = 0.13 (cũ, có thể chưa hỗ trợ Qwen3.5). Image `vllm/vllm-openai:v0.27.1` (CUDA 13.0) đang chạy được trên máy → **ứng viên serve thực tế** (cần xác nhận hỗ trợ Qwen3.5 + vá CVE-2026-27893/22807 theo bao-security).
- Spark: đặt `--gpu-memory-utilization` 0.3–0.5 trên máy chung (mặc định ~0.9+ sẽ OOM cả host), `--max-num-seqs` nhỏ, NVFP4 trên sm_121 chạy được (FlashInfer CUTLASS), JIT warm-up ~25s lần đầu.
### Eval
- NeMo Evaluator 0.3.0 / launcher 0.2.6 — VLM chỉ qua container VLMEvalKit **x86-only** → không dùng trực tiếp.
- lmms-eval 0.7.3 (`--model openai` cho endpoint, `--model vllm`/hf in-process). VLMEvalKit: không có release → pin commit; tự nhận "không tái lập số paper".
### Data
- NeMo Curator 1.4.0 / container 26.07: image+video curation chạy arm64 (audio không). Energon 7.4.1 (WebDataset). Với quy mô dự án (≤ vài trăm nghìn mẫu) **HF datasets + JSONL là đủ**; Curator dùng cho dedup/filter ảnh nếu cần.

## 3. Base model — so sánh (📄 model card; điểm theo card, chế độ thinking nếu card mặc định)
| Model | Params | License | MMMU | MMBench | OCRBench | MathVista | AI2D | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 0.8B dense hybrid (Gated DeltaNet) | Apache-2.0 | 49.0 | 69.9 | 74.5 | 62.2 | 69.9 | CI |
| Qwen/Qwen3.5-2B | 2B | Apache-2.0 | 64.2 | 83.3 | 84.5 | 76.7 | 83.3 | dev/smoke |
| Qwen/Qwen3.5-4B | ~5B | Apache-2.0 | 77.6 | 89.4 | 85.0 | 85.1 | 89.6 | recipe sẵn |
| Qwen/Qwen3.5-9B | ~10B | Apache-2.0 | 78.4 | 90.1 | 89.2 | 85.7 | 90.2 | recipe sẵn; main |
| Qwen/Qwen3.6-27B | 27B dense | Apache-2.0 | 82.9 | 92.3 | 89.4 | 87.4 | — | FP8/NVFP4 chính chủ |
| Qwen/Qwen3.6-35B-A3B | 35B/3B MoE | Apache-2.0 | 81.7 | 92.8 | — | 86.4 | 92.7 | ~70GB BF16 |
| Qwen/Qwen3.8-27B (08/2026) | 27B dense | Apache-2.0 | — | — | — | — | — | card thiếu benchmark chuẩn; recipe AutoModel ✅ |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning | ~31B/3B hybrid Mamba MoE | NVIDIA Open Model | — | — | V2-EN 67.0 | 82.8 | — | image+video+audio; ~70GB BF16 |
| nvidia/Nemotron-Nano-12B-v2-VL | 12.6B | NVIDIA Open Model | 68 | — | 85.6 | 76.9 | 87.1 | DocVQA 94.4, ChartQA 89.7; đang bị deprecate |
| google/gemma-4-* | E2B…31B | cần security kiểm | MMMU-Pro only | | | | | recipe sẵn |
Loại: Mistral Small 4 (119B, quá lớn), InternVL3.5 (lạc hậu so với Qwen3.5), Qwen3-VL (bị Qwen3.5 thay thế), Cosmos-Reason2 (chuyên physical AI).
⚠️ Qwen3.5 card không ghi DocVQA/ChartQA và mặc định **thinking mode** (~32K output) → khi tái lập phải bật thinking + max_new_tokens lớn, hoặc đo non-thinking và **không** so trực tiếp với card.

## 4. Best practices (recipe)
- LoRA BF16 trên LLM, **đóng băng vision tower** (recipe chính chủ cũng exclude `*vision*`), r 8–32, alpha 2–4×r, lr 1e-4 (LoRA) / 1e-5 (full), AdamW β(0.9,0.95), activation checkpointing, `FusedLinearCrossEntropy`, local_batch 1 + grad accumulation, seed cố định.
- Giới hạn `max_pixels` (ví dụ 512×28×28 cho Qwen) và số frame video — token ảnh là thứ ăn bộ nhớ nhiều nhất.
- Luôn giữ held-out split + chạy general benchmark để đo **catastrophic forgetting** sau fine-tune.
- RL (GRPO) chỉ sau SFT, với reward kiểm chứng được; bắt đầu trên model 2B.
- Không train trên dữ liệu benchmark; dataset sinh từ GPT-4 (LLaVA-Instruct, ShareGPT4V) có rủi ro license/ToS (bao-security).

## 5. Pitfalls GB10 / UMA
- `docker --memory` **không chặn** cấp phát CUDA trên UMA → OOM có thể treo cả host (forum). Guard trong code: kiểm `MemAvailable` trước khi launch, abort nếu < ngưỡng.
- `cudaMemGetInfo` báo thấp do page cache; NVIDIA khuyên `drop_caches` (cần sudo → hỏi PM).
- Load model từ pageable memory chậm bất thường (nhiều H2D copy nhỏ) → ưu tiên safetensors + load thẳng device.
- flash-attn không có wheel sm_121 aarch64 → **SDPA mặc định**; cuDNN SDPA có báo cáo lỗi trên SM121.
- NVFP4 không mặc nhiên nhanh hơn FP8 trên GB10 → luôn benchmark cả hai.
- Pin `transformers` theo `config.json` của model (Qwen3.5 cần transformers 5.x).
- Docs "latest" của NVIDIA thường trễ so với "nightly"/GitHub → đối chiếu cả hai.

## 6. Checklist áp dụng cho dự án
- [ ] Mỗi image: chạy `scripts/research/ngc_image_env.sh` + probe GPU trước khi pin; pin digest arm64.
- [ ] Mỗi model: repo chính chủ + SHA 40 ký tự + license gốc + safetensors + trust_remote_code=False (bao-security).
- [ ] Mỗi job nặng: GPU_LOCK + kiểm MemAvailable ≥ budget + 10GB headroom.
- [ ] Mỗi kết quả: theo protocol 002 (gate reproduce rule-scored ±1.5pt; judge-based chỉ so nội bộ).
- [ ] Mỗi download >20GB: báo PM; báo bao-cleanup repo/rev/size.

## Nguồn
- NeMo AutoModel releases: https://github.com/NVIDIA-NeMo/Automodel/releases · release log: https://docs.nvidia.com/nemo/automodel/nightly/whats-new/model-release-log.html · recipes: https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune · dataset: https://docs.nvidia.com/nemo/automodel/datasets/multi-modal-dataset.md
- NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel · https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags · registry nvcr.io (đọc trực tiếp config blob)
- Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases · https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron-nano-v2-vl.html
- NeMo-RL: https://github.com/NVIDIA-NeMo/RL/releases
- ModelOpt: https://github.com/NVIDIA/Model-Optimizer/releases · https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq
- TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/latest/release-notes.html · https://nvidia.github.io/TensorRT-LLM/models/supported-models.html
- vLLM: https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-08.html · https://vllm.ai/blog/2026-06-01-vllm-dgx-spark · https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B
- NeMo Evaluator: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks · VLMEvalKit: https://github.com/open-compass/VLMEvalKit · lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
- NeMo Curator: https://docs.nvidia.com/nemo/curator/latest/reference/infra/container-environments
- DGX Spark playbooks: https://build.nvidia.com/spark/vlm-finetuning · https://build.nvidia.com/spark/nemo-fine-tune/instructions · https://build.nvidia.com/spark/nvfp4-quantization/instructions · https://build.nvidia.com/spark/trt-llm/instructions · https://build.nvidia.com/spark/vllm/instructions
- GB10 issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html · https://forums.developer.nvidia.com/t/vllm-26-04-py3-compatibility-mode-is-unavailable-on-driver-580-173-02-no-dgx-dashboard-path-to-595-58/379168 · https://forums.developer.nvidia.com/t/urgent-dgx-spark-gb10-uma-nim-container-memory-runaway-all-limitation-parameters-invalid-memory-occupies-120gb/371180/1 · https://forums.developer.nvidia.com/t/dgx-spark-arm64-cuda-13-pathological-slowdown-for-many-small-h2d-copies-from-pageable-cpu-memory-50x-vs-pinned-impacts-pytorch-model-load-patt/354506
- Model cards: https://huggingface.co/Qwen/Qwen3.5-2B · https://huggingface.co/Qwen/Qwen3.5-9B · https://huggingface.co/Qwen/Qwen3.6-27B · https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://huggingface.co/Qwen/Qwen3.8-27B · https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 · https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 · https://huggingface.co/google/gemma-4-31B-it
