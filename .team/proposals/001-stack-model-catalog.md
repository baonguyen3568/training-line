# Proposal 001: Engine stack NVIDIA + catalog model được hỗ trợ (bản sau RESET)
Tác giả: bao-research · 2026-10-12 · Trạng thái: **REVIEW**
Reviewer: bao-backend (engine/image), bao-security (license/remote code), bao-cleanup (dung lượng), bao-tester (eval), bao-pm.
Chi tiết bằng chứng ở `.team/knowledge/research.md`. ✅ = đã đo/xác minh · ❓ = chưa xác minh.

## 0. Ảnh hưởng của RESET (R-4)
- Trước RESET: chọn **1** base model + 1 recipe. Sau RESET: app no-code cho user **tự chọn model**, nên 001 định nghĩa:
  (1) **engine** chạy phía sau mỗi loại node, (2) **catalog model** có metadata để UI chặn lỗi trước khi chạy (RAM, license, quant được), (3) **recipe mặc định** cho template.
- Catalog là **dữ liệu** (`configs/catalog/models.yaml`), không hard-code. Thêm model = thêm 1 entry + được security vet.

## 1. Ràng buộc nền (đã chốt với user qua PM)
- Driver **580.142** được giữ nguyên, nên **mọi engine image phải là CUDA ≤ 13.1** (13.0 chạy native, 13.1 chạy forward-compat, cả hai ✅ đo).
  Phương án nâng cấp (b'): cập nhật DGX OS chính chủ lên driver 580.159.03 để mở khoá image 26.08 (❓ cần probe). User quyết khi cần.
- RAM thực cho job ~**40GB** (máy dùng chung). Ước lượng RAM trong catalog tính theo con số này.
- Không tải gì cho tới khi PM mở lại (HF token đang 401).

## 2. Engine theo loại node
| Node | Engine mặc định | Image (arm64, CUDA) | Fallback / ghi chú |
|---|---|---|---|
| Train SFT/LoRA | **NeMo AutoModel v0.4.0** (`FinetuneRecipeForVLM`) | `vlm-automodel` FROM `nvcr.io/nvidia/nemo-automodel:26.02` (13.0, digest arm64 `sha256:6ee69689…`) + `nemo_automodel==0.4.0 --no-deps` + `transformers==5.5.0`, torch giữ nguyên 2.10.0a0 | Engine thứ 2 (**sau v1**, cần security duyệt + tự build image arm64): **LLaMA-Factory 0.9.5** (Apache-2.0, có playbook Spark) build FROM `pytorch:25.11-py3` (13.0 ✅ chạy, Triton ✅). Megatron-Bridge: chưa dùng (cần nemo:26.08 CUDA 13.3) |
| Train QLoRA | AutoModel `qlora.py` (bitsandbytes) | như trên | ❓ kernel bnb trên sm_121. Chỉ cần cho model ≥27B; gắn cờ "thử nghiệm" |
| RL (GRPO/DPO) | Giai đoạn 2. DPO qua LLaMA-Factory; GRPO qua TRL | pytorch:25.11 | NeMo-RL v0.7.0 là CUDA 13.2 ❌ với driver hiện tại |
| Chạy thử / serve | **vLLM v0.31.0** (đã vá RCE GHSA-h3rc-6mm3-gc2m) | `vllm/vllm-openai:v0.31.0` arm64 `sha256:3f7dd5b7…`, `cuda>=13.0` (security APPROVED, ngoại lệ D1 ngoài NGC; bind 127.0.0.1) | TRT-LLM 1.2.1 (13.1) cho perf. `--gpu-memory-utilization` ≤0.4 trên máy chung |
| Quantize | **ModelOpt 0.47** `examples/hf_ptq/hf_ptq.py --vlm` | cài vào `vlm-automodel` hoặc pytorch:25.11 ❓ test | Qwen3.5/3-VL: **chỉ FP8 chính thức**; NVFP4/INT4 chính thức cho Qwen2.5-VL, Nemotron VL. Vision tower giữ BF16 |
| Benchmark accuracy | **lmms-eval 0.7.3** gọi endpoint vLLM (+ in-process HF cho gate reproduce) | host venv hoặc pytorch:25.11 | VLMEvalKit pin commit. NeMo Evaluator VLM container **không có arm64** ❌ |
| Benchmark perf | vLLM bench / script riêng (P002, bao-tester) | như serve | — |
| Data | HF `datasets` + JSONL chat messages | host/pytorch:25.11 | NeMo Curator 26.07 (arm64 image+video) cho dedup/filter. Để giai đoạn 2 |
| Export | HF safetensors (merge LoRA) / unified ckpt của ModelOpt | — | — |

Format dữ liệu chuẩn nội bộ: messages `[{role, content:[{type:image|video|text,...}]}]` (giống AutoModel/HF). Converter sang `dataset_info` của LLaMA-Factory nằm trong engine adapter.

## 3. Catalog model v1
RAM ước tính = weights BF16 + LoRA/optimizer + activation với `max_pixels` mặc định và grad checkpointing. Đây là **ước lượng, phải hiệu chỉnh bằng smoke run**.
Mốc tham chiếu: Qwen3-8B LoRA đo được ≈ 21GiB.

| Tier | Model (repo@SHA do security vet) | Params | License | Remote code | Disk | RAM LoRA BF16 (ước) | Engine train | Quant được (chính thức) | Trong app v1 |
|---|---|---|---|---|---|---|---|---|---|
| CI | Qwen/Qwen3.5-0.8B | 0.8B | Apache-2.0 | Không | ~2GB | ~6GB | AutoModel (YAML tự viết từ qwen3_5_4b) | FP8 | Ẩn (chỉ CI) |
| **Dev/Template** | **Qwen/Qwen3.5-2B** | 2B | Apache-2.0 | Không | ~5GB | ~10GB (full SFT ~25GB) | AutoModel | FP8 | ✅ mặc định template |
| Main | Qwen/Qwen3.5-4B@851bf6e8 | ~5B | Apache-2.0 | Không | 9.3GB | ~16GB | AutoModel (`qwen3_5_4b.yaml`) | FP8 | ✅ |
| **Main** | **Qwen/Qwen3.5-9B@c2022362** | ~10B | Apache-2.0 | Không | 19.3GB | ~28GB | AutoModel (`qwen3_5_9b.yaml`) | FP8 | ✅ "khuyên dùng" |
| Stretch | Qwen/Qwen3.6-27B@6a9e13bd | 27B | Apache-2.0 | Không | 55.6GB | ~65GB BF16 ❌ / QLoRA ~35GB ❓ | AutoModel (cùng kiến trúc 3.5 ❓) | FP8 (+`nvidia/…-NVFP4` có sẵn) | ⚠️ "cần máy rảnh", ẩn mặc định |
| Stretch | Qwen/Qwen3.8-27B@1d4bf0f2 | 27B | Apache-2.0 | Không | 55.6GB | như trên | AutoModel ≥v0.5 ❓ (cần b') | FP8, NVFP4 (nvidia/) | ⚠️ sau b' |
| Stretch | Qwen/Qwen3.6-35B-A3B@995ad96e | 35B/3B MoE | Apache-2.0 | Không | 71.9GB | ~80GB ❌ | AutoModel qwen3_5_moe | FP8 | ❌ v1 |
| Alt | google/gemma-4-{E2B,E4B}-it | 2–8B | Apache-2.0 (security: commercial=yes; hiện Prohibited Use Policy như khuyến nghị) | Không | ~5–16GB | ~10–20GB | AutoModel (`gemma4_*_peft.yaml`) | FP8 ❓ | ✅ (sau khi pin SHA) |
| R&D | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e99324 | ~31B/3B | NVIDIA Open Model Agreement | **Có, bắc cầu 2 repo (C-RADIOv4-H)** | 66GB | ~75GB ❌ | AutoModel ≥v0.5 (cần b') | FP8/NVFP4 (nvidia/) | ❌ chỉ R&D (security: approved-with-conditions) |
Loại khỏi catalog: Mistral Small 4 (119B), InternVL3.5 (lạc hậu), Qwen3-VL (đã bị Qwen3.5 thay thế, nhưng vẫn là fallback vì 26.02 gốc hỗ trợ native), Nemotron Nano 12B v2 VL (đang bị deprecate, chỉ có recipe multi-GPU).

Schema entry (đề xuất cho `configs/catalog/models.yaml`):
```yaml
- id: qwen3.5-9b
  repo: Qwen/Qwen3.5-9B
  revision: c2022362...        # 40-char SHA, security-vetted
  license: {spdx: Apache-2.0, commercial: true, attribution: null}
  remote_code: false
  modalities: [image, video, text]
  disk_gb: 19.3
  ram_gb: {lora_bf16: 28, qlora: null, full_sft: null, serve_bf16: 24, serve_fp8: 14}   # calibrate by smoke run
  engines: {train: [automodel], serve: [vllm], quantize: [modelopt]}
  quant_formats: {fp8: official, nvfp4: experimental, int4_awq: unsupported}
  recipe: configs/train/qwen3_5_9b_lora_spark.yaml
  published_scores: {mmmu_val: 78.4, mmbench_en: 90.1, ocrbench: 89.2, mathvista_mini: 85.7, ai2d: 90.2, note: "thinking mode, model card"}
  tier: main
```
UI dùng `ram_gb` + `MemAvailable` để chặn trước khi chạy; `license.commercial`/`remote_code` để hiện cảnh báo; `quant_formats` để ẩn lựa chọn quant không hỗ trợ.

## 4. Recipe mặc định cho template (LoRA SFT, Spark)
Dựa trên recipe chính chủ (`qwen3_5_9b.yaml`, `qwen3_8_27b_lora.yaml`), chỉnh cho máy chung 1 GPU:
- `peft: dim 16, alpha 32, exclude *vision*/*visual*/*lm_head*`, `freeze_vision_tower: true`.
- `lr 1e-4`, AdamW β(0.9,0.95), wd 0.01, warmup 3%, cosine. `local_batch_size 1`, `global_batch_size 8–16`.
- `activation_checkpointing: true`, `loss_fn: FusedLinearCrossEntropy`, `attn: sdpa` (không dùng flash-attn trên sm_121).
- `max_pixels` mặc định 512·28·28 (user chỉnh được trong Advanced), video ≤16 frame.
- Seed cố định, `ckpt_every_steps` theo thời gian (~30 phút), `model_save_format: safetensors`, `save_consolidated: true` để export thẳng sang vLLM/ModelOpt.
- Dataset mẫu cho template: **phải được security duyệt license**. Ứng viên: `mmoukouba/MedPix-VQA` (dùng trong recipe NVIDIA, cần kiểm license), CORD-v2 (OCR hoá đơn, CC-BY-4.0 ❓). Không dùng LLaVA-Instruct/ShareGPT4V (ToS OpenAI).

## 5. Rủi ro
| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Combo 26.02 + AutoModel v0.4.0 không phải tổ hợp NVIDIA test | Trung bình | Smoke test đầu tiên khi mở GPU; fallback LLaMA-Factory trên pytorch:25.11 |
| Qwen3.5 chưa có NVFP4 chính thức trong ModelOpt | Trung bình | v1 chỉ hiện FP8; NVFP4 gắn "thử nghiệm" + so với checkpoint `nvidia/*-NVFP4` |
| Ước lượng RAM sai trên UMA → treo host | Cao | memguard (P007/P008) + hiệu chỉnh `ram_gb` bằng đo thật; headroom 10GB |
| Model ≥27B không chạy được trong 40GB | Chắc chắn | Ẩn mặc định, hiện lý do dễ hiểu |
| Qwen3.5 mặc định thinking mode → eval/chat dài và chậm | Trung bình | Template chat/eval đặt non-thinking mặc định, ghi rõ khi so với số công bố (P002) |

## 6. Việc cần làm khi PM mở GPU (thứ tự)
1. Probe vài giây `nemo-automodel:26.08` trên 580.142 (bao-backend). Nếu chạy được thì cân nhắc dùng thẳng 26.08/v0.6.0.
2. Build `vlm-automodel` (26.02 + v0.4.0), smoke LoRA Qwen3.5-0.8B/2B 20 step: đo peak RAM, it/s.
3. Smoke vLLM v0.31.0 serve Qwen3.5-2B và ModelOpt FP8 trên 2B.
4. Hiệu chỉnh `ram_gb` trong catalog từ số đo.

## Nguồn
Xem `.team/knowledge/research.md` §Nguồn; P008 Phụ lục A.1; `.team/security-reviews/2026-10-12-nemotron-3-nano-omni.md`.
