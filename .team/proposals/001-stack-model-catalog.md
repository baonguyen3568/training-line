# Proposal 001: Engine stack NVIDIA + catalog model được hỗ trợ (bản sau RESET)
Tác giả: bao-research · 2026-10-12 · Trạng thái: **backend APPROVED §2/§3.1 (19:45) · security APPROVED kèm 4 chỉnh sửa (đã áp dụng 19:55) → sẵn sàng để PM chốt ADR-001 để PM chốt ADR-001** (S-6, cập nhật 2026-10-12 19:30: default image 26.08)
Reviewer: bao-backend (engine/image), bao-security (license/remote code), bao-cleanup (dung lượng), bao-tester (eval), bao-pm.
Chi tiết bằng chứng ở `.team/knowledge/research.md`. ✅ = đã đo/xác minh · ❓ = chưa xác minh.

## 0. Ảnh hưởng của RESET (R-4)
- Trước RESET: chọn **1** base model + 1 recipe. Sau RESET: app no-code cho user **tự chọn model**, nên 001 định nghĩa:
  (1) **engine** chạy phía sau mỗi loại node, (2) **catalog model** có metadata để UI chặn lỗi trước khi chạy (RAM, license, quant được), (3) **recipe mặc định** cho template.
- Catalog là **dữ liệu** (`configs/catalog/models.yaml`), không hard-code. Thêm model = thêm 1 entry + được security vet.

## 1. Ràng buộc nền (đã chốt với user qua PM)
- Driver **580.142** được giữ nguyên. **Cập nhật 19:10 (P008 A.4, bao-backend probe):** `nemo-automodel:26.08` (CUDA 13.3) **CHẠY được** trên 580.142 qua forward-compat (bf16 matmul ~40 TFLOPS, SDPA OK). Vì vậy quy tắc đổi từ "CUDA ≤13.1" thành **"mỗi image phải probe PASS trên 580.142"**. Không suy rộng: forum báo vllm 26.04 fail trên 580.173. (b') vẫn nên làm theo lịch vá toolkit.
  Phương án nâng cấp (b'): cập nhật DGX OS chính chủ lên driver 580.159.03 để mở khoá image 26.08 (❓ cần probe). User quyết khi cần.
- RAM thực cho job ~**40GB** (máy dùng chung). Ước lượng RAM trong catalog tính theo con số này.
- Không tải gì cho tới khi PM mở lại (HF token đang 401).

## 2. Engine theo loại node
| Node | Engine mặc định | Image (arm64, CUDA) | Fallback / ghi chú |
|---|---|---|---|
| Train SFT/LoRA | **NeMo AutoModel nhánh v0.6.0** (trong image `nemo_automodel.__version__` báo **0.5.0**; luôn ghi kèm digest) (`FinetuneRecipeForVLM`) | **`vlm-automodel:26.08-1`** = FROM `nemo-automodel:26.08@b0bd0e50…` (CUDA 13.3 forward-compat, torch 2.13 nv26.06, transformers 5.12.1, flash_attn 2.7.4, bnb 0.49.2) + `chmod -R o+rX /opt/Automodel` (security đã duyệt có điều kiện: FROM pin digest, không secret, có label, hadolint pass, digest vào IMAGES.lock, security review Dockerfile). **Điều kiện: spike train 20 step PASS** | (26.02 + v0.4.0 **đã bỏ** theo PM 2026-10-12, không pull; chỉ là phương án lịch sử nếu một bản cập nhật driver làm hỏng forward-compat.) Engine thứ 2 (**sau v1**): LLaMA-Factory 0.9.5 FROM pytorch:25.11. Megatron-Bridge (nemo:26.08) ❓ probe |
| Train QLoRA | AutoModel `qlora.py` (bitsandbytes 0.49.2 có sẵn trong 26.08) | như trên | ❓ kernel bnb trên sm_121. Chỉ cần cho model ≥27B; gắn cờ "thử nghiệm" |
| RL (GRPO/DPO) | Giai đoạn 2. NeMo-RL v0.7.0 (CUDA 13.2) ❓ cần probe; fallback: DPO qua LLaMA-Factory, GRPO qua TRL | nemo-rl:v0.7.0 ❓ / pytorch:25.11 | — |
| Chạy thử / serve | **vLLM v0.31.0** (đã vá RCE GHSA-h3rc-6mm3-gc2m) | `vllm/vllm-openai:v0.31.0` arm64 `sha256:3f7dd5b7…`, `cuda>=13.0` (security APPROVED, ngoại lệ D1 ngoài NGC; bind 127.0.0.1) | TRT-LLM 1.2.1 (13.1) cho perf. `--gpu-memory-utilization` mặc định **≤0.3** (~36GB; 0.4 ≈ 48GB vượt ngân sách), chỉ tăng khi preflight cho phép |
| Quantize | **ModelOpt 0.47** `examples/hf_ptq/hf_ptq.py --vlm` | **0.47 KHÔNG có sẵn trong image nào**: 26.08 kèm ModelOpt 0.44 (trước khi VLM gộp vào hf_ptq ở 0.46). Phải cài pip 0.47 (`--no-deps` lên vlm-automodel, hoặc image `vlm-modelopt` riêng) ❓ test | Qwen3.5/3-VL: **chỉ FP8 chính thức**; NVFP4/INT4 chính thức cho Qwen2.5-VL, Nemotron VL. Vision tower giữ BF16 |
| Benchmark accuracy | **lmms-eval 0.7.3** gọi endpoint vLLM (+ in-process HF cho gate reproduce) | **trong container** (pytorch:25.11@digest hoặc image vlm-*), `HF_HUB_OFFLINE=1` sau khi pre-download theo AssetRef (yêu cầu security; không chạy host venv) | VLMEvalKit pin commit. NeMo Evaluator VLM container **không có arm64** ❌ |
| Benchmark perf | vLLM bench / script riêng (P002, bao-tester) | như serve | — |
| Data | HF `datasets` + JSONL chat messages | host/pytorch:25.11 | NeMo Curator 26.07 (arm64 image+video) cho dedup/filter. Để giai đoạn 2 |
| Export | HF safetensors (merge LoRA) / unified ckpt của ModelOpt | — | — |

Format dữ liệu chuẩn nội bộ: messages `[{role, content:[{type:image|video|text,...}]}]` (giống AutoModel/HF). Converter sang `dataset_info` của LLaMA-Factory nằm trong engine adapter.

## 3. Catalog model v1
RAM ước tính = weights BF16 + LoRA/optimizer + activation với `max_pixels` mặc định và grad checkpointing. Đây là **ước lượng, phải hiệu chỉnh bằng smoke run**.
Mốc tham chiếu: Qwen3-8B LoRA đo được ≈ 21GiB.

| Tier | Model (repo@SHA do security vet) | Params | License | Remote code | Disk | RAM LoRA BF16 (ước) | Engine train | Quant được (chính thức) | Trong app v1 |
|---|---|---|---|---|---|---|---|---|---|
| CI | Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 | 0.8B | Apache-2.0 | Không | ~2GB | **đo: 7–13GB** (S-3) | AutoModel (YAML tự viết từ qwen3_5_4b) | FP8 | Ẩn (chỉ CI) |
| **Dev/Template** | **Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc** | 2B | Apache-2.0 | Không | ~5GB | ước **10–16GB** (chờ S-3b) | AutoModel | FP8 | ✅ mặc định template |
| Main | Qwen/Qwen3.5-4B@851bf6e8 | ~5B | Apache-2.0 | Không | 9.3GB | ước **16–24GB** | AutoModel (`qwen3_5_4b.yaml`) | FP8 | ✅ |
| **Main** | **Qwen/Qwen3.5-9B@c2022362** | ~10B | Apache-2.0 | Không | 19.3GB | ước **28–38GB** (sát trần 40GB; **chưa được coi là "khuyên dùng" cho tới khi đo**) | AutoModel (`qwen3_5_9b.yaml`) | FP8 | ✅ (nhãn "khuyên dùng" chờ S-3c) |
| Stretch | Qwen/Qwen3.6-27B@6a9e13bd | 27B | Apache-2.0 | Không | 55.6GB | ~65GB BF16 ❌ / QLoRA ~35GB ❓ | AutoModel (cùng kiến trúc 3.5 ❓) | FP8 (+`nvidia/…-NVFP4` có sẵn, **chờ vet**) | ⚠️ "cần máy rảnh", ẩn mặc định |
| Stretch | Qwen/Qwen3.8-27B@1d4bf0f2 | 27B | Apache-2.0 | Không | 55.6GB | như trên | AutoModel v0.6.0 (`qwen3_8_27b_lora.yaml`), 26.08 chờ spike | FP8, NVFP4 (nvidia/, **chờ vet**) | ⚠️ sau b' |
| Stretch | Qwen/Qwen3.6-35B-A3B@995ad96e | 35B/3B MoE | Apache-2.0 | Không | 71.9GB | ~80GB ❌ | AutoModel qwen3_5_moe | FP8 | ❌ v1 |
| Alt | google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7 (10.2GB), google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2 (16.0GB) | 2–8B | Apache-2.0 (security: commercial=yes; hiện Prohibited Use Policy như khuyến nghị) | Không | ~5–16GB | ~10–20GB | AutoModel (`gemma4_*_peft.yaml`) | FP8 ❓ | ✅ (security APPROVED) |
| R&D | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e99324 | ~31B/3B | NVIDIA Open Model Agreement | **Có, bắc cầu 2 repo (C-RADIOv4-H)** | 66GB | ~75GB ❌ | AutoModel v0.6.0 (26.08, chờ spike) | FP8/NVFP4 (nvidia/) | ❌ chỉ R&D (security: approved-with-conditions) |
Loại khỏi catalog: Mistral Small 4 (119B), InternVL3.5 (lạc hậu), Qwen3-VL (đã bị Qwen3.5 thay thế; chỉ dùng Qwen3-VL-2B cho G1-harness của P002), Nemotron Nano 12B v2 VL (đang bị deprecate, chỉ có recipe multi-GPU).

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

### 3.2 Hiệu chỉnh RAM bằng số đo thật (cập nhật 2026-10-12 20:10)
Nguồn: S-3, LoRA Qwen3.5-0.8B, dim 8, vision frozen, local_batch 1, GBS 4, act-ckpt, sdpa, MedPix (~100–1200 label tokens/step), image `vlm-automodel:26.08-1`.
Số liệu do bao-backend và bao-tester đo. File: `outputs/runs/20261012-1630-lora-qwen3.5-0.8b-s3a/`, `outputs/perf/s3-20261012/`, `benchmarks/perf/s3-lora-qwen3.5-0.8b-20261012.json`.
| Đại lượng | Giá trị |
|---|---|
| CUDA max_allocated | 8.14GiB ở step 0 (warmup + Triton JIT ~2m50s), steady 3.1–4.9GiB |
| RSS container | ≤5.0GB |
| Host MemAvailable giảm | steady **+6.5GB**, peak **+12.6GB**. Peak rơi vào step 10–15, **không trùng** đợt lưu checkpoint (adapter chỉ 61MB). Có thể lẫn nhiễu từ service khác (MemAvailable dao động 30–36GB cùng lúc) → coi là cận trên bảo thủ |
| Tốc độ | ~2.5 s/step, tps median ~550, 20 step ≈ 3m45s (gồm 2m50s step 0) |
| Disk | adapter + optimizer 61MB/ckpt |
Mô hình ước lượng tạm thời (1 điểm đo, chưa tách được phần cố định và phần scale theo model):
- `steady ≈ 4.7GB cố định (CUDA ctx + lib + dataloader) + 2GB × P(tỷ tham số, gồm vision)`, cộng activation tăng theo hidden size
- `peak ≈ steady × 1.6–1.9` (hệ số lấy từ 0.8B, gồm warmup/JIT step 0)
Theo đúng ngưỡng của tester (ước lượng ≥ thực đo, ≤ +30%), các dải ở bảng §3 cho 2B/4B/9B **chưa đạt chuẩn "measured"**: `source=estimated`.
UI phải dùng **cận trên** của dải cho preflight.
Cần đo tiếp:
- **S-3b**: Qwen3.5-2B (template mặc định), cùng config + CORD-v2. Mục đích: có điểm thứ 2 để fit hệ số scale, và chạy thật template.
- S-3c: Qwen3.5-9B. Chỉ chạy khi S-3b cho thấy ước lượng ≤35GB.
- Tách nhiễu bằng cách đo thêm trong container: `torch.cuda.max_memory_allocated` + RSS process. Không chỉ dựa vào MemAvailable của cả host.

### 3.1 Image tối thiểu theo model (train / serve), dưới driver 580.142
| Model | Train (tối thiểu) | Serve | Quantize đã kiểm chứng (nguồn) |
|---|---|---|---|
| Qwen3.5-0.8B/2B/4B/9B | `vlm-automodel:26.08-1` | vllm-openai:v0.31.0 | FP8 (ModelOpt hf_ptq support matrix "Qwen3.5: FP8"); NVFP4/INT4 = chưa kiểm chứng |
| Gemma 4 E2B/E4B | như trên (`gemma4_{2b,4b}_peft.yaml`) | vllm-openai:v0.31.0 ❓ | chưa kiểm chứng (ModelOpt chỉ liệt kê Gemma 3 FP8) |
| Qwen3.6-27B | như trên ❓ (cùng kiến trúc) + QLoRA ❓ | vllm-openai:v0.31.0 | FP8 `Qwen/…-FP8`, NVFP4 `nvidia/…-NVFP4` (checkpoint có sẵn, **chờ security vet từng repo**, license có thể khác bản gốc) |
| Qwen3.8-27B, Nemotron Omni | `vlm-automodel:26.08-1` (v0.6.0), chờ spike | vllm-openai:v0.31.0 | checkpoint FP8/NVFP4 (chờ security vet) |
Template mặc định (ADR-003: chỉ asset dùng thương mại được): **Qwen3.5-2B** (Apache-2.0). Dataset mẫu của template: CORD-v2 (CC-BY-4.0), xem §4.

## 4. Recipe mặc định cho template (LoRA SFT, Spark)
Dựa trên recipe chính chủ (`qwen3_5_9b.yaml`, `qwen3_8_27b_lora.yaml`), chỉnh cho máy chung 1 GPU:
- `peft: dim 16, alpha 32, exclude *vision*/*visual*/*lm_head*`, `freeze_vision_tower: true`.
- `lr 1e-4`, AdamW β(0.9,0.95), wd 0.01, warmup 3%, cosine. `local_batch_size 1`, `global_batch_size 8–16`.
- `activation_checkpointing: true`, `loss_fn: FusedLinearCrossEntropy`, `attn: sdpa` (không dùng flash-attn trên sm_121).
- `max_pixels` mặc định 512·28·28 (user chỉnh được trong Advanced), video ≤16 frame.
- Seed cố định, `ckpt_every_steps` theo thời gian (~30 phút), `model_save_format: safetensors`, `checkpoint.save_consolidated: final` (mặc định trên 26.08; `true` = mỗi lần ckpt) để export thẳng sang vLLM/ModelOpt, `max_recent_checkpoints` giới hạn dung lượng.
- Dataset mẫu cho template: **`naver-clova-ix/cord-v2`@7f0115a4…** (HF tag `license:cc-by-4.0`, dùng thương mại được nhưng phải ghi attribution; không gated; ~1K ảnh hoá đơn → JSON có cấu trúc). AutoModel đã có recipe dùng CORD-v2 (`gemma3_vl_4b_cord_v2*.yaml`, `nemotron_omni_cord_v2*.yaml`), nên có sẵn hàm dataset/collate. Task trích xuất hoá đơn dễ hiểu với người không biết code và không trùng benchmark eval. **Đã bỏ MedPix-VQA** (license gốc chưa rõ, security xếp CONDITIONAL). **Security APPROVED** (license CC BY 4.0, xác minh trong README clovaai/cord; SHA đầy đủ `7f0115a4b758a71d6473b8d085751692da2fef98`). App phải hiện attribution: "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing" (Park et al., NAVER Clova), github.com/clovaai/cord, CC BY 4.0 kèm link, ghi chú nếu dữ liệu đã bị sửa đổi. Không dùng LLaVA-Instruct/ShareGPT4V (ToS OpenAI).

## 5. Rủi ro
| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| 26.08 chạy qua forward-compat (không phải cấu hình NVIDIA công bố cho Spark) | Trung bình | Spike train 20 step trước khi pin (S-3); nếu hỏng thì (b') cập nhật DGX OS chính chủ |
| Qwen3.5 chưa có NVFP4 chính thức trong ModelOpt | Trung bình | v1 chỉ hiện FP8; NVFP4 gắn "thử nghiệm" + so với checkpoint `nvidia/*-NVFP4` |
| Ước lượng RAM sai trên UMA → treo host | Cao | memguard (P007/P008) + hiệu chỉnh `ram_gb` bằng đo thật; headroom 10GB |
| Model ≥27B không chạy được trong 40GB | Chắc chắn | Ẩn mặc định, hiện lý do dễ hiểu |
| Qwen3.5 mặc định thinking mode → eval/chat dài và chậm | Trung bình | Template chat/eval đặt non-thinking mặc định, ghi rõ khi so với số công bố (P002) |

## 6. Việc cần làm khi PM mở GPU (thứ tự)
1. ~~Probe `nemo-automodel:26.08` trên 580.142~~ → **PASS** (bao-backend, P008 A.4).
2. Build `vlm-automodel:26.08-1`, smoke LoRA Qwen3.5-0.8B/2B 20 step: đo peak RAM (cả lúc khởi động), it/s. Fail → báo PM (phương án: (b') cập nhật DGX OS chính chủ, hoặc LLaMA-Factory trên pytorch:25.11).
3. Smoke vLLM v0.31.0 serve Qwen3.5-2B và ModelOpt FP8 trên 2B.
4. Hiệu chỉnh `ram_gb` trong catalog từ số đo.

## Nguồn
Xem `.team/knowledge/research.md` §Nguồn; P008 Phụ lục A.1; `.team/security-reviews/2026-10-12-nemotron-3-nano-omni.md`.

### B-4 probe FP8 (research, 2026-10-13): ModelOpt 0.44 trong image 26.08-5 ĐỦ cho Qwen3.5 FP8
Run `outputs/runs/20261013-0937-b4-fp8-probe-q35-08b-b28b`, script `scripts/research/probe_modelopt_fp8.py`. Qwen3.5-0.8B; calib 32 ảnh CORD train; parity 8 ảnh CORD val, 96 token.
- `mtq.quantize(FP8_DEFAULT_CFG)` + `export_hf_checkpoint` chạy OK với transformers 5.12.1, dù ModelOpt cảnh báo "transformers>=5.0 support is experimental".
  - 890 quantizer. Giữ bf16: lm_head, `model.visual*`, `linear_attn.conv1d` (lớp Gated DeltaNet, ModelOpt tự loại).
- Parity so với bf16 (teacher-forced trên câu trả lời của bf16): top-1 khớp trung bình **96.1%** (thấp nhất 92.5%), KL trung bình **0.0087** (cao nhất 0.013). Greedy exact 0/8 ở 96 token: dự kiến, vì 1 token khác là hai chuỗi tách nhau, nên **không dùng exact-match làm gate**.
- Dung lượng 1.77 → **1.23 GB**. Model 0.8B có embeddings + vision lớn so với phần còn lại; model 2–4B sẽ giảm nhiều hơn. Peak bộ nhớ thật (run.json, memguard) 16.6 GB; torch allocator 6.3 GB. Calib 29 s, export 14 s.
- **Pitfall:** lần đầu chạy, `modelopt_cuda_ext_fp8` JIT-compile bằng nvcc mất ~3 phút ⇒ image cho node Nén phải build sẵn extension này (hoặc cache TORCH_EXTENSIONS_DIR).
- **Pitfall:** ModelOpt ≥0.44 có `quant_cfg` dạng **list** (entry sau thắng), không phải dict.
- ⇒ **Chưa cần image ModelOpt 0.47** cho FP8 Qwen3.5. Còn mở: vLLM 0.31 load checkpoint này (nhờ fixcode trong B-7); Parse (remote code, encoder-decoder) FP8 là spike riêng (P-5).
- Gate đề xuất cho node: sanity = top-1 ≥ 0.95 và KL ≤ 0.02 trên ảnh val (rẻ); gate người dùng = `quant_gate` (Δ benchmark, configs/eval/gates.yaml).
