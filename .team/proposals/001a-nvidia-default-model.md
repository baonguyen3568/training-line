# Amendment 001a: model NVIDIA làm template mặc định (ADR-001)
Tác giả: bao-research · 2026-10-12 · Trạng thái: **ACCEPTED (ADR-001 amendment 1, 22:30)**. Điều kiện vào gallery cho Parse-v1.1: test build-from-config offline PASS + P-2 có field-F1 thật. USER ĐÃ CHỐT (22:10, qua PM): template mặc định = **Nemotron-Parse-v1.1 + CORD-v2** (không gated); template 2 = Cosmos-Reason2-2B (gated, có wizard). Amendment: `docs/adr/ADR-001-amendment-1.md` (PROPOSED)
Bối cảnh: user yêu cầu "dùng model NVIDIA". Template mặc định chuyển sang VLM của NVIDIA; Qwen thành catalog phụ.
Nguồn: HF API (`/api/models`, config.json tại SHA, danh sách file), model card, AutoModel 26.08 image (bao-backend kiểm), ADR-001.

## 1. Kết luận (bản đề xuất ban đầu của research; user đã chốt khác, xem trạng thái ở trên và §7)
**Template mặc định = `nvidia/Cosmos-Reason2-2B`; bản nâng cao = `nvidia/Cosmos-Reason2-8B`. Template thứ 2 "Đọc tài liệu/hoá đơn" = `nvidia/NVIDIA-Nemotron-Parse-v1.1` + CORD-v2** (cập nhật 21:40: đổi từ 2.0 sang v1.1, lý do ở §2.1).
Nemotron Nano VL 12B v2 và 8B **không** làm mặc định. Lý do ở §3; chỉ đi đường R&D/serve.

Vì sao chọn Cosmos-Reason2 (trả lời câu hỏi "12B hay 8B" của PM: không phải cả hai):
- Là model **của NVIDIA** (NVIDIA Open Model License), post-train từ Qwen3-VL-2B/8B-Instruct. Kiến trúc là **`Qwen3VLForConditionalGeneration` native trong transformers**: không có file .py, không `auto_map`, không phụ thuộc C-RADIO. Chỉ có model này tránh được lỗi N1 (code bắc cầu không pin) mà security nêu cho Omni.
- **Cả stack hỗ trợ sẵn**: AutoModel có recipe Qwen3-VL từ v0.3 (`qwen3_vl_4b/8b_instruct_rdr.yaml`); vLLM hỗ trợ Qwen3-VL; ModelOpt hf_ptq liệt kê **Qwen3-VL: FP8** chính thức.
- **Vừa ngân sách RAM dư dả**: 2B = 4.9GB weights, cùng cỡ Qwen3.5-0.8B/2B đã đo (S-3: 7–13GB). 8B = 17.5GB.
- Lợi ích phụ: G1-harness của P002 dùng **Qwen3-VL-2B-Instruct**, chính là base của Cosmos-Reason2-2B. Một harness kiểm được cả base Qwen lẫn bản NVIDIA post-train, và đo được ngay Δ do NVIDIA post-train.

## 2. Bảng VLM của NVIDIA (HF, 12/10/2026)
| Model (repo@SHA) | Params / kiến trúc | Size BF16 | License | Remote code | Engine train | RAM LoRA (ước) | Quant chính chủ có sẵn | Đánh giá |
|---|---|---|---|---|---|---|---|---|
| **nvidia/Cosmos-Reason2-2B**@9ce19a19 | 2.4B, Qwen3-VL dense | 4.9GB | NVIDIA Open Model License (+Apache-2.0 ghi chú); **gated** (phải chấp nhận điều khoản) | **Không** | AutoModel (qwen3_vl, native) | 8–14GB | — (tự quant FP8 bằng ModelOpt) | **✅ Template mặc định** |
| **nvidia/Cosmos-Reason2-8B**@a9fae2cf | 8.8B, Qwen3-VL dense | 17.5GB | như trên, gated | Không | AutoModel (`qwen3_vl_8b`) | 24–32GB | — | ✅ "Nâng cao" (chờ đo) |
| nvidia/Cosmos-Reason2-32B@4ed98283 | 33B | 66.7GB | như trên | Không | AutoModel | >70GB | — | ❌ quá ngân sách |
| nvidia/NVIDIA-Nemotron-Parse-2.0@b6742064 (→ thay bằng v1.1, §2.1) | ~1B, encoder-decoder parse | 3.6GB | OpenMDW-1.1 | Có (12 .py) + **encoder.auto_map → nvidia/C-RADIOv2-H** | AutoModel (`nemotron_parse_v1_1.yaml`) | ~8–12GB | — | ✅ Template 2: tài liệu/hoá đơn (task hẹp: OCR/parse, không chat tổng quát) |
| nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16@ca9543b1 | 12.6B, hybrid Mamba-Transformer | 26.4GB | NVIDIA Open Model License | **Có** (12 .py) + `vision_config.auto_map` → `nvidia/C-RADIOv2-H` (**repo thứ 2, không pin**, cùng lỗi N1 của Omni) | **Không có trong AutoModel**; Megatron-Bridge recipe chỉ TP2/TP4 và **bị gỡ ở v0.7.0**; HF-fallback cần kernel mamba trên arm64 ❓ | 35–45GB ❌ sát/vượt | FP8@e9550fde (15.4GB), **NVFP4-QAD**@16853d6c (10.6GB), cũng remote code | ⚠️ R&D + serve-only |
| nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1@437f4e28 | 8B, Llama-3.1 + C-RADIOv2-H | 17.4GB | NVIDIA Open Model License + điều khoản Llama 3.1 ❓ | Có (4 .py) + C-RADIO không pin | Không có trong AutoModel | 24–32GB | FP4-QAD | ❌ cũ (2025), remote code, 2 license |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning (BF16/FP8/NVFP4) | 31B/3B MoE hybrid | 66 / 35.2 / 22.4GB | NVIDIA Open Model Agreement | Có + C-RADIO không pin | AutoModel recipe có, nhưng LoRA cần ~70–80GB | ❌ | FP8@0acd4b1c, NVFP4@16993199 | R&D (security: conditional). Pin C-RADIO thì làm được về kỹ thuật (snapshot repo 2 tại SHA + `HF_HUB_OFFLINE=1` + sửa `auto_map` trỏ về local), **nhưng RAM vẫn chặn** |
| nvidia/Eagle2-2B / Eagle2.5-8B | 2B / 8B | 4.4 / 16.1GB | **CC-BY-NC-4.0** / NSCLv1 (non-commercial) | Có | — | — | — | ❌ vi phạm ADR-003 cho template |
| nvidia/Nemotron-Labs-Diffusion-VLM-8B, LocateAnything-3B, NV-Reason-CXR/CT, VILA-HD | — | — | NSCL / NVIDIA License / chuyên y tế | Có | — | — | — | ❌ nghiên cứu/chuyên biệt |
Nemotron 3.5 Super VL 120B-A12B (AutoModel 25/09): ❌ quá lớn.

### 2.1 Nemotron Parse: chọn v1.1, không chọn 2.0 / v1.2 (cập nhật 21:40, bao-backend kiểm class native)
| Bản | SHA | License | Khác biệt config | Class native AutoModel (26.08 **và main 11/10/2026**) |
|---|---|---|---|---|
| **v1.1** | fd1e3c6b | NVIDIA Open Model License; tokenizer CC-BY-4.0; card ghi "ready for commercial use" | gốc, khớp recipe `nemotron_parse_v1_1.yaml` | ✅ khớp |
| v1.2 | 2bd0189b | NVIDIA Open Model License | encoder đổi `image_processor_normalizes` → `processor_normalizes` | ⚠️ class đọc tên cũ → **chuẩn hoá ảnh 2 lần** (sai âm thầm, không crash) |
| 2.0 | b6742064 | OpenMDW-1.1 | như v1.2 + `tie_word_embeddings: true`, vocab 72256 | ❌ class khai `UNTIED_ONLY` → **từ chối load**; cộng thêm lỗi chuẩn hoá 2 lần |
Kết luận: dùng **v1.1@fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4** (3.8GB, 9 .py). **Đính chính 22:00**: `encoder.auto_map` của v1.1 (giống 2.0) **bắc cầu sang `nvidia/C-RADIOv2-H`**. Bản trước ghi "chỉ trong repo" là sai, vì script cũ chỉ kiểm `vision_config`; đã sửa thành `scripts/research/hf_model_meta.py`, quét mọi sub-config. Điều kiện của security: pin C-RADIOv2-H@0d8f4c18 bằng thư mục local, hoặc backend chứng minh đường class native của AutoModel không kéo remote code. Theo dõi upstream để nâng lên 2.0 khi class hỗ trợ (`processor_normalizes` + tied). Serve qua vLLM ❓ cần kiểm.

## 3. Vì sao không chọn Nano VL 12B v2 hay 8B làm mặc định
1. **Remote code bắc cầu không pin**: `vision_config.auto_map` trỏ sang `nvidia/C-RADIOv2-H` theo `main`. Đây đúng là phát hiện N1 HIGH mà security đã nêu cho Omni.
2. **Không có engine train chính chủ chạy được 1 GPU**: AutoModel 26.08 không có class. Megatron-Bridge chỉ có recipe TP2/TP4 và tài liệu ghi model "no longer actively maintained, will be removed in v0.7.0".
3. **RAM**: 26.4GB BF16 + LoRA + activation ≈ 35–45GB, sát hoặc vượt ngân sách 40GB. Bản FP8/NVFP4-QAD chỉ để **inference**, không phải base cho QLoRA trong AutoModel.
4. Mamba-hybrid cần kernel `mamba_ssm`/`causal_conv1d` trên aarch64 sm_121 ❓.
Điểm mạnh của 12B v2 (DocVQA 94.4, OCRBench 85.6 trên card) vẫn đáng giá. Đề xuất đưa nó vào catalog dạng **"chạy thử / serve NVFP4-QAD"** (vLLM, 10.6GB) sau khi security vet và pin C-RADIO, chưa cho fine-tune.

## 4. Lưu ý / rủi ro của Cosmos-Reason2
| Vấn đề | Xử lý |
|---|---|
| **Gated** (`gated=auto`): phải đăng nhập HF và chấp nhận điều khoản. Token mới đã hợp lệ (whoami OK), nhưng repo trả **403** vì user chưa accept trên trang model và/hoặc token fine-grained thiếu quyền "read public gated repos" (PM đang nhờ user) | **Blocker**: user cần sửa token và bấm accept trên HF cho cả 2B/8B. App phải có bước "chấp nhận license" cho user cuối (P010 đã có error catalog "license chưa chấp nhận") |
| License: NVIDIA Open Model License. Cho dùng thương mại, cho phát hành bản derivative. Phải kèm bản license + attribution **"Built on NVIDIA Cosmos"**. Có điều khoản **chấm dứt quyền nếu bỏ qua Guardrail** mà không thay bằng cái tương đương | Security đọc điều khoản Guardrail và đưa vào ADR-003. App hiện attribution |
| Model được post-train cho **physical AI** (robot, xe tự lái, video); card **không** báo MMMU/DocVQA/OCRBench | Tự đo baseline bằng P002 (G1-internal + so với Qwen3-VL-2B-Instruct). Nếu năng lực chung (OCR/chart) tụt rõ so với Qwen3-VL-2B thì cân nhắc lại. Mô tả template theo đúng thế mạnh: "hiểu ảnh/video và suy luận" |
| Card khuyên `max_new_tokens ≥ 4096` (CoT `<think>`), video FPS=4 | Đặt mặc định trong template/eval |
| Card ghi "min 24GB GPU, test trên H100/A100" | Không phải giới hạn thật cho model 2.4B. Smoke trên GB10 để xác nhận |

### 4.1 Vấn đề sản phẩm: repo gated (security nêu)
Cosmos-Reason2-2B/8B có `gated=auto`. Nếu nó là template mặc định thì **mọi end-user** phải có tài khoản HF, tự bấm accept trên web, và cấp token có quyền đọc gated repo trước khi chạy template đầu tiên. Đây là ma sát lớn với người dùng no-code.
Phương án (research đề xuất **(i) + (ii)**):
- (i) **Wizard onboarding trong app**: đăng nhập HF → mở trang accept → app kiểm `HEAD` file config (200 thay vì 401/403) → báo "đã sẵn sàng". Lỗi 403 đưa vào error catalog P010 với hướng dẫn tiếng Việt. Llama và Gemma 3 cũng gated, nên người dùng HF đã quen bước này.
- (ii) **Đường "chạy ngay không cần tài khoản"**: template thứ 2 (Nemotron-Parse-v1.1, **không gated**) hoặc Qwen3.5-2B (không gated, catalog phụ) để người mới chạy được thứ gì đó ngay. Gallery đánh dấu template nào cần tài khoản HF.
- (iii) Mirror/đóng gói sẵn weights Cosmos trong bản phát hành app. NVIDIA Open Model License có vẻ cho phép phân phối lại kèm license + attribution, nhưng gate của HF có điều khoản thu thập thông tin liên hệ. **Cần security/pháp lý xác nhận**, chưa làm.

## 5. Smoke plan (khi token HF OK + security APPROVE)
| # | Việc | Image | Đo / tiêu chí |
|---|---|---|---|
| N-1 | Tải `Cosmos-Reason2-2B`@9ce19a19 (4.9GB) + `Qwen3-VL-2B-Instruct` (G1-harness, ~4.3GB) | — | security vet SHA |
| N-2 | LoRA 30 step, CORD-v2, cùng hyperparams S-3/S-3b (YAML dẫn xuất từ `qwen3_vl_4b_instruct_rdr.yaml`) | vlm-automodel:26.08-1 | 4 số RAM (P002 B5), s/step, loss giảm |
| N-3 | Serve qua vLLM v0.31.0, chat thử ảnh + video 4fps | vllm-openai:v0.31.0 | TTFT, tokens/s, mem-util ≤0.3 |
| N-3b | **Gate gallery (ADR-001 am.1, áp cho mọi template)**: test ảnh trắng tổng hợp + train/serve parity. Với Qwen3-VL/Cosmos phải so **thêm số image token và grid_thw**, ngoài pixel_values: AutoModel processor (min/max_pixels trong YAML) vs vLLM `--mm-processor-kwargs`. Hai đường resize (smart_resize) khác nhau thì số token khác, lệch train/serve mà không báo lỗi | AutoModel + vLLM | pixel_values khớp 1e-3, số token khớp tuyệt đối |
| N-4 | **GATE** (§5.1): regression 500 × 6 benchmark, paired, Cosmos-Reason2-2B vs Qwen3-VL-2B | lmms-eval (container) | PASS/FAIL theo §5.1, chặn ADR |
| N-5 | ModelOpt FP8 (hf_ptq `--vlm`, pip 0.47) → serve → smoke eval | vlm-automodel + modelopt 0.47 | gate FP8 ≤1pt |
| N-6 (sau) | Nemotron-Parse-v1.1 LoRA trên CORD-v2 (recipe `nemotron_parse_v1_1.yaml`) | vlm-automodel:26.08-1 | template 2 |
| N-7 (sau) | Cosmos-Reason2-8B LoRA | như N-2 | đo RAM, quyết nhãn "nâng cao" |

## 5.1 Gate N-4 (PM yêu cầu, chặn ADR): năng lực chung của Cosmos-Reason2-2B so với Qwen3-VL-2B-Instruct
Lý do: Cosmos được post-train cho physical AI; card không báo benchmark chung.
- **Tầng chạy**: regression của P002 (500 mẫu cố định × 6 benchmark rule-scored: MMMU, MMBench v1.1, DocVQA, ChartQA, OCRBench, AI2D), **paired** cùng sample id, greedy, cùng harness/pixels. Subset 50 mẫu quá nhiễu để làm gate (CI ±10pt).
- **Prompt**: cả 2 model chạy non-thinking. Cosmos dùng system prompt mặc định, KHÔNG ép `<think>`, max_new_tokens 1024. Bước trích đáp án phải xử lý được `<answer>…</answer>`.
- **Điều kiện PASS** (Δ = Cosmos − Qwen3-VL-2B, kèm CI95 paired bootstrap):
  1. Trung bình Δ trên 6 benchmark ≥ **−3pt**.
  2. Không benchmark nào có Δ < **−5pt** với CI95 nằm hẳn dưới 0.
  3. Nhóm "Đọc tài liệu" (DocVQA, OCRBench) Δ ≥ −5pt, vì template mặc định dùng CORD-v2 là bài OCR.
- **FAIL → phương án cho PM hỏi user**:
  - (A) Template mặc định đổi sang **Nemotron-Parse-v1.1** cho bài hoá đơn (NVIDIA, chuyên tài liệu). Cosmos-Reason2 giữ cho template "hiểu video/suy luận".
  - (B) Giữ Cosmos-Reason2-2B, mô tả rõ trong template là "chuyên ảnh/video thực tế", chọn dataset mẫu hợp thế mạnh (video). Bài hoá đơn dùng Parse.
  - (C) Dùng Qwen3-VL-2B-Instruct (không phải NVIDIA, Apache-2.0) làm mặc định. Trái yêu cầu user nên chỉ là phương án cuối.
- Thời gian ước tính: 2 model × 3000 mẫu ≈ 1–1.5h trên máy chung, có GPU lock.

## 6. Thay đổi đề xuất cho ADR-001
- Catalog: thêm tier **`nvidia-default`** (Cosmos-Reason2-2B), **`nvidia-advanced`** (Cosmos-Reason2-8B), **`nvidia-doc`** (Nemotron-Parse-v1.1), **`nvidia-rnd`** (Nano 12B v2 VL serve-only, Omni). Qwen3.5 chuyển sang tier "phụ".
- Template mặc định: Qwen3.5-2B → **Cosmos-Reason2-2B + CORD-v2** (giữ CORD-v2: dataset đã APPROVED, task trích xuất hoá đơn vẫn hợp với Qwen3-VL).
- Thêm trường catalog `gated: true` + `attribution: "Built on NVIDIA Cosmos"`.

## 7. Sau khi user chốt (22:10)
| Vai trò | Model | Gated | Remote code | Điều kiện security |
|---|---|---|---|---|
| **Template mặc định** "Trích xuất hoá đơn → JSON" | nvidia/NVIDIA-Nemotron-Parse-v1.1@fd1e3c6b + CORD-v2@7f0115a4 | Không | Có (9 .py) + **bắc cầu C-RADIOv2-H** qua `encoder.auto_map`/`_name_or_path` và `hf_nemotron_parse_config.py` | **APPROVED** (23:10, sau P-0 PASS); P-1 phải assert processor nạp từ bản prepared |
| Template 2 "Hiểu ảnh/video tổng quát" | nvidia/Cosmos-Reason2-2B@9ce19a19 | **Có** | Không (theo card; config chưa đọc được) | CONDITIONAL tới khi đọc được config. Wizard: tài khoản HF → accept → token fine-grained chỉ đọc (D12). **Không mirror weights** (security) |
| Baseline / G1-harness | Qwen/Qwen3-VL-2B-Instruct@89644892 | Không | Không | APPROVED |

### 7.1 Smoke plan (ưu tiên mới)
| # | Việc | Config | Tiêu chí |
|---|---|---|---|
| **P-0** | Bước prepare: snapshot Parse-v1.1 + C-RADIOv2-H@0d8f4c18 → vá `encoder.auto_map`/`_name_or_path` trỏ về thư mục local → kiểm sha256 các .py → load offline (CPU) | launcher (bao-backend) | load không truy cập mạng (chạy `--network none`); security xác nhận |
| ↳ kết quả P-0 | **PASS** (bao-backend, commit 80f0efd): config = remote NemotronParseConfig (bản copy đã kiểm hash trong `.cache/prepared`), **model = class NATIVE của AutoModel**, encoder RADIO = remote từ bản pin C-RADIOv2-H@0d8f4c18; 936M params. PREPARED_MODEL_DIR=`/workspace/.cache/prepared/nemotron-parse-v1.1_fd1e3c6b/model`. Lưu ý: AutoModel build cần CUDA kể cả ở meta, nên mọi check đều phải lấy GPU_LOCK | | |
| **P-1** | Fine-tune 50 step (decoder; vision + embeddings frozen như upstream) | `configs/train/smoke/nemotron_parse_v1_1_cordv2_smoke.yaml` | chạy hết, loss giảm, 4 số RAM (P002 B5), s/step |
| **P-2** | Eval trên CORD-v2 **test** (100 ảnh) | eval script (bao-tester) | metric ở §7.2; dùng để so base vs sau fine-tune |
| P-3 | "Template run" đầy đủ: 800 ảnh train × 3 epoch (~300 step với GBS 8) | như P-1 với max_steps 300 | field-F1 ≥ 80 trên test (mốc tham khảo Donut ~84 F1 ❓) |
| P-4 | Serve/chạy thử: vLLM có hỗ trợ `nemotron_parse` không ❓; nếu không thì đường HF generate trong container | — | node "Chạy thử" của template hoạt động |
| P-5 | **Spike R&D (không chặn template)**: ModelOpt FP8 PTQ trên Parse-v1.1 (`hf_ptq.py`, encoder-decoder, ModelOpt chưa liệt kê). Quantize decoder Linear, giữ encoder RADIO ở BF16; so field-F1 FP8 vs BF16 trên CORD test | vlm-automodel + modelopt 0.47 | ΔF1 ≤1pt thì mở bước quantize cho template mặc định; nếu không, template trình diễn đủ chuỗi train → quantize → benchmark là **Cosmos-Reason2-2B** (Qwen3-VL, FP8 chính thức) |
| N-2/N-4 | Cosmos (hạ ưu tiên) | như §5 | như §5.1 |

### 7.2 Metric cho template hoá đơn (CORD-v2 test, 100 ảnh)
Chuẩn so sánh: theo cách đánh giá của Donut/CORD. Output của recipe là chuỗi token kiểu Donut (`json2token`), nên đổi ngược `token2json` rồi tính:
- **Field-level F1** (micro, trên các cặp key-value lá; so khớp chuỗi sau khi chuẩn hoá khoảng trắng). Đây là metric chính hiển thị cho user ("đọc đúng bao nhiêu % trường").
- **JSON-exact** (% ảnh có toàn bộ JSON khớp hoàn toàn). Metric khắt khe, phụ.
- **nTED accuracy** (1 − tree-edit-distance chuẩn hoá). Metric phụ, khớp báo cáo của Donut.
- **Parse-fail rate** (% output không đổi được thành JSON). Bắt buộc ≤5% sau P-3.
- Base model zero-shot xuất markdown/bbox (định dạng khác CORD) nên F1 gần 0. Đó là chuyện bình thường, **không** dùng làm baseline. Baseline là checkpoint step 0 → step cuối, kèm learning curve.
- higher_is_better: true cho F1/JSON-exact/nTED, false cho parse-fail. Thêm vào registry benchmark P002 với `eval_only=false` (CORD được phép train; split test giữ held-out, không bao giờ train).

## 8. So sánh Parse-v1.1 với Parse-2.0 (PM 2026-10-12 23:30: user quyết tự vá AutoModel, image `vlm-automodel:26.08-2`)
Điều kiện tiên quyết:
- Image 26.08-2 có patch của fixcode (alias `processor_normalizes` + hỗ trợ tied embeddings).
- **Test pixel_values bằng ảnh trắng tổng hợp PASS cho cả v1.1 lẫn 2.0**, để chứng minh patch không làm hỏng v1.1.
- Security duyệt 2.0 (OpenMDW-1.1, cùng điều kiện pin C-RADIO).

Config: `configs/train/smoke/nemotron_parse_2_0_cordv2_smoke.yaml`. Giống hệt bản v1.1, chỉ khác model và `NemotronParseLoss.class_token_start_idx`: recipe v1.1 dùng 50000, config 2.0 khai 52315. **Cần fixcode xác nhận** ngưỡng token toạ độ đúng cho vocab mới. Nếu sai thì trọng số loss toạ độ áp nhầm token và so sánh vô nghĩa.

Protocol:
| Bước | v1.1 | 2.0 | Ghi chú |
|---|---|---|---|
| Smoke 50 step | P-1 | Q-1 | chỉ để kiểm chạy được, RAM và tốc độ. **Không** kết luận chất lượng từ 50 step |
| Template run | P-3: 300 step | Q-3: 300 step | cùng seed, data, GBS, lr |
| Eval CORD test (100) | P-2 | Q-2 | `src/vlm_pipeline/eval/cord.py`; **paired bootstrap** theo ảnh |
| Zero-shot (tham khảo) | có | có | so năng lực parse tài liệu gốc trên vài ảnh. Không dùng làm gate vì định dạng output khác CORD |

**Nâng mặc định lên 2.0 chỉ khi đủ tất cả**:
1. field-F1 (2.0 − v1.1) ≥ **+2pt** và CI95 paired nằm hẳn trên 0. Với n=100, CI có thể rộng ±3–4pt. Nếu không tách được thì coi là "ngang nhau" và **giữ v1.1** (đang chạy ổn, đã APPROVED, ít rủi ro patch).
2. parse_fail_rate của 2.0 ≤ v1.1 và ≤5%.
3. Peak RAM job ≤ v1.1 +20%, s/step ≤ v1.1 +20%.
4. Security APPROVED 2.0.
5. Patch được viết thành test hồi quy trong CI (fixcode/tester), vì ta phải tự duy trì bản vá cho tới khi upstream sửa.
Kết quả ghi vào amendment ADR-001 mới. Nếu 2.0 không đạt thì ghi lý do và giữ v1.1.

### 8.1 Cập nhật (23:55)
- Security xác nhận Parse-2.0@b6742064 (vẫn là main) + C-RADIOv2-H@0d8f4c18: **APPROVED-WITH-CONDITIONS** (OpenMDW-1.1, tokenizer CC-BY-4.0, C-RADIO pin local). Hash nằm ở `configs/models/pinned/`.
- Gợi ý cho P-4 (serve): README của 2.0 có hướng dẫn chạy **vLLM** kèm `vllm_tied_patch/` (sitecustomize.py). Nghĩa là NVIDIA có đường serve Parse qua vLLM ❓ (cần kiểm với v1.1). Theo quy tắc của security: không đưa vào PYTHONPATH của host; file được vendor, kiểm hash, chỉ mount vào container vLLM.

### 8.2 Ngữ nghĩa loss khác nhau giữa v1.1 và 2.0 (bao-fixcode xác minh từ tokenizer + code tham chiếu của NVIDIA, 00:05)
- Code tham chiếu: `idx = getattr(config,'class_token_start_idx',50000)`; `loss[labels >= idx] *= 10`.
- v1.1 (config không có field): ngưỡng 50000 = `<output_plain>`, nên **mọi token đặc biệt** (định dạng, toạ độ từ 50009, class từ 52313) bị ×10.
- 2.0: ngưỡng 52315 = `<class_Text>`. Chỉ token class bị ×10, **toạ độ (từ 50011) không còn được nhân**. Không rõ đây là thay đổi chủ ý hay không, nên ta làm theo code tham chiếu đi kèm checkpoint.
- **Ảnh hưởng tới bài CORD**: `make_cord_v2_dataset` tạo đích bằng `json2token` (`<s_key>…</s_key>`), **không thêm token mới vào tokenizer** (đã kiểm AutoModel v0.6.0). Các thẻ này được tách thành token văn bản thường (<50000), nên đích CORD gần như không có token đặc biệt và trọng số ×10 hầu như không áp dụng ở cả 2 bản. → So sánh v1.1 với 2.0 trên CORD **công bằng về loss**. Khác biệt về toạ độ/bbox chỉ ảnh hưởng năng lực parse tài liệu kèm bbox (zero-shot), nên theo dõi riêng bằng vài ảnh tài liệu, không đưa vào gate.
- Checkpoint 2.0 không có `lm_head.weight`. Patch phải **tie thật** (lm_head = decoder.embed_tokens), không chỉ bỏ dòng reject; nếu không lm_head sẽ ngẫu nhiên. Test bắt buộc: logits của model đã load so với code tham chiếu.

### 8.3 ⚠️ v1.1 CŨNG bị chuẩn hoá ảnh 2 lần trong AutoModel native (bao-fixcode đo thật trên 26.08-1, 00:15)
- Nguyên nhân: `NemotronParseConfig.__init__` dựng `encoder` từ `AutoConfig(C-RADIOv2-H)`, rồi chỉ copy những key mà RADIOConfig có sẵn. `image_processor_normalizes` không thuộc RADIOConfig nên bị **rơi ở cả v1.1**; model đọc False. Trong khi đó processor v1.1 có `do_normalize: True`. Kết quả: ảnh bị normalize 2 lần.
- **Hệ quả cho template mặc định** (nghiêm trọng hơn chuyện so v1.1/2.0):
  1. Vision encoder đang **freeze**, nên decoder học trên đặc trưng ảnh bị lệch. Accuracy đo trong AutoModel không phản ánh chất lượng thật và không so được với model card.
  2. **Lệch train/serve**: model fine-tune trong AutoModel (normalize 2 lần), khi serve qua vLLM/HF tham chiếu (normalize 1 lần) sẽ nhận phân phối input khác lúc train. Chất lượng khi user "Chạy thử" có thể tụt mà không có cảnh báo nào.
- **Quyết định đề xuất**:
  - P-1 trên 26.08-1 vẫn chạy được, nhưng **chỉ để đo RAM/tốc độ** (không phụ thuộc chuẩn hoá).
  - **P-2/P-3 (field-F1) và mọi so sánh v1.1/2.0 chỉ chạy trên image 26.08-2 đã vá**.
  - Template Parse **chưa vào gallery** cho tới khi có 26.08-2 + test ảnh trắng tổng hợp PASS (khớp đúng `(1-m)/s`).
  - Thêm test train/serve parity: cùng 1 ảnh, `pixel_values` đầu vào patch-embed của đường AutoModel phải khớp đường serve (vLLM/HF) tới 1e-3.
- Bài học (đưa vào research-model-selection): mọi model có encoder dựng từ repo khác thì **cờ tiền xử lý có thể rơi khi merge config**. Phải kiểm bằng test số (ảnh tổng hợp), không kiểm bằng cách đọc config.

### 8.4 Quyết định của PM (00:20)
- P-1 trên 26.08-1 chạy xong, **chỉ lấy RAM/tốc độ**. P-2 và so v1.1/2.0 **chỉ chạy trên 26.08-2** sau khi tester xác minh patch.
- Số field-F1 trên image cũ: **chưa có số nào** tính tới 00:20 (đã kiểm `benchmarks/`, `outputs/runs/`). Nếu sau này xuất hiện thì gắn **INVALID (double-normalize, image 26.08-1)**.
- Recipe chính chủ của NVIDIA (`nemotron_parse_v1_1.yaml` trên AutoModel) cũng dính lỗi này. Nếu NVIDIA có công bố số fine-tune Parse bằng AutoModel thì số đó có thể **thấp hơn thực tế**, nên không dùng làm mốc so.

### 8.5 Trạng thái 00:30
- P-1 trên 26.08-1 **không chạy được**: processor remote của Parse import `albumentations` + `cv2`, image không có hai gói này. → P-1 dời sang **26.08-2** (patch fixcode + thêm deps).
- Gate trước P-1 trên 26.08-2: `check.sh --container` chạy probe ảnh trắng + parity pixel_values AutoModel vs HF tham chiếu (atol 1e-3). Parity với vLLM để ở P-4.
- ADR-002 A1: các run gate/số công bố (G1-harness, N-4, P-3) phải chạy từ `.worktrees/<run_id>` sạch, `gaps == []`; nếu không thì gắn NON-REPRODUCIBLE.

### 8.6 Gate "khớp NVIDIA" bằng golden output (00:45)
Snapshot Parse-v1.1 có sẵn `golden_outputs.json` + `test_golden.py` (đã nằm trong bảng hash security vet). Sau khi build 26.08-2, chạy zero-shot ảnh mẫu của repo qua **cả 2 đường**: (a) HF remote modeling của NVIDIA (sandbox, pin), (b) **class native của AutoModel** (đường train của ta). Cả hai phải khớp golden theo ngưỡng của `test_golden.py`. Đường (b) khớp golden thì chứng minh tiền xử lý + model native của ta khớp NVIDIA gốc, không chỉ khớp nội bộ. Gate này áp cho template mặc định, cùng với ảnh trắng + parity.
- (00:50, bao-tester) Tiêu chí số của gate: so ảnh trắng với **kỳ vọng `(1−m)/s` đã làm tròn về bf16**, vì RADIO ép pixel sang bf16 nên lệch tới ~0.005 ngay cả khi đúng. Parity train/serve so **cùng dtype ở 2 đường**, hoặc so trước bước cast. Phân biệt 1 lần/2 lần vẫn rất rõ: ảnh trắng sau 1 lần ≈ 2.2, sau 2 lần lớn hơn nhiều lần.

## 9. P-1 trên 26.08-2: peak +35GB. Phân tích nguyên nhân gốc (bao-research, 02:20)
Số đo (bao-tester, `benchmarks/perf/p1-parse-v1.1-full-26.08-2-20261012.json`): startup peak +35.0GB (cgroup 15.3GB + ~20GB CUDA/UMA). PM dừng job ở step 0.
**Kết luận: không phải do "full SFT quá nặng", mà do 3 lỗi cấu hình/engine. Đổi sang LoRA sẽ không chữa được nguyên nhân chính.**
| # | Bằng chứng | Nguyên nhân | Ảnh hưởng RAM |
|---|---|---|---|
| 1 | log: `Trainable parameters: 936,275,968 … 100.00%` dù YAML có `freeze_vision_tower: true`, `freeze_embeddings: true` | `apply_parameter_freezing` (nemo_automodel/components/utils/model_utils.py) chỉ freeze theo thuộc tính `vision_tower` hoặc tên module chứa `vision`/`visual`/`image_encoder`. Encoder của Parse tên là **`encoder`** (RadioWithNeck) nên **không khớp, không freeze gì cả**. Key `freeze_embeddings` **không tồn tại** trong recipe VLM (bị bỏ qua âm thầm) | Encoder ViT-H ~0.6B bị train: lưu activation để backward ở ~13k token ảnh + grad + Adam state. Đây là phần lớn của ~20GB CUDA |
| 2 | `enable_radio_vit_fused_attn` (cùng file) tìm `model.vision_model.radio_model.model.blocks`; Parse dùng `encoder`, nên **no-op**. Docstring của chính hàm ghi: không bật thì "materializes the full (B,H,seq,seq) attention tensor (~5 GiB per block)" | Attention RADIO chạy eager, không fused | Ma trận attention vài GB mỗi block, cộng dồn qua backward |
| 3 | log: `could not derive turn markers from chat template … Falling back to BPE pattern-match labels, which may produce nan loss` | Collate của Parse không xác định được đoạn trả lời | Không ảnh hưởng RAM, nhưng **có thể sai mask label**: loss tính cả prompt, hoặc nan. Phải kiểm trước khi tin bất kỳ số F1 nào |
| 4 | dataset load mất ~1 phút; cgroup 15.3GB | `make_cord_v2_dataset` trả về **list** đã decode toàn bộ ảnh (train + val) trong RAM, cộng `pin_memory: true` | vài GB phía CPU (cgroup) |
Lưu ý: recipe chính chủ `nemotron_parse_v1_1.yaml` của NVIDIA dính cùng lỗi 1–2 (chạy trên 8×H100 80GB nên không ai để ý). Đây cũng là ứng viên cho issue upstream.

### 9.1 Đề xuất (thay cho "chuyển sang LoRA")
Sửa trong adapter của ta (fixcode, cùng cơ chế với bản vá normalize), **không sửa recipe NVIDIA**:
- (a) Freeze theo **prefix tham số `encoder.`** (+ embedding của decoder nếu muốn giữ đúng ý recipe), rồi **assert**: tỉ lệ tham số train khớp kỳ vọng (~30–35% cho decoder mBART), nếu không thì job fail ngay ở bước build.
- (b) Bật `fused_attn=True` cho các block RADIO trong `encoder` của Parse. Assert số block đã bật > 0.
- (c) Kiểm label mask: in 1 batch, đoạn có label ≠ -100 phải đúng bằng chuỗi đích CORD. Thêm unit test.
- (d) `pin_memory: false`, giới hạn val bằng `limit_dataset_samples` (val chỉ cần ~100).
Config chạy lại: giữ nguyên `nemotron_parse_v1_1_cordv2_smoke.yaml` (`local_batch 1`, act-ckpt), chỉ thêm `pin_memory: false` + val limit. 4 sửa ở adapter nằm trong image 26.08-3.
**Ước lượng sau sửa (chưa đo)**: encoder chỉ chạy forward không grad, fused attention, train decoder ~0.3B (bf16 Adam) → job ~8–14GB. Gate an toàn để chạy lại: memguard abort khi MemAvailable < 10GB (thay vì 6GB) + chỉ chạy tới step 5 rồi đo.
LoRA chỉ là phương án 2, nếu sau khi sửa vẫn > 20GB: LoRA trên decoder chỉ tiết kiệm thêm grad + Adam của ~0.3B (~2–3GB).

### 9.2 Checklist "lần đo hợp lệ" cho Parse trên 26.08-3 (thống nhất với bao-tester, 02:40)
Thiếu 1 điều kiện → run **INVALID**, không ghi số vào catalog, không chấm F1.
- [ ] **(a) Tỉ lệ tham số train đúng thiết kế.** Runner ghi `trainable_params` / `total_params` vào run.json. Điều kiện: `trainable_params == số tham số KHÔNG thuộc prefix encoder.` (trừ embedding nếu freeze), đếm từ chính model lúc build. Mốc ~30–35% chỉ là ước lượng của research; số đúng là số đếm được.
- [ ] **(b) Memguard thực sự bật** (backend xác nhận trước khi start): abort khi MemAvailable < 10GB, **hoặc** khi tốc độ tụt > 1GB/s kéo dài 5s (lần trước tụt ~2GB/s).
- [ ] **(c) 4 số RAM tách theo pha**: step 0 (cache JIT lạnh) và step 1–5 (steady). Gồm cgroup memory.peak, torch `max_memory_reserved` (runner log mỗi step), tổng a+b, delta MemAvailable của host + baseline nhiễu 60s.
- [ ] **(d) Mask label đúng**: test của fixcode chạy trên batch thật của run. Chỉ token câu trả lời (chuỗi json2token) có label ≠ −100. Fail thì F1 INVALID.
- [ ] (e) Gate số học (ảnh trắng, parity, golden NVIDIA) vẫn PASS **sau khi bật fused_attn**.

### 9.3 Đính chính + chốt phương án freeze (03:00)
- **Đính chính dòng 2 bảng §9**: bao-fixcode dựng Parse trên meta và xác nhận cả 32 block RADIO **đã `fused_attn=True`**, vì timm tự bật SDPA. `enable_radio_vit_fused_attn` đúng là no-op với Parse nhưng **không** gây eager. Research kết luận sai ở điểm này vì suy ra từ docstring mà không kiểm object thật. Fixcode vẫn thêm assert "mọi block RADIO fused".
- **Nguyên nhân chính đã xác định lại** (bằng chứng trong log run c14e):
  1. 100% tham số train (936.3M);
  2. activation checkpointing chỉ chọn **10/10 layer decoder** (`groups={'unknown': 10}`). **32 block RADIO không được checkpoint** mà lại đang train, nên activation của ~13k token ảnh × 32 block ViT-H bị giữ lại cho backward. Đây nhiều khả năng là phần lớn ~20GB CUDA;
  3. tham số + grad + Adam: log ghi Adam state dùng **bf16**, nên ≈ 936M × 8B ≈ 7.5GB (không phải fp32 12–16B/param).
  Freeze RADIO xoá được cả (2) lẫn phần lớn (3).
- **Chốt: phương án A** = freeze backbone RADIO (`encoder.model_encoder.*`, 651.6M), **neck vẫn train** (9.4M). Lý do: đúng lệ VLM (vision tower đóng băng, projector/neck học), và neck giúp thích nghi miền ảnh hoá đơn với chi phí nhỏ.
- **Embedding: KHÔNG freeze, cho cả v1.1 lẫn 2.0.** Lý do:
  (i) `freeze_embeddings` của recipe NVIDIA chưa từng có hiệu lực ở upstream, nên không freeze là giữ đúng hành vi thực tế của recipe;
  (ii) 2.0 có lm_head **tied** với embed_tokens, freeze embed là freeze luôn lớp output;
  (iii) cùng 1 chính sách cho 2 bản thì so sánh v1.1/2.0 công bằng.
  → Bỏ key `freeze_embeddings` khỏi YAML (bản vá sẽ báo lỗi với key lạ).
- Kỳ vọng tham số train: v1.1 = **284.6M (30.4%)**, assert ±1%. 2.0: fixcode đếm lại (tied, vocab 72256).
- Ghi nhận thêm: Adam bf16 có cảnh báo ổn định từ AutoModel. Giữ như recipe, nhưng theo dõi loss ở P-3. Nếu dao động thì thử fp32 master weights cho decoder (thêm ~1–2GB).

## 10. P-1b PASS trên 26.08-3 (04:30)
- Checklist §9.2: (a) trainable 284,633,088 = **30.40%** ✓; (b) memguard abort 10GB ✓; (c) host peak **+17.8–18.0GB** (backend/tester), cgroup 22.9GB (cận trên vì gồm page cache); ~20 s/step với GBS 8.
- Catalog: `ram_gb_train` 35 → **20** (measured), commit c7e323d. 20 so với 18.0 = +11%, đạt luật estimator (≥ số đo, ≤ +30%). Preflight cộng thêm 10GB headroom (cần ~30GB trống). Đây là biên an toàn riêng, không tính vào luật +30%.
- **Chưa xong**:
  - (d) cảnh báo `could not derive turn markers` VẪN xuất hiện trên 26.08-3, nên test mask label của fixcode phải PASS trên batch thật trước khi chấm F1;
  - (e) gate số học sau khi sửa adapter.
- P-2 eval qua đường HF remote (BUG-001: native generate lỗi), từ worktree sạch.

### 10.1 Preset thời lượng cho template no-code (PM gợi ý)
CORD train 800 ảnh, GBS 8 → 100 step/epoch × ~20 s ≈ **33 phút/epoch** (+ ~1 phút khởi động, + validation).
| Preset (Simple) | Cấu hình | Thời gian ước | Dùng khi |
|---|---|---|---|
| **Thử nhanh** | 50 step | ~17–20 phút | xem pipeline chạy được, chưa kỳ vọng chất lượng |
| **Nhanh** (mặc định template) | 1 epoch (100 step) | ~35 phút | kết quả đầu tiên dùng được ❓ (P-2 sẽ cho biết F1 sau 1 epoch) |
| Tiêu chuẩn | 3 epoch (300 step) | ~1h40 | chất lượng tốt nhất của template |
Nếu P-2/P-3 cho thấy 1 epoch đã đạt ≥90% F1 của 3 epoch thì giữ "Nhanh" làm mặc định; nếu không thì mặc định "Tiêu chuẩn" và hiện rõ thời gian. Cần đo F1 theo checkpoint ở step 100/200/300 trong cùng run P-3 (ckpt_every 100), không chạy 3 run riêng.

### 10.2 (d) mask label PASS (bao-fixcode, 04:45)
- 4 mẫu thật, 26.08-3: số token có label = số token đích + 1; `decode(labels) == json2token + "</s>"`. Không có token task (`<predict_bbox>`, `<predict_classes>`, `<output_markdown>`), không có câu hỏi user. Cảnh báo turn markers là vô hại với Parse. Có test khoá hành vi: `test_nemotron_parse_freeze.py::test_label_mask…`.
- ⚠️ **Hệ quả cho P-2 (train/serve parity về prompt)**: model fine-tune học sinh chuỗi CORD **ngay sau prefix decoder mà collate dùng, không có token task**. Khi eval/chạy thử qua đường HF remote, prefix khởi đầu của decoder phải **giống hệt lúc train**. Không được dùng prompt task mặc định của Parse (`</s><s><predict_bbox><predict_classes><output_markdown>`), vì model sẽ rơi vào chế độ parse markdown của bản gốc và F1 thấp giả tạo. Cần backend/tester lấy đúng `decoder_input_ids` khởi đầu từ `nemotron_parse_collate_fn` và assert nó trước khi chấm.

### 10.3 O-1 overfit sanity trước P-3 (đề xuất 05:00)
- Train Parse-v1.1 trên **20 ảnh train cố định**, 60 step (GBS 4, lr 5e-5), rồi chấm F1 trên CHÍNH 20 ảnh đó. Config: `configs/train/smoke/nemotron_parse_v1_1_overfit20.yaml`. Ước ~10–15 phút GPU.
- PASS: field-F1 ≥ 90, parse_fail ≈ 0. Chứng minh chuỗi label → prefix → generate → token2json → scorer đúng đầu-cuối. FAIL thì lỗi ở đường ống, sửa trước khi chạy P-3 (1h40).
- Record gắn `purpose: pipeline_check`; không phải số chất lượng, không đưa vào báo cáo hay catalog.
- Checkpoint P-1b (5 step) chỉ dùng để tester kiểm đường eval chạy được.
