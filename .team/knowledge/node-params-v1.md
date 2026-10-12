# Tham số node v1 (M1-1): Simple / Advanced + mặc định an toàn cho GB10
Tác giả: bao-research · 2026-10-12 · Dùng cho bao-cleancode (NodeSpec), bao-design (copy), bao-backend (worker).
Nguồn mặc định: recipe chính chủ AutoModel v0.6.0 (`examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml`,
`qwen3/qwen3_vl_4b_instruct_rdr.yaml`, `qwen3_8/qwen3_8_27b_lora.yaml`), YAML smoke đã chạy (S-3), ADR-001 (+am.1), P002, ModelOpt hf_ptq.

Quy ước:
- **Simple** ≤5 tham số, tên tiếng Việt, dạng enum/preset, không dùng thuật ngữ.
- **Advanced** = tên kỹ thuật, kèm khoảng hợp lệ.
- `locked` = hiện ra cho minh bạch nhưng không cho sửa (do catalog/security quyết).
- `per-model` = mặc định lấy từ entry catalog (`configs/catalog/models.yaml`), không hard-code trong node.
- Giới hạn (min/max) là chặn cứng ở validate, không chỉ là gợi ý.

## 1. Dataset
Ra: `Dataset` (format `vlm-chat`, có split train/val/test).
| Mức | Tham số | Mặc định | Lựa chọn / khoảng | Ghi chú |
|---|---|---|---|---|
| Simple | Nguồn dữ liệu | `CORD-v2 (hoá đơn mẫu)` | CORD-v2 · Thư mục của tôi (ảnh + `labels.jsonl`) · HF dataset (id) | HF id ngoài danh sách đã vet → cảnh báo license (ADR-003) |
| Simple | Tỉ lệ dữ liệu kiểm tra | 10% | 5–30% | CORD có sẵn split nên ẩn nút này |
| Simple | Giới hạn số mẫu | Tất cả | Tất cả · 100 · 500 · 1000 | "Chạy thử nhanh" = 100 |
| Advanced | `seed` | 1234 | int | dùng cho chia split + shuffle |
| Advanced | `max_image_side_px` | 4096 | 512–8192 | chặn ảnh quá lớn / decompression bomb (D7) |
| Advanced | `dedupe` | true | bool | hash ảnh + câu hỏi |
| Advanced | `revision` | locked (SHA đã vet) | — | |
| Advanced | `check_contamination` | true | bool | so trùng với split eval (G5) |

## 2. Model
Ra: `Model{origin: hub}`.
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Chọn model | **Nemotron-Parse v1.1** (template mặc định) | danh sách catalog, mỗi model có nhãn: RAM cần · license · "cần tài khoản HF" (gated) · "NVIDIA" | ẩn model có `ram_gb` > ngân sách hiện tại hoặc gắn "cần máy rảnh" |
| Advanced | `revision` | locked (catalog SHA) | — | |
| Advanced | `dtype` | `bfloat16` locked | — | GB10 dùng BF16 |
| Advanced | `attn_implementation` | per-model (Parse: `eager`; Qwen-VL/Cosmos: `sdpa`) locked | — | flash-attn không dùng trên sm_121 |
| Advanced | `trust_remote_code` | per-model locked (Parse: true, chỉ đi qua bản prepared đã pin; Qwen/Cosmos: false) | — | security quyết |

## 3. Fine-tune
Vào: `Dataset`, `Model` · Ra: `Model{Checkpoint}` hoặc `Adapter`.
| Mức | Tham số | Mặc định | Lựa chọn / khoảng | Nguồn / ghi chú |
|---|---|---|---|---|
| Simple | Cách huấn luyện | **Tự động** | Tự động · Nhẹ (LoRA) · Toàn phần | Tự động: Parse → toàn phần decoder (model 0.9B, recipe chính chủ); VLM ≥2B → LoRA. "Toàn phần" bị khoá cho model >4B (RAM) |
| Simple | Thời lượng huấn luyện | **3 lượt** (epochs) | 1 · 3 · 5 lượt, hoặc "Chạy thử 50 bước" | hiện ước lượng thời gian + RAM bên cạnh |
| Simple | Tốc độ học | **Vừa** | Chậm-an toàn · Vừa · Nhanh | full: 5e-6 / **1e-5** / 2e-5 (recipe Parse, Qwen3-VL = 1e-5); LoRA: 5e-5 / **1e-4** / 2e-4 (recipe qwen3_8 lora) |
| Simple | Lưu điểm khôi phục | Mỗi 30 phút | 15 · 30 · 60 phút | quy ra `ckpt_every_steps` theo s/step đo được |
| Advanced | `global_batch_size` | 8 | 4–64 | recipe gốc 16 (8 GPU); 1 GB10 dùng 8 với grad accumulation |
| Advanced | `local_batch_size` | **1** | 1–4 | 1 = an toàn cho máy chung; tăng → RAM tăng, preflight tính lại |
| Advanced | `max_steps` | suy từ epoch | 10–100000 | |
| Advanced | `lr_scheduler` | cosine, warmup 3% | cosine · constant | |
| Advanced | `optimizer` | AdamW β(0.9,0.95), eps 1e-8 | — | đúng recipe |
| Advanced | `weight_decay` | 0.01 | 0–0.1 | |
| Advanced | `lora.dim` / `lora.alpha` | 8 / 32 | dim 4–64, alpha 8–128 | recipe qwen3_8_27b_lora; S-3 đã chạy với 8/32 |
| Advanced | `lora.exclude_modules` | `*vision*`, `*visual*`, `*lm_head*`, `*audio*` locked | — | |
| Advanced | `freeze_vision_tower` | true | bool | Parse + Qwen-VL recipe đều freeze |
| Advanced | `activation_checkpointing` | **true** | bool | tắt → RAM tăng mạnh |
| Advanced | `loss` | per-model locked (Parse: `NemotronParseLoss`, coord weight 10, ngưỡng đọc từ config; khác: MaskedCE / FusedLinearCE) | — | |
<!-- freeze_embeddings removed 2026-10-12: not an AutoModel key, no model needs it (001a §9.3) -->
| Advanced | `min_pixels` / `max_pixels` | Qwen-VL/Cosmos: 256·28² / 1280·28²; **Parse: không có** (ảnh cố định 2048×1664) | 128·28²–2048·28² | phải khớp với node Chạy thử/Evaluate (gate parity) |
| Advanced | `max_recent_checkpoints` | 2 | 1–5 | disk |
| Advanced | `val_every_steps` | = ckpt | | |
| Advanced | `seed` | 1234 | | |
| Advanced | `save_consolidated` | `final` | final · every | `every` tốn disk |

## 4. Evaluate (độ chính xác)
Vào: `Model` (+ `Dataset` tuỳ chọn) · Ra: `EvalReport`.
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Đánh giá trên | **Tự động theo mẫu** | Parse: "Hoá đơn (CORD test)"; VLM: nhóm năng lực (Đọc tài liệu · Biểu đồ & sơ đồ · Kiến thức đa ngành · Nhận thức chung · Toán có hình · Video) · Dữ liệu kiểm tra của tôi | P002 A1 / P010 |
| Simple | Mức độ | **Nhanh** | Nhanh (~50 mẫu/bộ, ≤15') · Tiêu chuẩn (500/bộ, ≤1.5h) · Đầy đủ (≤6h) | P002 B5 |
| Simple | So với | Model gốc | Không so · Model gốc · Kết quả trước | so sánh paired + CI95 |
| Advanced | `thinking` | false | bool | |
| Advanced | `max_new_tokens` | 1024 (MC 128; Parse/CORD 1536 ❓ theo độ dài JSON) | 64–16384 | |
| Advanced | `decoding` | greedy | greedy · sampling (chỉ G1) | |
| Advanced | `judge_model` | none | none · local judge (pin) | chỉ cho benchmark judge-based |
| Advanced | `subset_ids` | bộ cố định của repo | locked | |

## 5. Chạy thử (inference)
Vào: `Model` · Ra: `Endpoint` (+ kết quả hiển thị).
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Ảnh / video | (user tải lên) | jpg/png; video mp4 (chỉ model hỗ trợ video) | |
| Simple | Yêu cầu | Parse: "Trích xuất hoá đơn" (cố định); VLM: ô câu hỏi | | |
| Simple | Độ dài trả lời | Vừa | Ngắn (256) · Vừa (1024) · Dài (4096) | Cosmos khuyên ≥4096 khi suy luận |
| Advanced | `engine` | vLLM (nếu model hỗ trợ) → fallback HF generate | vllm · hf | Parse: vLLM ❓ (P-4) |
| Advanced | `temperature` / `top_p` | 0 / 1.0 | 0–1.5 / 0.1–1 | |
| Advanced | decoding mặc định | lấy từ `generation_config.json` của model (Parse: greedy + `repetition_penalty` 1.1) | locked trong Simple | không tự đặt greedy thuần khác cấu hình chuẩn của model (001a §10.5) |
| Advanced | `gpu_memory_utilization` | **0.3** | 0.2–0.5 | >0.3 chỉ khi preflight cho phép (ADR-001) |
| Advanced | `max_model_len` | 8192 | 2048–32768 | |
| Advanced | `mm_processor_kwargs` | lấy từ node Fine-tune / catalog, locked | | parity với lúc train |
| Advanced | `bind` | 127.0.0.1 locked | | security |

## 6. Quantize FP8
Vào: `Model` (+ `Dataset` để hiệu chỉnh) · Ra: `Model{QuantizedModel}`.
Chỉ bật cho model có `quant_formats.fp8 = official` (Cosmos/Qwen-VL). **Parse: ẩn** cho tới khi P-5 có kết quả.
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Định dạng nén | **FP8** | FP8 · (NVFP4 "thử nghiệm", ẩn trong v1) | ModelOpt: Qwen3-VL/3.5 chỉ FP8 chính thức |
| Simple | Số ảnh hiệu chỉnh | 512 | 128 · 512 | hf_ptq `--calib_size`, `--calib_with_images` |
| Simple | Đánh giá sau khi nén | Có (Nhanh) | Có · Không | gate FP8 ≤1pt (P002 B3) |
| Advanced | `quantize_vision` | false | bool | vision tower giữ BF16 mặc định |
| Advanced | `kv_cache_quant` | none | none · fp8 | |
| Advanced | `export_format` | HF unified checkpoint | locked | để vLLM/TRT-LLM load |
| Advanced | `calib_source` | split train của Dataset đầu vào | train · val | không dùng split test |

## 7. Benchmark (tốc độ)
Vào: `Model` hoặc `Endpoint` · Ra: `BenchReport`.
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Kịch bản | **Một người dùng** | Một người dùng (concurrency 1) · Nhiều người (4) | preset cố định, không tự do (P002 G4) |
| Simple | So với | Model gốc | Không · Model gốc · Bản chưa nén | |
| Advanced | `num_requests` | 50 | 10–500 | |
| Advanced | `warmup_requests` | 5 | locked ≥3 | bỏ JIT warmup (~25s lần đầu NVFP4) |
| Advanced | `image_size` | 1024×1024 (Parse: 2048×1664) | locked theo preset | |
| Advanced | `output_tokens` | 256 | 64–2048 | |
| Advanced | `gpu_memory_utilization` | 0.3 | như node 5 | |

## 8. Export
Vào: `Model` / `Adapter` · Ra: `ExportArtifact`.
| Mức | Tham số | Mặc định | Lựa chọn | Ghi chú |
|---|---|---|---|---|
| Simple | Xuất dạng | **Model hoàn chỉnh** | Model hoàn chỉnh (đã gộp LoRA) · Chỉ phần học thêm (LoRA, nhỏ) · Bản đã nén | |
| Simple | Lưu vào | `outputs/exports/<tên>` | chọn thư mục trong project | chỉ trong project (quy tắc an toàn) |
| Advanced | `merge_lora` | true khi chọn "Model hoàn chỉnh" | | |
| Advanced | `format` | safetensors locked | | không pickle |
| Advanced | `include_license_notice` | **true locked** | | NVIDIA OML: kèm license + "Built on NVIDIA Cosmos" (Cosmos); CORD: attribution CC BY 4.0 |
| Advanced | `push_to_hub` | ẩn trong v1 | | |

## 9. Mặc định an toàn chung cho GB10 (mọi node chạy GPU)
- Preflight bắt buộc: MemAvailable ≥ `ram_gb` (cận trên) + 10GB headroom; disk; GPU_LOCK rảnh, nếu không thì vào hàng đợi.
- Không có tham số nào cho phép tắt memguard / preflight.
- Ước lượng RAM/thời gian hiện trước khi chạy: lấy từ catalog (`source=measured` ưu tiên hơn `estimated`). Thiếu số đo thì hiện "ước lượng".
- ❓ = cần đo hoặc xác minh ở P-1/P-2/P-4 trước khi coi là cố định.
