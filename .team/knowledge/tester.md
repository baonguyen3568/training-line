# Knowledge — Tester / QA (bao-tester)
Cập nhật: 2026-10-12. Phạm vi: VLM fine-tune/quantize/eval trên DGX Spark GB10 (aarch64, 128GB unified memory, CUDA 13, driver 580).
Mức tin cậy: [O] = docs chính chủ, [P] = paper, [F] = forum/blog (cần tự xác minh trên máy mình).

---
## 1. Nguyên tắc chủ đạo (opinion của QA, có lý do)
1. **Sanity check base model trước, finetune sau.** Chạy harness của mình trên base model, so với số công bố
   (model card / OpenVLM leaderboard). Lệch > ngưỡng (đề xuất: ±1.5 pt cho MC benchmark, ±2 pt cho benchmark
   dùng judge) → harness có vấn đề, KHÔNG được báo kết quả finetune. (PM cũng đặt bar này.)
2. **So sánh luôn là paired, cùng harness, cùng commit, cùng prompt, cùng decoding.** Không bao giờ so số mình chạy
   với số copy từ paper khi kết luận "finetune tốt hơn".
3. **Một nguồn sự thật cho eval config**: harness version, task version, prompt template, chat template, max_new_tokens,
   temperature (=0 / greedy), seed, image resolution / max_pixels, answer extractor (rule vs LLM judge + judge model).
   Mọi kết quả JSON phải embed các field này + git SHA + container digest.
4. **Báo CI, không chỉ point estimate.** VQA benchmarks có variance đáng kể [P: arXiv 2508.02645]; lmms-eval ≥0.6 có
   CI + paired t-test [O]. Quy tắc: chênh lệch < CI 95% ⇒ "không khác biệt".
5. **Test nhỏ, nhanh, deterministic trước; test nặng có GPU lock.** Unit/integration chạy CPU hoặc model tí hon,
   không cần lock; smoke E2E/benchmark ⇒ bắt buộc `.team/GPU_LOCK`.

## 2. Công cụ eval accuracy
| Tool | Ghi chú | Nguồn |
|---|---|---|
| **VLMEvalKit** (open-compass) | Generation-based cho mọi model, báo cả exact-match và LLM-extraction; 330+ benchmark; README pin transformers theo từng họ model → **pin version** | [P] arXiv 2407.11691 |
| **lmms-eval** v0.7 (02/2026) | Chat models dùng `apply_chat_template()` (khuyên dùng, path `lmms_eval/models/chat/`); backend vLLM/SGLang/OpenAI-compatible; v0.6 thêm CI + paired t-test, HTTP eval server; v0.7 thêm token count/throughput per-sample. Dùng `--limit` để smoke | [O] github EvolvingLMMs-Lab/lmms-eval README |
| **NeMo Evaluator** (+ launcher) | Container `nvcr.io/nvidia/eval-factory/vlmevalkit` (tag 25.08.1, 26.01…) — task: `ai2d_judge, chartqa, mathvista-mini, mmmu_judge, ocr_reasoning, ocrbench, slidevqa`; gọi model qua endpoint OpenAI-compatible (`model_type: vlm`). `nemo-evaluator-launcher ls tasks` | [O] NGC catalog, docs.nvidia.com/nemo/evaluator |

Khuyến nghị sơ bộ (sẽ chốt trong Proposal 002 với bao-research):
- **Primary**: eval qua endpoint OpenAI-compatible (vLLM/TRT-LLM serve) ⇒ cùng 1 harness cho BF16 / FP8 / NVFP4 và
  cũng là đường phục vụ thật. Cần kiểm tra container eval-factory có **arm64** không (chưa xác minh!).
- **Fallback/đối chiếu**: lmms-eval chat backend chạy local HF để cross-check 1 benchmark (phát hiện lỗi serve/template).
- Judge-based metric (mmmu_judge, ai2d_judge…) phụ thuộc judge model → ghi rõ judge; ưu tiên rule-based extractor cho
  regression nội bộ (rẻ, deterministic), judge cho số báo cáo.

## 3. Pitfalls eval VLM (đã gặp nhiều trong cộng đồng)
- **Chat template / system prompt sai** → tụt vài-chục điểm. Luôn dùng `processor.apply_chat_template`; test unit so
  token ids với template gốc của model.
- **Image resolution / max_pixels / số tile** khác với công bố → lệch số (Qwen-VL nhạy với `min_pixels/max_pixels`).
- **Answer extraction**: model không in đúng chữ cái option → exact-match đánh trượt. Judge GPT-4 khớp người ~91.5%
  [P: ECCV'24] ⇒ judge cũng có lỗi. Log toàn bộ raw output (`log_samples`) để audit.
- **Contamination / text-only shortcut**: GeminiPro đạt 42.9% MMMU không cần ảnh [P: MMStar]. Thêm **blind test**
  (bỏ ảnh) trên tập eval của dữ liệu finetune riêng để phát hiện model học "mẹo" chữ.
- **Train/eval leakage**: dataset finetune có thể chứa ảnh/câu hỏi của benchmark → dedup (hash ảnh pHash + text) giữa
  train set và eval set, test tự động.
- **Thinking/reasoning models**: max_new_tokens quá thấp cắt mất đáp án; phải parse sau `</think>`.
- **Decoding**: greedy, `temperature=0`; vLLM vẫn có non-determinism nhỏ do batching → chạy 2 lần đo độ lệch.
- **Split**: MMMU val vs test (test không có label), MathVista testmini, MMBench dev vs test — ghi rõ.
- **Transformers version** ảnh hưởng output (VLMEvalKit cảnh báo) → pin.

## 4. Quantization QA (ModelOpt → vLLM/TRT-LLM)
- ModelOpt 0.46/0.47 [O changelog]: recipe FP8 cho ViT; recipe FP8 vision-encoder cho qwen3_vl/qwen3_5 (vision-only
  hoặc joint); AutoQuantize recipe để **vision tower unquantized**, LM search NVFP4/FP8. ⇒ mặc định giữ vision tower BF16
  hoặc FP8; NVFP4 chỉ LM.
- NVFP4 có thể tụt mạnh trên benchmark reasoning (vd model card bên thứ 3: GPQA −9.1pt) [F] ⇒ **mỗi quant format phải
  qua full accuracy suite**, không chỉ perplexity.
- Option `nvfp4_mlp_only` (attention giữ BF16) là phương án khi NVFP4 full tụt quá ngưỡng [F Jetson AI lab].
- vLLM load ModelOpt: `quantization="modelopt"` (FP8) / `"modelopt_fp4"` (NVFP4) [O docs.vllm.ai].
- Ngưỡng chấp nhận đề xuất: FP8 ≤ 1pt tụt trung bình; NVFP4 ≤ 2–3pt và không benchmark nào > 5pt (chốt trong P002).
- Calibration set: dùng ảnh+text domain thật, **tách biệt** với eval set (test leakage).

## 5. Performance testing trên DGX Spark (unified memory!)
- **`nvidia-smi` báo Memory-Usage "Not Supported"** trên GB10 — không phải lỗi [F nhiều nguồn]. Đo memory bằng
  `/proc/meminfo`: used = MemTotal − MemAvailable; sample 1s. Trong process: `torch.cuda.max_memory_allocated()`
  (allocator-level, chưa gồm cache/JIT) + RSS hệ thống. Báo **cả hai**.
- Máy dùng chung (~70–77GB đã dùng sẵn) ⇒ baseline memory phải đo ngay trước run và trừ đi; noise lớn ⇒ chạy ≥3 lần,
  báo median + p90; ghi load của host (các container khác) vào metadata.
- vLLM: `gpu_memory_utilization` mặc định 0.9 sẽ pre-allocate KV cache cực lớn → trên máy chung có thể OOM cả host.
  Dùng ≤0.3–0.5 tùy model, và so sánh trước/sau quantize **ở cùng giá trị** [F forum "Memory creep on DGX Spark"].
- CUDA graphs / torch.compile thêm ~13GB cho ~3% tốc độ [F] → ghi rõ eager vs graph khi benchmark.
- FlashInfer JIT trên SM121 tạo spike 20–30GB lúc khởi động [F] → peak memory đo **sau warmup** riêng với **startup peak**.
- FP4 CUTLASS kernel có thể fallback trên SM121 (chậm hơn, tốn RAM hơn) [F] → log backend kernel thực sự dùng.
- Không `drop_caches` (cần sudo, cấm) → thay bằng ghi lại `Cached` trong meminfo làm metadata.
- Metric serve: TTFT, TPOT/ITL, E2E latency (p50/p90/p99), request & output token throughput, ở concurrency 1/4/16;
  công cụ `vllm bench serve` (dataset `random_mm` / `custom_mm`), `vllm bench mm-processor` cho preprocessing;
  TRT-LLM `benchmark_serving.py` có random image dataset [O].
- Ảnh cố định kích thước (vd 1024×1024 và 448×448) — số token ảnh ảnh hưởng TTFT mạnh.

## 6. Môi trường aarch64 / CUDA 13
- GB10 chỉ hỗ trợ CUDA 13.0+; nhiều wheel pip build cho CUDA 12.x → crash/không có kernel sm_121 [F]. pip có thể
  âm thầm downgrade torch → dùng `uv` + index cu130 hoặc container NGC [F].
- Test môi trường (env smoke): `torch.cuda.is_available()`, `get_device_capability()==(12,1)`, matmul bf16 nhỏ,
  import modelopt/vllm/transformers không lỗi, kiểm tra `platform.machine()=='aarch64'`.

## 7. Chiến lược test cho repo (test pyramid)
| Tầng | Nội dung | Chạy ở | Thời gian |
|---|---|---|---|
| Unit | config pydantic (validate/reject), data formatter (LLaVA/HF chat JSONL), chat-template rendering, answer extractor, metric (so với ví dụ chuẩn của benchmark), dedup | CPU, `pytest -m "not gpu"` | < 1 phút |
| Integration | train 2–5 step trên `hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration` (~10MB) + 8 mẫu: loss giảm/finite, save/load checkpoint, LoRA merge; quantize tiny model chạy hết pipeline | CPU/GPU nhẹ, marker `integration` | < 5 phút |
| Smoke E2E | data mini → SFT LoRA vài chục step model nhỏ thật (2–4B) → eval `--limit 20` → quantize FP8 → serve → 1 request | GPU + lock, marker `gpu`/`slow` | < 30–60 phút |
| Regression | full benchmark suite base vs finetune vs quant, lưu `benchmarks/*.json`, so ngưỡng | GPU + lock | giờ |
- Golden tests: tập ~50 câu cố định + output kỳ vọng của base model (greedy) → phát hiện thay đổi hành vi do đổi deps.
- Overfit test: model nhỏ phải overfit 8 mẫu (loss → ~0) — bắt lỗi label masking/collator.
- Label masking test: chỉ token assistant có label ≠ −100; token ảnh/ prompt bị mask.
- Reproducibility test: 2 lần train cùng seed (CPU tiny) ⇒ loss giống nhau tới 1e-6.

## 8. Checklist "trước khi tin một con số"
- [ ] Base model reproduce số công bố trong ngưỡng
- [ ] Config eval đầy đủ được embed trong JSON kết quả (harness ver, SHA, container digest, prompt, decoding, resolution, judge)
- [ ] Không leakage train↔eval (dedup report)
- [ ] Raw outputs được lưu; spot-check ≥20 mẫu bằng mắt
- [ ] CI 95% / paired test; chạy lặp nếu non-deterministic
- [ ] Perf: đo sau warmup, baseline memory host ghi lại, cùng gpu_memory_utilization, ≥3 lần
- [ ] GPU lock tạo/xoá đúng

## Nguồn
- VLMEvalKit paper: https://arxiv.org/abs/2407.11691
- lmms-eval README: https://github.com/EvolvingLMMs-Lab/lmms-eval ; paper https://arxiv.org/abs/2407.12772
- NeMo Evaluator: https://docs.nvidia.com/nemo/evaluator/latest/about/key-features.html ;
  NGC vlmevalkit container: https://catalog.ngc.nvidia.com/orgs/nvidia/eval-factory/containers/vlmevalkit/latest ;
  legacy containers: https://docs.nvidia.com/nemo/evaluator/0.3.0/tutorials/legacy-containers.html
- Variance VQA: https://arxiv.org/abs/2508.02645
- MMStar ("Are we on the right way..."): https://liner.com/ko/review/are-we-on-the-right-way-for-evaluating-large-visionlanguage
- ModelOpt changelog: https://nvidia.github.io/Model-Optimizer/ ; release 0.46.0: https://newreleases.io/project/github/NVIDIA/Model-Optimizer/release/0.46.0
- vLLM ModelOpt: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- Jetson ModelOpt tutorial: https://www.jetson-ai-lab.com/tutorials/modelopt-on-jetson
- vLLM bench: https://docs.vllm.ai/projects/spyre/en/latest/user_guide/performance.html
- TRT-LLM serve benchmark: https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html
- DGX Spark memory creep: https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
- DGX Spark 56GB VRAM thread: https://forums.developer.nvidia.com/t/dgx-spark-gb10-shows-only-56gb-vram-inside-ai-workbench-128gb-expected/351441
- GB10 software compat: https://github.com/natolambert/dgx-spark-setup ; https://docs.technotim.com/posts/ubuntu-gb10/
- DGX Spark VLM finetuning playbook: https://build.nvidia.com/spark/vlm-finetuning (Qwen2.5-VL-7B GRPO, InternVL3-8B; chỉ so sánh side-by-side, không có metric định lượng ⇒ ta phải tự thêm eval)
- Tiny model: https://huggingface.co/hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration

## Câu hỏi mở (cần xác minh)
- Container `eval-factory/vlmevalkit` có manifest arm64 không? (sẽ `docker manifest inspect`, không pull)
- Base model chốt (P001) → lấy số công bố + protocol chính xác của họ.
