# Proposal 002 — Eval / Benchmark protocol
Đồng chủ trì: bao-research (Phần A: bộ benchmark & protocol đối chiếu) + bao-tester (Phần B: harness, tái lập, gate, smoke, perf).
Trạng thái: DRAFT (2026-10-12). Quy tắc sửa: mỗi bên chỉ sửa phần của mình; góp ý phần kia → ghi vào mục "Góp ý" cuối file.
Phụ thuộc: Proposal 001 (base model + container). Tham khảo: `.team/knowledge/tester.md`.

---
## Phần A — Bộ benchmark & protocol đối chiếu số công bố (owner: bao-research)
_TODO bao-research: danh sách benchmark + split + revision, metric chính thức, prompt template, max_new_tokens,
thinking on/off, max_pixels, judge model (local, pin), subset size cho smoke/regression, số công bố tham chiếu + nguồn._

Đã thống nhất sơ bộ (16:2x, research ⇄ tester):
- Gate reproduce chỉ dùng benchmark **rule-scored**: DocVQA (ANLS), ChartQA (relaxed acc), OCRBench, AI2D, MMMU (MC), MMBench (EM).
- Benchmark judge-based: judge **local, pin model**, ghi rõ "judge ≠ judge của paper" ⇒ chỉ dùng so sánh nội bộ.
- OCRBench/DocVQA báo riêng trong quant gate (nhạy nhất với quant).

---
## Phần B — Harness, tái lập, gate, smoke, perf (owner: bao-tester)

### B1. Kiến trúc harness
```
model (bf16 | fp8 | nvfp4)  ──serve──►  OpenAI-compatible endpoint (vLLM / TRT-LLM)  ──►  harness (primary)
        │                                                                                 │
        └────────── in-process HF (lmms-eval chat backend) ── cross-check ────────────────┘
                                                     ▼
                            outputs/<run_id>/eval/{samples.jsonl, metrics.json}  →  benchmarks/accuracy/*.json
```
- **Primary harness**: NeMo Evaluator `eval-factory/vlmevalkit` container *nếu* có arm64 (bao-research đang xác minh);
  nếu không ⇒ **lmms-eval v0.7** (pin commit) với backend OpenAI-compatible. Quyết định cuối sau xác minh, ghi vào ADR.
- **Cross-check path**: lmms-eval chat model in-process (HF transformers, `apply_chat_template`). Chỉ dùng cho gate
  reproduce base model và khi debug lệch số.
- Lý do chọn endpoint làm primary: BF16/FP8/NVFP4 dùng cùng một đường ⇒ so sánh quant công bằng, và đó là đường serve thật.
- Server phải pin: `--mm-processor-kwargs` (min/max_pixels), chat template (hash), `--max-model-len` đủ cho thinking,
  `--gpu-memory-utilization` (cố định cho mọi biến thể), seed, eager/cuda-graph.

### B2. Bản ghi kết quả (bắt buộc — thiếu field ⇒ INVALID)
`benchmarks/accuracy/<model>-<precision>-<task>-<date>.json` (schema chung với bao-frontend/backend, `schema_version`):
harness+commit, task+split+dataset revision, container digest, git sha, model path+revision, precision (LM + vision tower),
chat_template_hash, system/prompt template, decoding (greedy, max_new_tokens), seed, min/max_pixels, answer_extractor
(rule | judge+model), n_samples, score, CI95, path tới samples.jsonl (raw output từng mẫu).

### B3. Các gate
| Gate | Khi nào | Điều kiện PASS |
|---|---|---|
| **G0 Env smoke** | mỗi container/deps mới | aarch64, `torch.cuda.is_available()`, capability (12,1), matmul bf16, import torch/transformers/vllm/modelopt OK |
| **G1 Reproduce (base)** | 1 lần / base model / harness version | Rule-scored benchmark: \|ours − published\| ≤ 1.5pt, trên **cả 2 đường**; 2 đường lệch nhau ≤ 1pt (nếu không ⇒ lỗi serve/preprocess) |
| **G2 Regression (finetune)** | mỗi checkpoint ứng viên | Không benchmark general nào tụt quá CI95 so với base (catastrophic forgetting); benchmark target tăng có ý nghĩa (paired, CI95 không chứa 0) |
| **G3 Quant** | mỗi checkpoint quant | FP8: tụt TB ≤ 1pt; NVFP4: tụt TB ≤ 2–3pt, không benchmark nào > 5pt, OCRBench/DocVQA ≤ 3pt; vision tower precision ghi rõ |
| **G4 Perf** | cùng G3 + khi đổi serve config | Báo TTFT/TPOT/E2E p50/p90/p99, tok/s, memory; quant phải nhanh hơn hoặc ít memory hơn BF16 một cách có ý nghĩa (≥3 lần đo, median) — không thì không đáng deploy |
| **G5 Leakage** | mỗi dataset train mới | Overlap train↔eval (sha256 + pHash ảnh, text chuẩn hoá) = 0, hoặc đã loại bỏ và ghi lại |
Số G1/G3 là đề xuất ban đầu — chốt trong ADR.
Tiền điều kiện cho mọi gate trên một checkpoint đã biến đổi (convert HF⇄Megatron, LoRA merge, quantize export):
**R6 của Proposal 007** (test tính đúng của biến đổi). Ngưỡng tester đề xuất cho R6: convert cùng dtype phải bit-exact;
logits so bằng cosine/KL, không dùng atol tuyệt đối; greedy ~16 prompt có ảnh, chấp nhận lệch tại near-tie; quantize
assert vision precision == config, mini-eval 50 mẫu chỉ là smoke, gate thật là G3.
GPU lock cho test integration/gpu: do tiến trình host giữ (check.sh/run.py), truyền token vào container (theo P004).

### B4. Thống kê
- Per-sample paired comparison; Δ + CI95 bằng bootstrap (1000 lần, seed cố định) hoặc paired t-test lmms-eval.
- Δ trong CI ⇒ "không khác biệt". Không bao giờ so số mình chạy với số copy từ paper khi kết luận ft/quant.
- Non-determinism vLLM: mỗi harness version, chạy lặp 2 lần 1 benchmark để đo spread; spread > 0.5pt ⇒ điều tra.

### B5. Tầng chạy & ngân sách thời gian (máy dùng chung)
| Tầng | Nội dung | Ngân sách | GPU lock |
|---|---|---|---|
| Smoke-eval | 3–4 benchmark × `--limit 20–50` | ≤ 15 phút | có |
| Regression | subset cố định (~300–500 mẫu/benchmark, id list commit vào repo) | ≤ 1.5h | có |
| Full | toàn bộ split | ≤ 6h, chạy ngoài giờ cao điểm | có |
Subset regression chọn ngẫu nhiên có seed + stratify theo category (bao-research chọn), file id list versioned.

### B6. Test pyramid cho code (layout theo Proposal 004 của bao-cleancode)
- `tests/unit` (CPU): config pydantic, data formatter, chat template rendering (so token ids với processor gốc),
  label masking (chỉ token assistant có label), answer extractor + metric (ANLS, relaxed acc, MC parse) với ví dụ chính thức
  của benchmark, dedup, bootstrap CI, gpu_lock (race, stale PID, không xoá lock người khác).
- `tests/integration` (CPU, tiny model `hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration`):
  train 2–5 step loss finite, overfit 8 mẫu, save/load, LoRA merge, reproducibility cùng seed.
- `tests/gpu` (lock): smoke E2E Qwen3.5-0.8B (đề xuất backend): data mini → LoRA vài chục step → eval `--limit 20`
  → quantize FP8 → serve → 1 request; + test token count ảnh HF processor == vLLM cho cùng ảnh.
- Golden test: ~50 prompt cố định + output greedy kỳ vọng của base model ⇒ phát hiện đổi hành vi do bump deps.

### B7. Perf protocol (tóm tắt — chi tiết `.claude/skills/tester-perf-bench`)
- Memory GB10: `nvidia-smi` không báo memory ⇒ đo `/proc/meminfo` (MemTotal − MemAvailable) 1Hz + `torch.cuda.max_memory_allocated`;
  báo baseline / startup peak (JIT) / steady / peak under load.
- Đo sau warmup; cùng `gpu_memory_utilization` (≤0.5), cùng eager/graph, ảnh 448² & 1024², concurrency 1/4/16, ≥3 lần, median + spread.
- Ghi metadata noise host: containers khác đang chạy, MemAvailable lúc bắt đầu.
- Tool: `vllm bench serve` (random-mm/custom-mm) hoặc TRT-LLM `benchmark_serving.py` — pin version.

### B8. Rủi ro / câu hỏi mở
1. Container eval-factory có arm64? (research xác minh) — ảnh hưởng chọn primary harness.
2. Judge local nào, đủ tốt không — cần đo agreement với tập nhỏ chấm tay (~100 mẫu) trước khi dùng.
3. RAM khả dụng ~40GB: server eval + judge cùng lúc có thể không vừa ⇒ chạy tuần tự (generate trước, judge sau).
4. Số công bố của base model có thể dùng protocol không public đầy đủ ⇒ G1 có thể phải nới cho benchmark cụ thể, ghi lý do.

---
## Góp ý
_(ghi `[tên] nội dung`)_
