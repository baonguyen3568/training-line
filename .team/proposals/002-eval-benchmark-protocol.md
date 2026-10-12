# Proposal 002 — Eval / Benchmark protocol
Đồng chủ trì: bao-research (Phần A: bộ benchmark & protocol đối chiếu) + bao-tester (Phần B: harness, tái lập, gate, smoke, perf).
Trạng thái: REVIEW (2026-10-12) — Phần A + B xong, chờ PM chốt ADR. Quy tắc sửa: mỗi bên chỉ sửa phần của mình; góp ý phần kia → ghi vào mục "Góp ý" cuối file.
Phụ thuộc: Proposal 001 (base model + container). Tham khảo: `.team/knowledge/tester.md`.

---
## Phần A — Bộ benchmark & protocol đối chiếu số công bố (owner: bao-research)
Viết 2026-10-12. Task name lmms-eval cần xác minh lại trên bản 0.7.3 khi cài (❓). Số công bố lấy từ model card (P001).

### A1. Bộ benchmark chuẩn (registry dữ liệu, không hard-code trong UI)
| Benchmark | Split (task lmms-eval ❓) | n | Metric chính thức → thang 0-100 | Chấm | Gate reproduce? | Nhóm năng lực (P010) | higher_is_better |
|---|---|---|---|---|---|---|---|
| MMMU | val (`mmmu_val`) | 900 | accuracy MC | rule (trích chữ cái); judge tùy chọn | ✅ | Kiến thức & suy luận đa ngành | true |
| MMBench-EN (v1.1 cho G1; v1.0 cho so sánh nội bộ, xem ghi chú) | dev (lmms `mmbench_en_dev` = **v1.0**; v1.1 = VLMEvalKit `MMBench_DEV_EN_V11`) | ~4.3k | accuracy (circular eval nếu harness hỗ trợ) | EM; judge tùy chọn | ✅ (EM) | Nhận thức tổng quát | true |
| DocVQA | val (`docvqa_val`) | 5,349 | ANLS ×100 | rule | ✅ | Đọc tài liệu | true |
| ChartQA | test (`chartqa`) | 2,500 | relaxed accuracy (±5%) | rule | ✅ | Hiểu biểu đồ & sơ đồ | true |
| OCRBench (v1) | test (`ocrbench`) | 1,000 | score/1000 → /10 | rule | ✅ | Đọc tài liệu | true |
| AI2D | test (`ai2d`) | 3,088 | accuracy | rule | ✅ | Hiểu biểu đồ & sơ đồ | true |
| MathVista | testmini (`mathvista_testmini`) | 1,000 | accuracy | **judge/extractor** | ❌ (chỉ nội bộ) | Toán có hình | true |
| Video-MME | w/o subs (`videomme`) | 2,700 QA | accuracy MC | rule | ❌ (đắt; release only) | Video | true |
| **CORD-v2** (template mặc định) | test (`cord_v2_test`, held-out, không bao giờ train) | 100 | **field_f1** (chính) · json_exact · nted_accuracy · parse_fail_rate; thang 0-100; theo đúng Donut (token2json + JSONParseEvaluator, **chỉ strip 2 đầu**, không gộp khoảng trắng giữa) | rule | — (tham khảo Donut ~84 F1 ❓) | Đọc tài liệu | true; parse_fail_rate=false |
| Perf | — | — | TTFT/TPOT/e2e ms, peak_mem GB, disk GB | đo | — | — | false |
| Perf | — | — | tokens/s, req/s | đo | — | — | true |
Tầng chạy (khớp B5): **smoke** = 50 mẫu/benchmark cố định (seed, danh sách id commit vào repo) × {MMBench, ChartQA, OCRBench, DocVQA}; **regression** = 500 mẫu cố định × 6 benchmark rule-scored; **full** = toàn bộ split + MathVista + Video-MME.

Ghi chú A1 (cập nhật 23:20, bao-tester phát hiện):
- **MMBench**: `lmms-lab/MMBench` là **v1.0**. N-4 và regression nội bộ (paired) dùng v1.0. **G1** (đối chiếu card Qwen3.5, ghi v1.1) bắt buộc dùng v1.1 (TSV `MMBench_DEV_EN_V11` của OpenCompass/VLMEvalKit). Nguồn này **cần security vet** trước khi tải. Chấm **circular** theo nhóm biến thể xoay vòng: 500 câu → 1.865 dòng.
- **Subset regression** (500/benchmark, phân tầng, seed 20261012, khoá theo id + sha256 câu hỏi): `configs/eval/subsets/regression_500/*.json`, tạo bằng `scripts/qa/make_regression_subsets.py`. lmms-eval không lọc được theo id, nên đóng gói mỗi subset thành dataset parquet đã lọc + task YAML riêng (pin hash). Bao-backend làm cùng tester.
- **MMBench v1.1** (security, 23:40): APPROVED về nguồn và license. Nguồn: VLMEvalKit@21e52b8d `image_mcq.py` L67, URL `https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv`, md5 `30c05be8f2f347a50be25aa067248184`; eval_only. **Host timeout từ máy này** nên tạm chưa tải được. Không dùng re-upload của bên thứ 3. Hệ quả: **G1 (đối chiếu card Qwen3.5) tạm hoãn**. Sau RESET, Qwen3.5 chỉ còn là catalog phụ, nên G1 không chặn template NVIDIA; G1-harness với Qwen3-VL-2B (OpenVLM leaderboard) vẫn làm được nếu leaderboard dùng benchmark ta có. Muốn bỏ hoãn thì thử lại sau, hoặc user tải từ mạng khác rồi security verify md5 + sha256.
- CORD scorer: `src/vlm_pipeline/eval/cord.py` (port Donut, nTED Zhang–Shasha khớp `zss` 400/400). Parse-fail tính là dự đoán rỗng.

### A2. Protocol cố định (mọi run ghi đủ vào record)
| Tham số | Mặc định | Lý do |
|---|---|---|
| Chat template | template của processor HF tại đúng revision model | lệch template là nguồn sai số lớn nhất |
| Thinking | **off** cho regression/quant/so sánh nội bộ; **on** chỉ cho gate reproduce khi card báo số thinking | Qwen3.5 card mặc định thinking (~32K tokens) → off để đủ thời gian trên máy chung |
| max_new_tokens | off: 1024 (MC/short-answer: 128) · on: 16384 | thinking cắt cụt → sai đáp án |
| Decoding | greedy (temperature 0), seed 1234. **Ngoại lệ G1**: dùng sampling params của card, 3 seed (A3) | tái lập |
| Ảnh | `min_pixels` 256·28·28, `max_pixels` 1280·28·28 cho eval (khác train) — pin cả phía vLLM (`--mm-processor-kwargs`) | OCR/DocVQA rất nhạy độ phân giải |
| Video | 2 fps, tối đa 64 frame (card Qwen3.5: 2 fps) | — |
| Prompt | prompt mặc định của task lmms-eval; prompt model-specific (nếu card công bố) ghi thành `prompt_variant` | — |
| Judge (MathVista, MMMU/MMBench judge mode) | LLM local, pin repo@SHA, chạy qua vLLM; ghi `judge_model` | không gọi API ngoài; số judge-based **không so** với card |

### A3. Số công bố tham chiếu cho gate reproduce (nguồn: model card HF, đọc lại 2026-10-12)
Card Qwen3.5 nhỏ có **cả cột thinking và non-thinking**. G1 mặc định dùng cột non-thinking (theo đề xuất của tester).
| Model | MMMU | MMBench-EN-dev **v1.1** | OCRBench | AI2D-test | MathVista-mini | MMStar | RealWorldQA |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B non-thinking | 64.2 | 81.3 | 85.4 | 81.5 | 73.9 | 68.0 | 71.2 |
| Qwen3.5-2B thinking | 64.2 | 83.3 | 84.5 | 83.3 | 76.7 | 71.7 | 74.5 |
| Qwen3.5-0.8B non-thinking | 47.4 | 68.0 | 79.1 | 68.7 | 58.6 | 55.9 | 61.6 |
| Qwen3.5-0.8B thinking | 49.0 | 69.9 | 74.5 | 69.9 | 62.2 | 58.3 | 63.4 |
| Qwen3.5-4B / 9B (thinking) | 77.6 / 78.4 | 89.4 / 90.1 | 85.0 / 89.2 | 89.6 / 90.2 | 85.1 / 85.7 | — | — |
Thiết lập trên card:
- **Sampling, không phải greedy.** Non-thinking: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. Thinking: temperature 0.6 (top_p 0.95). Video: temperature 1.0, fps 2.
- Card không ghi max tokens, độ phân giải ảnh hay prompt riêng cho từng benchmark. Khuyến nghị chung là 32,768 output tokens.
- MMBench là **v1.1** (≠ v1.0). Phải chọn đúng task.
- DocVQA và ChartQA **không có** trên card.

Quyết định G1 (đã thống nhất với tester, chi tiết điều kiện ở B3):
1. **G1 chạy full split**, không chạy subset (subset 500 có CI ±3.5pt nên ngưỡng ±1.5 vô nghĩa).
2. **G1-nonthinking**: Qwen3.5-2B, cột non-thinking. Benchmark: MMMU, MMBench v1.1, OCRBench, AI2D. Dùng **đúng sampling params của card + 3 seed**, báo mean (sampling 1 seed có nhiễu). max_new_tokens 4096.
3. **G1-thinking** (tùy chọn, đắt): chỉ MMMU (900) + OCRBench (1000), max_new_tokens 16384.
4. Regression/quant nội bộ vẫn dùng greedy (A2). Sampling chỉ dùng cho G1 vì phải khớp card. **G1-internal (HF vs vLLM) luôn greedy**, vì HF generate không có presence_penalty nên hai đường không chạy cùng tham số sampling được.
5. **G1-harness** (bắt lỗi harness độc lập với việc card thiếu protocol): chạy 1 model nhỏ có số trên **OpenVLM Leaderboard** (số do chính VLMEvalKit sinh với config công khai), ngưỡng ±1.5pt, chạy bằng VLMEvalKit cùng commit.
   - Ứng viên: **Qwen/Qwen3-VL-2B-Instruct** (Apache-2.0, ~4.3GB, transformers native, AutoModel 26.02 hỗ trợ sẵn).
   - ❓ Cần xác nhận model này có trên leaderboard; nếu không thì dùng Qwen2.5-VL-7B-Instruct (Apache-2.0, ~16GB).
   - Tránh Qwen2.5-VL-3B (license qwen-research, non-commercial).
   - Tải cần PM duyệt.
6. DocVQA/ChartQA: baseline tự đo (greedy, thinking off), không có G1.

### A4. Quy tắc diễn giải (cho UI và báo cáo)
- **License dữ liệu eval** (security, 2026-10-12): ChartQA là GPL-3.0 (repo vis-nlp/ChartQA). Hệ quả: `eval_only=true` cho mọi split; app không đóng gói data mà end-user tự tải từ nguồn; không publish file prediction có chứa câu hỏi/ảnh, chỉ publish điểm tổng hợp; không train trên ChartQA. Áp dụng cùng nguyên tắc eval_only cho mọi benchmark trong A1.
- So sánh chỉ hợp lệ khi trùng: harness+commit, task, split, subset ids, thinking, max_new_tokens, pixels, judge. UI cảnh báo khi khác `prompt_hash`/`config_hash` (khớp contract của frontend).
- Fine-tune: báo **Δ trên domain eval** cạnh **Δ trên general benchmark** (forgetting). General tụt quá CI ở ≥2 benchmark ⇒ cảnh báo "model quên kiến thức chung".
- Quant: theo B3 (FP8 TB ≤1pt; NVFP4 ≤2–3pt, OCR ≤3pt, không benchmark nào >5pt). Qwen3.5 NVFP4 là thử nghiệm (P001) ⇒ gắn nhãn.
- Không hiện điểm gộp; nhóm năng lực hiện badge ↑/≈/↓ theo CI95 paired (đã thống nhất với design).
- Contamination: dataset train phải qua kiểm trùng ảnh (perceptual hash) + câu hỏi (n-gram) với toàn bộ split eval ở A1 (G5).

### A5. Ảnh hưởng của RESET (research)
- Bộ benchmark trở thành **registry** (`configs/eval/benchmarks.yaml`) mà node Benchmark đọc: user chọn nhóm năng lực hoặc từng benchmark, còn tầng (Nhanh/Tiêu chuẩn/Đầy đủ) map vào A1.
- Model ngoài catalog ⇒ không có số công bố ⇒ G1 chỉ chạy G1-internal, nhãn "chưa kiểm chứng".

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
Raw predictions (`samples.jsonl`) chỉ lưu local trong `outputs/<run_id>/eval/`. Với benchmark `eval_only` có license hạn chế
(vd ChartQA GPL-3, theo A4 và P003), artifact chia sẻ/export **mặc định không kèm prediction, câu hỏi hay ảnh**, chỉ kèm điểm tổng hợp.
Test: export một EvalReport của benchmark eval_only thì không có `samples.jsonl` và không có field `question`/`image`.
**Run dùng cho gate (G1, G3, N-4, P-3, số công bố)** phải chạy từ **git worktree sạch** ở một commit cụ thể (không push)
(`git worktree add .worktrees/<run_id> <sha>`, trong project theo ADR-002 A1), nên `runinfo.reproducibility_gaps(run.json) == []`. Run từ working tree chung
có `git_dirty=true` chỉ được dùng để smoke/debug, kết quả mang nhãn "not reproducible" và không được dùng làm gate.

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

**Trạng thái G1** (sau khi research viết A3): `PASS` | `FAIL` | `INCONCLUSIVE` | `NOT_AVAILABLE` (model ngoài catalog, không có số công bố).
- `INCONCLUSIVE` chỉ được dùng khi **đủ cả 4 điều kiện**, nếu không thì là FAIL. Mục đích là không để FAIL thật bị gắn nhãn "thiếu protocol":
  1. card không công bố protocol (prompt/pixels/max tokens) cho benchmark đó, ghi rõ field nào thiếu;
  2. |Δ| ≤ 5pt. Lệch lớn hơn gần như chắc chắn là lỗi harness/serve/template, không phải chi tiết protocol;
  3. G1-internal PASS: HF in-process vs endpoint vLLM lệch ≤ 1pt trên cùng benchmark;
  4. `unparseable_rate` ≤ 2% và đã spot-check bằng mắt ≥ 30 output raw.
- G1 so với số công bố phải chạy **full split**, không chạy subset. Số công bố tính trên toàn split, nên subset 500 mẫu thêm sai số lấy mẫu
  khoảng ±3pt, làm ngưỡng ±1.5pt mất ý nghĩa. Để giảm chi phí: ưu tiên cột **non-thinking** của card nếu có (rẻ hơn rất nhiều), và chỉ chạy thinking-on cho
  benchmark nhỏ (MMMU 900, OCRBench 1000).
- **G1-harness** (khuyến nghị): khi model chính chỉ đạt INCONCLUSIVE ở mọi benchmark, harness chưa từng được đối chiếu với nguồn ngoài nào.
  Khi đó chạy harness trên 1 model nhỏ mà số công bố do **chính harness đó** sinh ra với config công khai (OpenVLM leaderboard/VLMEvalKit),
  ngưỡng ±1.5pt. Việc này cần tải thêm model, nên cần PM duyệt.
- **G1 với sampling** (card Qwen3.5 dùng T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5 cho non-thinking; research A3):
  - Chạy G1 qua **vLLM** với đúng các tham số đó × **3 seed**, mỗi request có seed riêng (cố định theo sample id + seed),
    rồi so **mean** với số công bố. PASS nếu |mean − published| ≤ 1.5pt.
  - Độ ổn định: spread (max − min) giữa 3 seed ≤ 2pt. Nếu vượt, chạy thêm 2 seed (tổng 5). Vẫn vượt thì G1 không được PASS: ghi
    `INCONCLUSIVE (variance)` và mở bug kiểm tra harness/serve. Báo cả mean, spread và từng seed.
  - Kiểm tái lập: chạy lại cùng seed trên 1 benchmark phải cho output giống hệt (vLLM seed per-request). Nếu không giống thì ghi
    nguồn non-determinism (batching) vào record.
  - **G1-internal (HF vs vLLM ≤1pt) luôn chạy greedy**, không dùng sampling. Lý do: HF `generate` không có `presence_penalty`
    (`repetition_penalty` là cơ chế khác), nên 2 đường không thể dùng cùng tham số sampling, và so 2 phân phối ngẫu nhiên thì vô nghĩa.
  - Regression/quant nội bộ: greedy như cũ.
  - Benchmark phải đúng **phiên bản** card dùng (MMBench-EN-dev **v1.1**); task name lmms-eval xác minh khi cài.
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

---
## Ảnh hưởng của RESET (2026-10-12, bao-tester)
Tầm nhìn mới: app no-code node-graph cho user không biết code. Nội dung P002 vẫn giữ nguyên giá trị. Thay đổi lớn nhất:
**user không tự đánh giá được một con số, nên các gate phải do app thực thi và hiển thị, không trông vào người đọc.**

### R.1 P002 → node & UI
| P002 | Trong app |
|---|---|
| G0 env smoke | Chạy tự động khi app khởi động / đổi container; fail ⇒ banner "Môi trường chưa sẵn sàng" + nút "Xem chi tiết" |
| G1 reproduce base | Thuộc tính của template/base model: app ship sẵn kết quả G1 đã kiểm (bảng sanity). Model ngoài danh sách ⇒ nhãn "chưa kiểm chứng" |
| G2/G3 regression/quant | Node **Benchmark**: mode `Nhanh (smoke)` / `Tiêu chuẩn (regression)` / `Đầy đủ`, hiện ước lượng thời gian. Kết quả là thẻ PASS/CẢNH BÁO/FAIL kèm câu giải thích, Δ ± CI |
| G4 perf | Node **Benchmark tốc độ**: điều kiện đo cố định theo preset, không cho user chỉnh tự do các thông số làm sai so sánh |
| G5 leakage | Chạy tự động khi nối node Dataset → Train nếu đồ thị có node Benchmark; trùng ⇒ cảnh báo trước khi chạy |
| R6 (P007) | Chạy tự động sau mỗi node biến đổi weights (convert/merge/quantize export); fail ⇒ chặn node phía sau |

### R.2 Test strategy bổ sung
1. **Graph executor** (unit/integration, CPU, worker giả lập):
   - Thứ tự topo đúng; phát hiện chu trình; node thiếu input bắt buộc ⇒ lỗi khi validate, không phải lúc chạy.
   - Kiểu kết nối: nối sai kiểu bị chặn cả lúc kéo dây (UI) và lúc load file workflow (backend) ⇒ một bộ test kiểu dùng chung 2 phía.
   - Cache: cùng (node type, version, config hash, hash input) thì tái dùng; đổi node phía trên ⇒ invalidate đúng các node phía dưới, không hơn không kém.
   - Resume: dừng giữa chừng (kill, memguard exit 75, mất điện giả lập) ⇒ chạy lại tiếp từ node/checkpoint cuối, không chạy lại node đã xong.
   - Cancel, lỗi một nhánh không phá nhánh độc lập; hàng đợi tôn trọng GPU lock (chỉ 1 node nặng tại 1 thời điểm).
   - File workflow: save → load round-trip giống hệt; file schema cũ phải load được (migration test với fixture từng version); file hỏng/lạ ⇒ lỗi dễ hiểu.
   - Property-based (hypothesis): sinh đồ thị ngẫu nhiên hợp lệ/không hợp lệ ⇒ validator không bao giờ crash, executor không bao giờ chạy đồ thị invalid.
2. **Template**: mỗi template ship kèm
   - (a) test validate (schema, tham số trong vùng an toàn, ước lượng RAM/disk/thời gian ≤ ngân sách DGX Spark);
   - (b) smoke E2E CPU với tiny model / worker giả lập, chạy trong gate thường;
   - (c) smoke E2E GPU với model nhỏ thật, chạy nightly có GPU lock (sau khi PM mở lại GPU).
   Template mới không có (a)(b)(c) ⇒ không được merge.
3. **Pre-run estimator**: so ước lượng RAM/thời gian/disk với số đo thật (từ tester-perf-bench). Ngưỡng đề xuất: RAM ước lượng ≥ thực đo (không được đánh giá thấp) và ≤ +30%. Thời gian ±50%. Ước lượng thấp hơn thực tế là lỗi nghiêm trọng vì có thể làm treo máy chung.
4. **Thông báo lỗi cho user no-tech**:
   - Error catalog: mỗi lớp lỗi đã biết (thiếu RAM, OOM/UMA, hết disk, GPU lock đang bận, thiếu token HF/model gated, license chưa chấp nhận, mạng, nối sai kiểu, dataset sai format, NVFP4 không hỗ trợ…) ⇒ mã lỗi + **chuyện gì xảy ra / vì sao / làm gì tiếp** bằng tiếng người, chi tiết kỹ thuật thu gọn.
   - Test fault-injection: worker giả ném từng lớp lỗi ⇒ assert message hiển thị không chứa traceback/đường dẫn nội bộ/secret, có mã lỗi, có hành động gợi ý. Exception chưa biết ⇒ thông báo chung + mã tham chiếu tới log, và test đếm số lỗi "chưa phân loại" như một chỉ số chất lượng.
5. **UI E2E** (Playwright, kiểm tra hỗ trợ Chromium linux-arm64): user journey "mở template → chạy → xem kết quả → export", nối sai kiểu bị chặn, lỗi hiển thị đúng. Kèm checklist usability thủ công với persona no-tech, phối hợp bao-design.

### R.3 Giai đoạn hiện tại
Không chạy job GPU. Việc tester làm: spec test (xfail strict) cho scoring/stats (đã có), cho validator/executor (khi P009 có interface), và error catalog (cùng bao-design + bao-fixcode).
