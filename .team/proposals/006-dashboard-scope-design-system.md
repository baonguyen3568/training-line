# Proposal 006 — Dashboard scope + design system

> ## Ảnh hưởng của RESET (2026-10-12)
> - UI chính là **canvas node-graph** (xem P009 + `009-annex-frontend-canvas.md`); dashboard này trở thành **các panel trong app** (run monitor, scorecard, Pareto, error analysis, demo).
> - **Phương án A (Gradio thuần) bị loại** (không làm được canvas). B (FastAPI + React SPA, 127.0.0.1:7870) giữ nguyên và là nền cho editor.
> - Giữ nguyên: data contract file-based, lớp đọc dữ liệu tách khỏi UI, bảo mật §8, design system/tokens (phụ lục 006).
> - Thay đổi: §1 "read-only v1" bỏ — app sẽ tạo/chạy workflow qua API backend (P009); metadata benchmark là registry dữ liệu (bao-research), không hard-code.

Tác giả: bao-frontend (stack, kiến trúc, data contract) + bao-design (UX flow, design system, wireframe)
Trạng thái: DRAFT · 2026-10-12 · Reviewer: bao-pm, bao-backend, bao-security, bao-tester

## 1. Mục tiêu & phạm vi v1
Người dùng nội bộ (team + user) trả lời nhanh 4 câu hỏi:
1. Job đang chạy có khoẻ không? (loss/lr/grad-norm/throughput, memory pressure máy dùng chung, GPU lock)
2. Model nào tốt hơn? (base vs fine-tuned vs quantized, accuracy + perf, có CI và sanity gate)
3. Quantize có đáng không? (Pareto accuracy vs latency/memory)
4. Model trả lời ảnh/video ra sao? (demo inference side-by-side)

Ngoài phạm vi v1: tạo/launch job từ UI (UI read-only, job chạy bằng CLI), multi-user auth, truy cập ngoài localhost,
export/lineage card (v1.1).

## 2. Stack — phương án & khuyến nghị (bao-frontend)
| | A. Gradio 6 thuần | B. FastAPI + Vite/React SPA tĩnh + Gradio mount cho demo | C. Next.js + FastAPI |
|---|---|---|---|
| Thời gian ra v1 | nhanh nhất (~3-4 ngày) | trung bình (~6-8 ngày) | chậm nhất |
| Bảng so sánh / Pareto / pin baseline / 4-state panel | khó, layout hạn chế | kiểm soát hoàn toàn | kiểm soát hoàn toàn |
| Demo ảnh/video VLM | rất tốt | rất tốt (Gradio mount) | phải tự viết |
| Runtime | 1 process Python | **1 process Python, 1 port**; Node chỉ lúc build | 2 runtime (Node + Python) |
| Chart | gr.LinePlot/Plotly | uPlot (curve dài) + ECharts (benchmark) | như B |
| Maintain bởi team (đa số Python) | dễ | trung bình (frontend do bao-frontend giữ) | khó |
| Design system tokens | theme Gradio hạn chế | CSS variables từ DTCG tokens | như B |

**Khuyến nghị: B** (bao-design và bao-frontend đồng thuận). Lý do: phần ra quyết định (scorecard, Pareto, diff
điều kiện đo) cần kiểm soát layout/màu mà Gradio làm khó; demo upload ảnh/video thì Gradio thắng → mount
`gr.mount_gradio_app` dưới `/demo`. Next.js SSR thừa cho tool localhost và thêm runtime Node lúc chạy.
Rủi ro B: 2 ngôn ngữ (TS + Python); giảm thiểu bằng build tĩnh commit-free (CI build), contract pydantic → JSON Schema →
TS types sinh tự động, nên đổi schema không lệch.
Fallback nếu cần ra cực nhanh: A cho tuần đầu, chuyển sang B không mất gì vì contract file-based độc lập UI.

## 3. Kiến trúc
```
pipeline (train/quantize/eval) ──ghi──▶ outputs/runs/<run_id>/{run.json,metrics.jsonl,system.jsonl,events.jsonl}
                                        benchmarks/**.json      outputs/system/disk_report.json   .team/GPU_LOCK
                                                   │ read-only
                          vlm-dashboard (FastAPI, 127.0.0.1:7870)
                          ├── /api/runs, /api/runs/{id}/metrics?downsample=…  (tail theo byte offset, LTTB)
                          ├── /api/benchmarks (pivot, Δ vs base, cảnh báo khác điều kiện)
                          ├── /api/system (MemAvailable, GPU util/clock/power/temp, GPU_LOCK, disk)
                          ├── /api/stream (SSE, 2-5s)
                          ├── /            (SPA tĩnh: React + uPlot + ECharts)
                          └── /demo        (Gradio 6) ──HTTP──▶ vLLM :7871 / TRT-LLM :7872 (OpenAI-compatible)
```
- Lớp đọc dữ liệu (parse/tail/downsample/pivot) là module Python thuần `src/vlm_pipeline/common/` + `dashboard/reader`, không phụ thuộc UI; FastAPI và Gradio chỉ gọi module này → đổi A↔B không phải viết lại (yêu cầu của PM).
- Mặc định trong lúc chờ user chốt: chỉ 127.0.0.1 + SSH tunnel; không dùng Trackio/TensorBoard ở v1.
- Dashboard **không load model, không import torch, không lấy GPU lock**; mục tiêu RSS < 500MB.
- Demo: client của serve endpoint; nếu GPU_LOCK do job khác giữ → banner cảnh báo (perf bị nhiễu).
- Bind 127.0.0.1, port 7870, không `share=True`, truy cập SSH tunnel. Container `vlm-dashboard` (arm64) tuỳ chọn.

## 4. Data contract (đồng thuận bao-backend, bao-research, bao-tester, bao-cleanup)
Nguồn chi tiết: `.claude/skills/frontend-metrics-contract/SKILL.md` (sẽ chuyển thành pydantic trong
`src/vlm_pipeline/common/` do bao-backend viết, frontend sinh TS types từ JSON Schema).
- Layout `outputs/runs/<run_id>/`; 4 file contract + logs/ được whitelist khỏi retention (Proposal 005).
- `run.json`: run_id, task, stage, parent_run_id (lineage), base_model, dataset, config_hash, git_commit, seed,
  container_image/digest, status, timestamps.
- `metrics.jsonl` (wide, phẳng, đơn vị trong tên key), `system.jsonl` (null khi GB10 không hỗ trợ), `events.jsonl`.
- Benchmark record phẳng: value thang 0-100 + scale, n_samples, ci95_low/high, sanity_gate, source measured|published,
  harness(+version), prompt_hash, thinking, judge_model, max_new_tokens; perf = record riêng (benchmark="perf")
  với p50/p90/p99 TTFT/TPOT/E2E, tok/s, req/s, memory startup/steady/peak + `perf_conditions` (gồm kernel_backend).
- Tái lập: seed, dataset_revision, container_digest, chat_template_hash, image_cfg_hash, eval_path, samples_path;
  thiếu trường tái lập → badge "not reproducible". So sánh paired dùng `delta` + `delta_ci95_*` do eval tính (UI không suy
  luận từ chồng error bar).
- `outputs/system/disk_report.json` (bao-cleanup, P005 D6) cho widget hệ thống; `checkpoint_pruned` event trong events.jsonl.
- `schema_version` mọi nơi; đổi không tương thích → bump, UI hỗ trợ N và N-1.

## 5. Màn hình v1 (bao-design) — chi tiết + wireframe: `006-annex-ux-design-system.md` §A2–A4
| # | Màn hình | Câu hỏi chính | Ưu tiên |
|---|---|---|---|
| a | Header toàn cục | Máy có rảnh/an toàn? (GPU_LOCK FREE/BUSY/STALE, RAM stacked + MemAvailable warn<20/crit<10 GB, disk) | P0 |
| b | Runs list | Có run nào, baseline/champion? (pin, chọn ≤8, Bảng⇄Biểu đồ giữ selection, lineage qua parent_run_id) | P0 |
| c | Run monitor | Run khoẻ không? (loss/lr/grad_norm tách chart, raw+smoothed, NaN marker, job mem, GPU metrics `unverified`) | P0 |
| d | Scorecard | Model nào tốt hơn? (benchmark × variant, Δ + delta_ci95 paired, CI chứa 0 → xám, sanity FAIL banner, perf chỉ so khi cùng điều kiện) | P0 |
| e | Pareto quantize | Precision nào đáng ship? (1 trục y, x chọn TTFT/tok/s/peak mem, quality floor, direct label) | P0 |
| f | Error analysis | Sai ở mẫu nào? (ảnh + Q + GT + pred, ma trận đúng/sai 2 variant; dùng `samples_path`) | P1 |
| g | Demo `/demo` | Trả lời ảnh/video ra sao? (side-by-side 2 endpoint, banner khi GPU bận) | P1 |
| h/i | Lineage card · Storage/retention dry-run | Artifact ở đâu / cái gì sắp bị xoá? | v1.1 |
Mọi panel có 4 state: loading (skeleton) · empty (kèm lệnh CLI) · error (path tương đối + schema_version) · stale.
Badge dùng chung: sanity FAIL · not reproducible · ≠ điều kiện · unverified · license 3 mức · remote code · eval-only data · unpinned
(chi tiết A4). UI read-only: hành động = lệnh CLI copy được.

## 6. Design system (bao-design) — chi tiết: annex §A5, tokens: `design/tokens/*.tokens.json` (DTCG 2025.10)
- Hue = dòng model (base blue / ft orange / ft2 aqua), precision = bước đậm dần ra xa nền + marker
  (BF16 ● → FP8 ▲ → NVFP4 ◆ → INT4-AWQ ■), published = viền rỗng + chữ nghiêng. Token `color.variant.<lineage>.<precision>`.
- Đã chạy validator CVD/contrast: 3 hue BF16 PASS all-pairs 2 theme; mỗi ramp precision PASS ordinal; phương án 5 hue riêng FAIL
  scatter → loại. Hệ quả: scatter bắt buộc shape + direct label, tối đa 2 dòng model.
- Δ có ý nghĩa dùng `color.delta.better|worse` + ▲/▼ (không dùng blue — trùng danh tính base). Status 4 mức luôn kèm icon + chữ.
- NVIDIA green chỉ cho chrome. System font, mono cho id/hash, tabular-nums trong bảng. Không radar, không dual-axis.

## 7. Đặc thù GB10 / máy dùng chung
- Không có "GPU memory %" (unified memory, nvidia-smi "Not Supported") → hiển thị system memory pressure
  (MemTotal−MemAvailable, chia other services | vlm-* | cache | free) + memory của job (torch max_allocated, RSS).
- Hiển thị GPU clock/power/temp để phát hiện throttle (có báo cáo clock ghim 721 MHz).
- Port 7860/6006/8080/11000 đã bị service khác chiếm → dải dự án 7870-7879.

## 8. Bảo mật (góp ý sơ bộ của bao-security đã đưa vào; chờ review chính thức)
- Bind 127.0.0.1 port 7870, SSH tunnel. **Auth bắt buộc kể cả localhost** (sửa theo bao-security, P003 D13) — chi tiết ở 009 annex F7. Kiểm `ss -tln` trước khi chiếm port.
- API key vLLM/TRT-LLM chỉ ở server (FastAPI/Gradio backend), browser không giữ key; không secret phía client.
- Header: CSP `default-src 'self'; frame-ancestors 'none'` (không inline script nếu được), `X-Content-Type-Options: nosniff`;
  bundle hết, không CDN lúc runtime. CSP tách 2 tầng (bao-security đồng ý): SPA chặt; `/demo` nới tối thiểu — 'unsafe-inline' chỉ khi đo được là cần, tránh 'unsafe-eval',
  connect-src 'self' + ws(s) cùng origin, không CDN (tắt font/JS CDN mặc định của Gradio), `analytics_enabled=False`, `share=False`,
  `max_file_size`. Gửi header đo thực tế cho bao-security trước khi merge.
- Text do model sinh / caption dataset render plain text (hoặc markdown tắt HTML thô), không innerHTML.
- Upload: whitelist MIME png/jpg/webp/mp4, giới hạn dung lượng (ảnh ≤ 20MB, video ≤ 200MB/60s) và MAX_IMAGE_PIXELS;
  không cho nhập URL để server tự tải (chống SSRF).
- Chỉ đọc `outputs/` và `benchmarks/` (+ `.team/GPU_LOCK`), chặn path traversal (realpath ∈ root), không đọc `.env`/config có key.
- npm: lockfile commit, `npm ci`, pin exact, `npm audit --omit=dev`, `.npmrc ignore-scripts=true` nếu được; deps có prebuilt linux-arm64.

## 9. Test (bao-tester)
Fixture `tests/fixtures/runs/` + `benchmarks` mẫu; pytest cho parser/pivot/downsample; smoke test khởi động không
GPU/serve; e2e UI (Playwright, arm64 Chromium) do bao-tester hỗ trợ.

## 10. Kế hoạch (sau khi ADR chốt)
1. Contract pydantic + fixtures (backend + frontend) · 2. FastAPI API + tests · 3. SPA: header + runs list + run monitor ·
4. Scorecard + Pareto · 5. Demo Gradio mount · 6. Review chéo (design, security, tester).

## 11. Câu hỏi mở cho PM/user
- Chấp nhận B (thêm toolchain Node lúc build) hay ưu tiên A để có UI sớm hơn ~3-4 ngày?
- Có cần truy cập từ máy khác ngoài SSH tunnel không?
- Có dùng Trackio/TensorBoard song song cho curve train không (đề xuất: không ở v1 — tránh 2 nguồn sự thật)?

## Nguồn
Xem `.team/knowledge/frontend.md` (mục Nguồn).
