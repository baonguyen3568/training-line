# Frontend knowledge — VLM Training Platform (bao-frontend)
Cập nhật: 2026-10-12. Phạm vi: dashboard theo dõi train/eval/benchmark, so sánh model, demo inference ảnh/video
trên 1x DGX Spark GB10 (aarch64, 128GB unified memory, máy dùng chung).

## 1. Sự thật về môi trường (đã kiểm tra trên máy, 2026-10-12)
- Node v24.15.0 + npm có sẵn; uv có sẵn; Python 3.12 qua uv.
- RAM: total 121G, used ~78G, available ~42G → dashboard phải **nhẹ** (mục tiêu < 500MB RSS, 0 VRAM).
- Port đang bị chiếm: 22, 53, 631, 3350, 3389, 4317, **6006** (phoenix – trùng TensorBoard mặc định),
  7070, **7860** (nemotron-voice-agent – trùng Gradio mặc định), **8080** (open-webui), **11000** (DGX Dashboard),
  11434 (ollama), 18002/18003, 50051…
  → **Đề xuất dải port dự án: 7870–7879** (7870 dashboard, 7871 vLLM, 7872 TRT-LLM, 7873 dashboard API nếu tách, 7874 Trackio/TensorBoard — đã thống nhất với bao-backend), bind `127.0.0.1`, truy cập qua SSH tunnel.
  Mọi port ngoài localhost phải hỏi bao-security trước (PM yêu cầu).

## 2. Đặc thù GB10 ảnh hưởng tới UI giám sát
- `nvidia-smi` báo memory **"Not Supported"** vì GB10 là iGPU dùng chung RAM; per-process GPU memory vẫn liệt kê được.
  NVIDIA khuyên dùng `free`/`top`/DGX Dashboard (:11000) cho bộ nhớ tổng. [NVIDIA Known Issues], [GB10 FAQ]
- Tool gọi memory query trực tiếp có thể vỡ (vd k8s-device-plugin issue #1482) → code thu thập metric phải chịu được
  giá trị `N/A`, không crash, hiển thị "n/a" thay vì 0.
- DCGM-exporter: **chưa thấy xác nhận hỗ trợ GB10** → không dựa vào Prometheus/DCGM ở v1.
- Có báo cáo cộng đồng GPU clock bị ghim 721 MHz → nên hiển thị cả clock/power/temp để phát hiện throttle khi throughput tụt.
- Hệ quả thiết kế panel "Memory":
  - Hiển thị **system memory pressure** (`/proc/meminfo` MemTotal/MemAvailable) — vì OOM trên Spark là OOM của cả máy, kể cả service người khác.
  - Hiển thị **memory của job** = `torch.cuda.max_memory_allocated/reserved` do trainer log + RSS process.
  - Không hiển thị "GPU memory %" kiểu GPU rời — gây hiểu sai.
- Cảnh báo sớm: MemAvailable < ~10GB → banner đỏ (ngưỡng chốt với backend/cleanup).

## 3. Lựa chọn stack (để đưa vào Proposal 006)
| Phương án | Ưu | Nhược |
|---|---|---|
| **Gradio 6** (Svelte 5, Navbar multipage, MultimodalTextbox, video+subtitle, custom HTML/JS inline) | Demo ảnh/video VLM gần như có sẵn; Python thuần, backend team tự sửa được; nhẹ | Layout/dashboard phức tạp kém linh hoạt; breaking changes giữa major; chart hạn chế |
| Streamlit | Nhanh cho bảng/biểu đồ | Rerun cả script mỗi tương tác, live update kém, demo video yếu hơn Gradio |
| **Next.js/React + FastAPI** | Đẹp, đúng design system, chart hiệu năng cao (uPlot/ECharts), kiểm soát hoàn toàn | Chi phí build/maintain cao, thêm toolchain Node, 2 codebase |
| Trackio / MLflow / TensorBoard (dùng sẵn cho curve train) | Không phải tự viết loss curve; Trackio local-first SQLite, dashboard là Gradio app, drop-in W&B, có `report_to="trackio"` | Không giải quyết benchmark compare/demo; thêm dependency; MLflow image arm64 chính chủ chưa rõ |
- Chính kiến hiện tại: **contract dữ liệu file-based là cốt lõi, UI là lớp mỏng thay được**. v1 nên ra nhanh
  (Gradio 6 multipage hoặc FastAPI + 1 SPA tĩnh nhỏ), không dựng hệ thống tracking riêng nếu Trackio/TensorBoard đủ.
  Quyết định cuối: bao-pm + user, sau khi bao-design góp ý.
- NeMo AutoModel có logger W&B / MLflow / Comet (API ref "Loggers") → cần bao-backend xác nhận logger nào bật,
  và bổ sung 1 sink JSONL cục bộ làm nguồn sự thật cho dashboard (không phụ thuộc cloud).

## 4. Chart & hiệu năng
- Curve train dài (10^5+ điểm): uPlot nhanh nhất cho line/time-series (~45KB; benchmark 2026: 100k điểm ~9ms),
  nhưng ít tính năng. ECharts 6 đầy đủ (bar, heatmap, tooltip) — hợp benchmark compare. [ApexCharts benchmark 2026], [uPlot]
- **Downsample phía server** (LTTB / min-max theo bucket) trước khi gửi trình duyệt; giữ raw trong file.
- Live update: polling 2–5s hoặc SSE; tail JSONL theo offset byte, không đọc lại cả file.
- Log scale cho loss, EMA smoothing có toggle (hiện cả raw mờ), trục x chọn step/time/tokens.

## 5. Demo inference VLM
- Gọi qua endpoint **OpenAI-compatible** của vLLM/TRT-LLM/NIM: `messages[].content` gồm `{"type":"image_url"}`,
  `{"type":"video_url"}` (vLLM). Cần chat template; giới hạn `--limit-mm-per-prompt` (cú pháp đổi theo version);
  timeout fetch ảnh mặc định 5s (`VLLM_IMAGE_FETCH_TIMEOUT`). [vLLM multimodal inputs]
- Dashboard **không tự load model** (không chiếm GPU/RAM); chỉ là client. Nếu serve chưa chạy → hiển thị trạng thái, không tự khởi động.
- Tôn trọng GPU lock: đọc `.team/GPU_LOCK` và hiển thị "GPU đang bận: ai/job gì/từ lúc nào" — demo phải cảnh báo
  khi đang có job train/benchmark (inference đồng thời làm sai số perf benchmark).
- Upload: giới hạn kích thước (vd ảnh ≤ 20MB, video ≤ 200MB/≤ 60s), gửi base64 hoặc file local, không public URL.
- So sánh side-by-side 2–3 model (base / finetune / quantized) cùng prompt + seed + temperature=0; hiển thị TTFT, tok/s.
- Output model là **untrusted** → render markdown đã sanitize (không HTML thô) để tránh XSS.

## 6. Benchmark visualization
- Bảng chính: hàng = benchmark, cột = variant (base / SFT / LoRA / FP8 / NVFP4 / INT4-AWQ), ô = score + **delta vs base**
  (màu theo dấu, không chỉ theo độ lớn) + n mẫu; có cột "published" (model card) để đối chiếu.
- Luôn hiển thị metadata tái lập: harness + version, prompt/config hash, ngày, commit, n samples, stderr/CI nếu có.
- Perf: TTFT p50/p95, tokens/s, peak memory, batch size, input resolution/số frame — phải ghi rõ điều kiện đo.
- Trade-off chart: accuracy vs throughput (scatter, mỗi điểm một variant) — câu hỏi chính khi chọn quantized.
- Không dùng radar chart để so sánh chính xác (khó đọc delta nhỏ); dùng dot/bar có baseline.
- Số "measured" và "published" phải phân biệt rõ trực quan, không trộn.

## 7. Pitfalls tổng hợp
1. Port mặc định (7860, 6006, 8080) đều đã bị chiếm trên máy này.
2. Hiển thị GPU memory như GPU rời trên GB10 → sai.
3. Dashboard load model để demo → tranh RAM với job train → OOM cả máy.
4. Đọc lại toàn bộ metrics file mỗi lần poll → CPU cao khi run dài.
5. Schema metric không version → dashboard vỡ khi backend đổi key. Cần `schema_version`.
6. npm deps native: kiểm tra có prebuilt linux-arm64 (esbuild/swc/sharp đều có); tránh package chỉ có x64.
7. Gradio/Streamlit major upgrade có breaking change → pin version.
8. Bind 0.0.0.0 vô tình → lộ ra mạng; Gradio `share=True` tạo tunnel public → **cấm**.
9. Trộn số benchmark đo với harness/prompt khác nhau trong cùng bảng mà không đánh dấu.

## 8. Checklist áp dụng cho dự án
- [ ] Contract `outputs/runs/<run_id>/{run.json, metrics.jsonl, system.jsonl, events.jsonl}` + `benchmarks/*.json` có `schema_version` (chốt với bao-backend, bao-research).
- [ ] Pydantic model chung trong `src/vlm_pipeline/common/` để backend ghi & dashboard đọc cùng 1 định nghĩa.
- [ ] Bind 127.0.0.1, port 7870–7879, không `share=True`; bao-security review.
- [ ] Dashboard RSS < 500MB, 0 GPU; không import torch.
- [ ] Panel GPU lock + system memory pressure luôn hiển thị ở header.
- [ ] Downsample server-side, tail incremental.
- [ ] Bảng benchmark có delta vs base, metadata tái lập, phân biệt measured/published.
- [ ] Demo inference qua OpenAI-compatible API, sanitize output, giới hạn upload.
- [ ] Pin version UI framework trong pyproject/package-lock; image arm64.
- [ ] Test: unit test parser/schema (pytest), smoke test UI khởi động + đọc fixture run mẫu.

## Nguồn
- NVIDIA DGX Spark Known Issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- DGX Spark / GB10 FAQ: https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344
- nvidia-smi on DGX Spark thread: https://forums.developer.nvidia.com/t/dear-nvidia-nvidia-smi-is-broken-on-the-dgx-spark/367765
- GB10 clock 721 MHz thread: https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039
- k8s-device-plugin GB10 issue: https://github.com/NVIDIA/k8s-device-plugin/issues/1482
- DGX Dashboard: https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html , https://build.nvidia.com/spark/dgx-dashboard/instructions
- DCGM Exporter docs: https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html
- Gradio 6 release coverage: https://alternativeto.net/news/2025/11/gradio-6-released-with-faster-performance-for-creating-machine-learning-apps-in-python , 6.21.0 notes: https://newreleases.io/project/github/gradio-app/gradio/release/gradio@6.21.0
- Trackio docs: https://huggingface.co/docs/trackio/en/index , https://www.infoq.com/news/2025/09/hugging-face-trackio
- vLLM multimodal inputs: https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html , https://docs.vllm.ai/en/v0.12.0/examples/online_serving/openai_chat_completion_client_for_multimodal/
- Chart benchmark 2026: https://apexcharts.com/blog/javascript-chart-library-benchmark/ ; uPlot: https://npmjs.com/package/uplot
- NeMo AutoModel API (loggers): https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components
- MLflow self-hosting: https://mlflow.org/docs/latest/self-hosting/
- Lưu ý: một số thông tin (Gradio 6.x chi tiết, DCGM trên GB10) là nguồn thứ cấp/chưa xác minh — sẽ verify khi chốt stack.
