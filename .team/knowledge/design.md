# Knowledge — Product / UI-UX Design (bao-design)

> Cập nhật: 2026-10-12. Phạm vi: UX dashboard train/eval/quantize VLM, design system, trực quan hoá benchmark,
> sơ đồ kiến trúc docs. Bối cảnh: 1x DGX Spark GB10 (aarch64, 128GB unified memory, máy dùng chung).
> Nguồn ghi ở cuối mỗi mục. Chỗ nào là **ý kiến của tôi** (không có nguồn) sẽ ghi rõ "(quan điểm)".

---

## 1. Người dùng & luồng chính (Jobs-to-be-done)

Người dùng thật của dashboard này: **chính team ML (1–5 người) + user/PM**, không phải khách hàng ngoài.
→ Ưu tiên *mật độ thông tin + tốc độ ra quyết định*, không ưu tiên marketing-polish. (quan điểm)

Luồng end-to-end cần dashboard phục vụ (theo pipeline trong CLAUDE.md):

```
[1 Tạo run] → [2 Theo dõi run] → [3 So sánh runs] → [4 Chọn model "champion"]
   → [5 Quantize (FP8/NVFP4/INT4-AWQ)] → [6 Benchmark acc + perf] → [7 Export/Serve]
```

| Bước | Câu hỏi người dùng | Màn hình / thành phần cần có |
|---|---|---|
| 1 | "Chạy recipe nào, với config gì, GPU có rảnh không?" | Form/CLI preview từ YAML, **trạng thái GPU_LOCK** nổi bật, ước lượng RAM |
| 2 | "Run đang khoẻ không? Loss có giảm? Có sắp OOM không?" | Live line charts (loss, lr, grad-norm, tokens/s), **unified memory gauge**, log tail, ETA |
| 3 | "Run nào tốt hơn và vì sao?" | Bảng runs (sort/filter) ⇄ chart view, **Diff-only** config, pin baseline |
| 4 | "Model nào đáng đem đi quantize/serve?" | Tag `champion`, scorecard benchmark vs base |
| 5–6 | "Quantize mất bao nhiêu accuracy, được bao nhiêu tốc độ/RAM?" | **Pareto chart acc vs latency**, bảng delta vs BF16 |
| 7 | "Artifact nằm đâu, chạy thế nào?" | Lineage card: data → ckpt → quantized → engine, lệnh copy-paste |

**Lineage là xương sống** (quan điểm): mọi artifact (dataset, checkpoint, quantized, engine, benchmark json) có ID và
link ngược nguồn gốc → UI chỉ là view trên lineage graph. Cần thống nhất schema với bao-backend sớm.

## 2. Pattern UX đã được chứng minh ở tool tracking (MLflow, W&B, ZenML)

- **Table ⇄ Chart toggle không mất selection** (MLflow 2.x experiments UI): chọn runs 1 lần, chuyển view tự do.
  Chart view có bar / line / scatter / **parallel coordinates** (brush để lọc theo khoảng param/metric).
- **Pin baseline run**: run được pin luôn ở đầu bảng kể cả khi đổi filter → so sánh với base model luôn sẵn.
- **Run Comparer kiểu W&B**: cột = run, hàng = config/metric, toggle **"Diff only"** ẩn giá trị giống nhau.
  Giới hạn hiển thị (W&B: 10 run đầu; ZenML: 20 run) — phải có giới hạn rõ ràng, không render vô hạn.
- **Workspace auto vs manual**: auto sinh panel cho mọi key log → nhanh nhưng loạn; manual → sạch. Đề xuất:
  *preset layout theo loại run* (SFT / LoRA / GRPO / quantize / benchmark) + cho phép tuỳ biến. (quan điểm)
- **Tag "champion"** cho run thắng để tìm lại (MLflow guide).
- **Tooltip chỉ khi text bị cắt**, row click được, header cột rõ ràng (GitLab ML experiment UX issue).
- Thanh tìm kiếm có cú pháp metric/param + autosuggest (MLflow).

Nguồn: [Databricks — new MLflow experiments UI](https://www.databricks.com/blog/accelerate-your-model-development-new-mlflow-experiments-ui) ·
[W&B Run comparer](https://docs.wandb.ai/models/app/features/panels/run-comparer) ·
[W&B Workspaces](https://docs.wandb.ai/guides/app/pages/workspaces) ·
[ZenML compare experiments](https://www.zenml.io/blog/new-dashboard-feature-compare-your-experiments) ·
[GitLab issue 388207](https://gitlab.com/gitlab-org/gitlab/-/issues/388207)

**Phản biện sớm (quan điểm):** đừng tự xây lại W&B/MLflow. Nếu backend đã log ra MLflow/TensorBoard/W&B thì
dashboard của ta nên tập trung vào cái các tool đó **không** có: lineage train→quantize→benchmark, Pareto quantization,
GPU_LOCK/host-sharing, demo inference VLM ảnh/video. Cần chốt cùng bao-frontend trong Proposal 006.

## 3. Đặc thù phần cứng DGX Spark GB10 → hệ quả thiết kế

- **`nvidia-smi` KHÔNG báo memory usage trên GB10** ("Memory-Usage: Not Supported") vì unified memory, không có VRAM
  riêng. pynvml trả `NVML_ERROR_NOT_SUPPORTED` cho memory. NVIDIA khuyên dùng `free`/`top`/`htop` hoặc
  **DGX Dashboard tại localhost:11000**. Workaround phổ biến: `/proc/meminfo` (MemTotal − MemAvailable).
- ⇒ **Thiết kế:** KHÔNG làm widget "GPU VRAM x/80GB" kiểu datacenter. Thay bằng **1 thanh unified memory 128GB
  chia lớp (stacked)**: `services khác (kafka/elastic/…)` | `job vlm-* hiện tại` | `page cache (buff/cache)` | `free`.
  Đây là câu hỏi số 1 của user trên máy dùng chung: "chạy job này có làm chết service khác không?".
  Cần ngưỡng cảnh báo (vd available < 16GB → warning, < 8GB → critical) — số cụ thể chờ bao-backend/bao-cleanup xác nhận.
- Page cache có thể "ăn" nhiều RAM và gây hiểu nhầm (memory creep) → hiển thị buff/cache tách riêng, có tooltip giải thích.
- GPU utilization/power/nhiệt: FAQ không nói nguồn đáng tin → **hỏi bao-backend/bao-research** nguồn metric thật
  (DCGM? nvidia-smi --query-gpu=utilization.gpu?), không vẽ widget cho metric chưa xác minh được.
- **GPU_LOCK** (`.team/GPU_LOCK`) là một trạng thái UI hạng nhất: banner "Đang bận: <ai> — <job> — từ <giờ> — PID",
  nút tạo run bị disable kèm lý do. Không bao giờ có nút "kill" job người khác (quy tắc an toàn).
- aarch64: chọn stack frontend phải chạy được trên arm64 (Node/Python đều OK; tránh lib có native binary chỉ x86).

Nguồn: [DGX Spark / GB10 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344) ·
[DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) ·
[Memory creep on DGX Spark](https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886) ·
[NVML unified memory community thread](https://forums.developer.nvidia.com/t/nvml-support-for-dgx-spark-grace-blackwell-unified-memory-community-solution/358869)

## 4. Trực quan hoá benchmark (accuracy + perf)

### 4.1 Chọn dạng chart theo "việc" của dữ liệu
| Câu hỏi | Dạng nên dùng | Tránh |
|---|---|---|
| Model A vs B trên N benchmark (MMMU, DocVQA, ChartQA, OCRBench…) | **Bảng có delta + dot plot / grouped bar**, small multiples theo benchmark | **Radar chart** (mỗi trục 1 thang khác nhau, khó so, diện tích đánh lừa) |
| Quantize đánh đổi gì | **Scatter Pareto**: x = latency (p50, ms) hoặc RAM, y = accuracy; góc tốt = **trên-trái**; nối Pareto front | Dual-axis (2 trục y) |
| Mất bao nhiêu điểm so với BF16 | **Delta bar** diverging quanh 0 (xanh/đỏ + gray midpoint) | Hiển thị số tuyệt đối không có baseline |
| Loss/lr theo step | Line 2px, smoothing có toggle, giữ raw mờ phía sau | Smoothing ẩn raw |
| Throughput / latency 1 con số | **Stat tile** (hero number + delta) | Bar chart 1 cột |

- Pareto chart **phải có "quality floor"** (ngưỡng accuracy tối thiểu): benchmark chỉ nhìn tốc độ có thể chọn cấu hình
  accuracy ≈ 0 (Inference Engineering Pareto Atlas, arXiv 2609.17863). Winner thay đổi theo ràng buộc & GPU →
  **luôn ghi rõ phần cứng (GB10), engine (TRT-LLM/vLLM), batch, seq len, độ phân giải ảnh, số frame video** trên chart.
- Mỗi điểm benchmark phải truy được về: version harness (VLMEvalKit/lmms-eval), commit, prompt template, seed, n mẫu.
  Thiếu metadata → hiển thị badge "không reproducible". (quan điểm, khớp yêu cầu reproducible trong CLAUDE.md)
- Hiển thị **khoảng tin cậy / n mẫu** khi chênh lệch nhỏ (< 1 điểm) — tránh tuyên bố "thắng" vì nhiễu. (quan điểm, cần bao-tester/bao-research chốt protocol trong Proposal 002)
- Accuracy benchmark có thang khác nhau (OCRBench 0–1000, đa số 0–100%) → **chuẩn hoá hiển thị** hoặc tách chart, không chung trục.

Nguồn: [Observable — Avoid radar charts](https://observablehq.com/blog/avoid-radar-charts) ·
[Scott Logic — critique of radar charts](https://blog.scottlogic.com/2011/09/23/a-critique-of-radar-charts) ·
[apxml — visualizing performance tradeoffs](https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/visualizing-performance-tradeoffs) ·
[Inference Engineering Pareto Atlas (arXiv 2609.17863)](https://arxiv.org/pdf/2609.17863) ·
[ParetoQ (arXiv 2502.02631)](https://arxiv.org/pdf/2502.02631) ·
[Quantization study Llama-3.1 (arXiv 2411.02355)](https://arxiv.org/html/2411.02355v1)

## 5. Màu sắc & accessibility (dark/light)

Quy tắc bất di bất dịch:
1. **Màu theo chức năng**: categorical (danh tính run/model) · sequential (độ lớn, 1 hue nhạt→đậm) ·
   diverging (delta ±, 2 hue nóng/lạnh + gray giữa) · status (good/warning/serious/critical, **luôn kèm icon + chữ**).
2. **Màu đi theo entity, không theo thứ hạng** — lọc bớt run không được đổi màu các run còn lại. → gán màu theo
   `run_id` hash ổn định / thứ tự tạo, lưu cố định.
3. Tối đa 8 màu categorical, **thứ tự cố định**; cần >8 → gộp "Other" / small multiples. Scatter/Pareto (mọi cặp đều
   cạnh nhau) chỉ dùng ≤3 màu an toàn CVD; còn lại dùng shape/label.
4. Không dùng màu một mình để mã hoá: kèm legend (≥2 series), direct label chọn lọc, shape, texture khi in.
5. Contrast WCAG: text 4.5:1, phần tử phi văn bản 3:1. **Dark mode là bộ step riêng** được validate trên nền tối,
   không đảo màu tự động (Pluma, Kumo, Calgary đều có bảng hex riêng cho dark).
6. **Validate bằng script, không bằng mắt** (ΔE CVD ≥ 8 giữa màu kề nhau). Okabe-Ito là chuẩn tham khảo an toàn
   màu cho người mù màu, nhưng màu đen/xanh đậm của nó mất contrast trên nền tối.
7. Không dual-axis; gridline mảnh liền nét; số liệu trong bảng dùng `tabular-nums`.

Palette khởi điểm đề xuất cho dự án (bản tham chiếu đã validate từ skill dataviz, sẽ đưa vào `design/tokens`):
| Slot | Light | Dark | Gợi ý gán |
|---|---|---|---|
| 1 blue | `#2a78d6` | `#3987e5` | Base model / baseline |
| 2 orange | `#eb6834` | `#d95926` | Fine-tuned (BF16) |
| 3 aqua | `#1baf7a` | `#199e70` | Quantized (FP8/NVFP4/AWQ) |
| 4–8 | yellow, magenta, green, violet, red | (bước dark riêng) | runs thêm |
Status: good `#0ca30c` · warning `#fab219` · serious `#ec835a` · critical `#d03b3b` (luôn kèm icon).
Surface: light `#fcfcfb` / dark `#1a1a19`.
⚠ **Không** dùng xanh lá NVIDIA (#76B900) làm series color — dễ nhầm với status "good". Có thể dùng làm accent thương hiệu
(logo/nav) nếu team muốn. (quan điểm)

Nguồn: [Okabe-Ito palette — ggokabeito](https://malcolmbarrett.r-universe.dev/ggokabeito/doc/manual.html) ·
[Adobe Spectrum — color for data viz](https://spectrum.adobe.com/page/color-for-data-visualization) ·
[Cloudscape data-vis colors](https://cloudscape.design/foundation/visual-foundation/data-vis-colors/) ·
[Pluma colors](https://pluma.customer.io/data-visualization/colors) · [Kumo chart colors](https://kumo-ui.com/charts/colors) ·
[Calgary data-viz palette](https://summit.calgary.ca/visual-identity/colour/data-visualization-colour-palette.html)

## 6. Design tokens & design system

- **W3C Design Tokens Format Module 2025.10** — bản stable đầu tiên (28/10/2025), vendor-neutral, hỗ trợ theming/
  multi-brand, màu Display P3/OKLCH, alias & kế thừa. Không phải W3C Standard nhưng đã được coi là ổn định.
  ⇒ Lưu tokens ở `design/tokens/*.tokens.json` theo format DTCG, build ra CSS variables (+ Python dict cho chart
  matplotlib/plotly nếu backend xuất report tĩnh). Công cụ build cụ thể (Style Dictionary v4+…) chốt với bao-frontend.
- Cấu trúc token đề xuất: `color.surface.*`, `color.text.*`, `color.series.1..8`, `color.status.*`,
  `color.seq.blue.100..700`, `color.div.*`, `space.*` (4px grid), `radius.*`, `font.*`, `chart.*` (line 2px, marker ≥8px).
- Typography: system sans cho UI; mono (vd JetBrains Mono / system mono) cho run_id, config, log. Số trong bảng: tabular-nums.
- Theme: `prefers-color-scheme` + toggle `data-theme` thắng cả 2 chiều. Mặc định dark (người dùng ML hay nhìn lâu) — (quan điểm, cần hỏi user).

Nguồn: [Design Tokens spec reaches first stable version](https://www.w3.org/community/design-tokens/2025/10/28/design-tokens-specification-reaches-first-stable-version/) ·
[Design Tokens Format Module 2025.10](https://www.w3.org/community/reports/design-tokens/CG-FINAL-format-20251028/)

## 7. Sơ đồ kiến trúc cho docs

- **Mermaid** render sẵn trên GitHub/Markdown → mặc định dùng `flowchart LR` cho pipeline (ổn định, render ở mọi nơi).
- `architecture-beta` (Mermaid ≥ 11.1) có group/service/icon nhưng **chưa xác minh** GitHub render được → chỉ dùng khi đã test.
- Quy ước: mỗi sơ đồ ≤ ~15 node; 1 sơ đồ = 1 câu hỏi (pipeline tổng, data flow, deployment container, lineage artifact);
  ghi chú phần cứng/version ở caption; nguồn `.mmd` commit cùng docs, SVG export khi cần nhúng vào dashboard.
- Màu trong mermaid: dùng `classDef` theo token (stage data / train / quantize / eval / serve), không tô màu tuỳ hứng.

Nguồn: [Retype — Mermaid](https://retype.com/components/mermaid/) ·
[Paradime — Mermaid architecture diagrams](https://docs.paradime.io/app-help/integrations/mermaid-js/architecture-diagrams)

## 8. Pitfalls (đã thấy ở dashboard ML)

1. Vẽ "GPU memory" từ nvidia-smi trên GB10 → luôn N/A hoặc sai. Dùng unified memory.
2. Biểu đồ live refresh quá dày (1s) làm lag tab và tốn CPU trên máy dùng chung → 5–10s, có pause. (quan điểm)
3. So sánh benchmark khác harness/prompt/độ phân giải như thể cùng điều kiện → bắt buộc hiển thị metadata, cảnh báo khi lệch.
4. Smoothing che mất spike loss / NaN → luôn show raw mờ + marker cho NaN/inf.
5. Màu nhảy khi filter; dual-axis; radar chart cho leaderboard; số trên mọi điểm.
6. Không có empty/loading/error state → người dùng tưởng job chết. Mọi panel cần 4 state: empty · loading · error · stale (dữ liệu cũ > N phút).
7. Throughput đo khác đơn vị (tokens/s vs samples/s vs images/s) → luôn ghi đơn vị và định nghĩa.
8. Quên rằng VLM có input ảnh/video: demo inference & error analysis cần **xem được ảnh + câu hỏi + đáp án đúng + đáp án model** cạnh nhau.

## 9. Checklist áp dụng cho dự án (dùng khi review UI / proposal)

- [ ] Mỗi màn hình trả lời 1 câu hỏi chính của 1 bước trong luồng §1
- [ ] Trạng thái GPU_LOCK + unified memory (stacked, có page cache) hiển thị toàn cục
- [ ] Bảng runs ⇄ chart giữ selection; pin baseline; diff-only config
- [ ] Màu theo entity cố định; ≤8 categorical; status kèm icon+chữ; palette đã chạy validator light+dark
- [ ] Không radar, không dual-axis; Pareto có quality floor + metadata phần cứng/engine
- [ ] Mọi số benchmark truy được lineage (harness version, commit, seed, n)
- [ ] 4 state cho mỗi panel; đơn vị rõ ràng; tabular-nums trong bảng
- [ ] Error analysis VLM: ảnh + prompt + GT + prediction
- [ ] Tokens theo DTCG 2025.10, dark/light đều validate contrast
- [ ] Sơ đồ mermaid ≤15 node, có caption phần cứng/version

## 10. Câu hỏi mở (cần đồng đội trả lời)
- bao-backend: định dạng log metrics (MLflow? TensorBoard? JSONL?), nguồn GPU util/power trên GB10, lineage IDs.
- bao-research/bao-tester: danh sách benchmark chốt, thang điểm, cách tính CI/n mẫu (Proposal 002).
- bao-frontend: stack (Next.js vs Streamlit/Gradio vs nhúng MLflow UI) — ảnh hưởng mức độ tuỳ biến design system.
- bao-cleanup: ngưỡng RAM an toàn trên máy dùng chung.
- bao-pm/user: ai dùng dashboard, truy cập từ đâu (localhost / LAN), có cần auth không (→ bao-security).

---
## 11. Bổ sung sau RESET (2026-10-12): UX no-code node-graph
Spec đầy đủ: `.team/proposals/010-nocode-ux.md`; skill: `design-nocode-node-ux`.
- ComfyUI: cổng/dây cùng màu theo kiểu, chỉ nối cùng màu; reroute native; subgraph gom node + expose widget cho template; lỗi missing node hiện
  trước khi queue nhưng có thể thành chuỗi dialog kẹt UI → bài học: không modal chain. ([links](https://docs.comfy.org/development/core-concepts/links),
  [subgraphs](https://blog.comfy.org/p/subgraphs-are-coming-to-comfyui), [troubleshooting](https://www.rundiffusion.com/troubleshooting-workflows-in-comfyui))
- Unreal Blueprint: exec pin trắng quyết định thứ tự, data pin màu theo kiểu; màu chỉ là gợi ý, cùng màu vẫn có thể không tương thích.
  ([UE 4.26 scripting](https://docs.unrealengine.com/4.26/ProgrammingAndScripting/Blueprints/Scripting), [forum](https://forums.unrealengine.com/t/ue4-node-lines-colour-meanings/404255))
- LlamaBoard: form 4 tab Train/Eval/Chat/Export, zero-code — đơn giản nhưng không có graph/lineage. ([docs](https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html))
- Novice & VPL: blank canvas là rào cản lớn (InstructPipe 2024, [arXiv 2312.09672](https://arxiv.org/pdf/2312.09672)); novice làm tốt hơn với control-flow
  ([Brunel](https://bura.brunel.ac.uk/handle/2438/9217)) → bù bằng số thứ tự bước + làn bước.
- Lỗi: ngôn ngữ thường, chỉ rõ vấn đề, gợi ý xây dựng, không đổ lỗi ([NN/g](https://www.nngroup.com/videos/error-message-communication-guidelines/));
  inline validation ~500 ms. Progressive disclosure cho Simple/Advanced, người dùng chủ động & đảo được ([NN/g](https://www.nngroup.com/videos/progressive-disclosure/)).
- Màu cổng: validator → chỉ 3 hue an toàn all-pairs; cổng dùng 4 họ (3 màu + ink) + hình dạng + nhãn luôn hiện.
