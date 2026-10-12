# Proposal 010 — UX cho người dùng no-code: canvas node-graph, thư viện node, template, kiểm tra trước khi chạy, lỗi dễ hiểu
Tác giả: bao-design · 2026-10-12 · DRAFT · Task R-3 (BOARD)
Reviewer: bao-pm, bao-frontend (R-2/009 — kiến trúc node-graph), bao-backend (node type system, executor), bao-research (R-1),
bao-fixcode (mã lỗi), bao-cleanup (panel Dung lượng), bao-security (license, tải model), bao-tester (benchmark panel).
Liên quan: P006 + phụ lục (panel monitor/benchmark/Pareto vẫn dùng, trở thành panel bên trong app), `design/tokens/`.

> Ký hiệu: **(quan điểm)** = ý kiến thiết kế của tôi, chưa có bằng chứng định lượng; các chỗ khác có nguồn ở cuối.

---

## 1. Người dùng & mục tiêu
**Persona chính — "Lan", kỹ sư nghiệp vụ / nhà nghiên cứu ứng dụng**: biết mình muốn model đọc hoá đơn/biểu đồ/ảnh y tế tốt hơn,
có thư mục ảnh + câu hỏi/đáp án, **không viết Python, không biết LoRA/NVFP4 là gì**. Thành công = trong 1 buổi: mở template →
trỏ tới dữ liệu → bấm Chạy → thấy model mới tốt hơn bao nhiêu → xuất ra dùng được.
**Persona phụ — "Minh", ML engineer trong team**: cần chế độ Advanced, sửa tham số, xem log, so sánh run, chia sẻ workflow file.

Nguyên tắc (xếp theo ưu tiên):
1. **Mở là chạy**: template có mặc định an toàn, chạy được ngay trên dataset mẫu nhỏ.
2. **Chặn lỗi trước khi chạy**, không để lỗi xảy ra sau 3 giờ train (CLAUDE.md "Tầm nhìn sản phẩm").
3. **Ngôn ngữ của người dùng** ("Dạy model bằng dữ liệu của bạn"), thuật ngữ kỹ thuật chỉ là chú thích phụ.
4. **Ít node, node to** (quan điểm): mỗi node = 1 việc người dùng hiểu được, không phải 1 hàm. Khác ComfyUI (node mịn) —
   ComfyUI phục vụ người thích "vọc", Lan thì không. Ai cần mịn → mở rộng node thành subgraph ở Advanced.
5. **Không bao giờ chỉ dùng màu** để mang nghĩa (cổng, trạng thái, lỗi): luôn kèm hình dạng + chữ.

## 2. Mô hình khái niệm
```
Workflow (file .vlmflow.json, chia sẻ được)
 └─ Node  (1 bước: "Tải dữ liệu", "Fine-tune (LoRA)", "Nén model", "Đo điểm benchmark"…)
     ├─ Cổng vào/ra có KIỂU (Dữ liệu · Model · Kết quả đo · Dịch vụ chạy thử · giá trị đơn)
     ├─ Tham số: Simple (2–4, có preset) | Advanced (đầy đủ)
     └─ Trạng thái: chưa sẵn sàng · sẵn sàng · đang chờ · đang chạy x% · xong · lỗi · dùng lại kết quả cũ (cache)
Run = 1 lần chạy workflow → tạo artifacts có lineage (P006/contract: run.json parent_run_id)
```
- **Dataflow, không có dây "exec"** kiểu Blueprint (quan điểm): pipeline của ta tuyến tính theo dữ liệu, dây exec chỉ thêm rối.
  Bù lại nhược điểm "không thấy thứ tự chạy" (nghiên cứu Brunel: novice làm tốt hơn với control-flow) bằng **số thứ tự bước tự
  động** trên mỗi node (①②③ theo topo-sort) và **làn bước** trên canvas (§3).
- **Chạy lại thông minh** (học ComfyUI): chỉ chạy lại node có input/tham số đổi; node không đổi hiện "↺ dùng lại kết quả 14:02".
  Với training đây là tính năng tiết kiệm hàng giờ — **cần bao-backend xác nhận khả thi** (hash input+params → cache key).

## 3. Bố cục màn hình chính
```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ▣ VLM Studio  Workflow: Fine-tune hoá đơn ▾  [Simple ◉ | Advanced ○]   GPU ● Rảnh   RAM 58/121 GB   💾 1.4 TB │
├────────────┬──────────────────────────────────────────────────────────────────────┬──────────────────┤
│ THƯ VIỆN   │  ① NGUỒN       ② CHUẨN BỊ    ③ HUẤN LUYỆN   ④ CHẠY THỬ   ⑤ NÉN   ⑥ ĐO ĐIỂM   ⑦ XUẤT │ THUỘC TÍNH       │
│ 🔍 tìm…    │ ┌──────────┐   ┌──────────┐  ┌────────────┐               ┌──────┐  ┌───────┐        │ Fine-tune (LoRA) │
│ ▸ Dữ liệu  │ │■ Thư mục │──▶│■ Chia    │─▶│● Fine-tune │──●──────────▶ │● Nén │─▶│◆ Đo   │        │ Dạy model bằng   │
│ ▸ Model    │ │  ảnh+QA  │■  │ train/val│■ │  (LoRA)    │●              │ FP8  │● │ điểm  │◆       │ dữ liệu của bạn. │
│ ▸ Huấn luy.│ └──────────┘   └──────────┘  │ ⏱ ~2h10    │               └──────┘  └───────┘        │ Mức: [Nhanh|Cân  │
│ ▸ Nén      │ ┌──────────┐                 └────────────┘                                        │  bằng|Chất lượng]│
│ ▸ Đo điểm  │ │● Model   │──●─────────────────▲                                                  │ Số vòng: 2       │
│ ▸ Xuất     │ │ Qwen3-VL │                                                                       │ ▸ Nâng cao (12)  │
│ ★ Template │ └──────────┘                                                                       │ ⓘ LoRA là gì?    │
├────────────┴──────────────────────────────────────────────────────────────────────┴──────────────────┤
│ ✓ Kiểm tra: 6/7 ổn · ⚠ 1 cảnh báo: RAM dự kiến 46 GB, còn trống 52 GB (sát)   [Xem chi tiết]   [▶ Chạy]   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
- **Làn bước (swimlane) nền** ①…⑦ theo pipeline CLAUDE.md: node tự rơi vào làn của nhóm → layout trái→phải tự nhiên, giảm
  "spaghetti" (quan điểm; vấn đề rối dây là phàn nàn phổ biến của node editor). Nút **"Sắp xếp gọn"** (auto-layout) luôn có.
- Làn ① = **Nguồn** (dữ liệu ở trên, model gốc ở dưới) — model gốc cũng là "nguyên liệu đầu vào" như dữ liệu; tránh thêm làn ⓪.
- Panel phải = thuộc tính node đang chọn (không mở modal). Mỗi cổng vào có dropdown **"Nhận từ: [node · cổng ▾]"** chỉ liệt kê nguồn tương thích
  — cách nối dây không cần chuột (React Flow không hỗ trợ tạo dây bằng bàn phím); chế độ "Danh sách bước" dùng lại component này.
- Editor: desktop ≥ 1280 px; màn nhỏ hơn chỉ xem. Thumbnail template = SVG sinh từ graph JSON. Thanh đáy = **thanh kiểm tra** luôn hiện trạng thái (§6) + nút Chạy.
- Header toàn cục kế thừa P006: GPU lock (ngôn ngữ thường: "GPU ● Rảnh / ● Đang bận: <ai> chạy <việc>"), RAM theo MemTotal thật,
  chip **💾 Dung lượng** mở panel Dung lượng (§9).
- Khi chạy: canvas giữ nguyên, node hiện progress; click node đang chạy → panel phải chuyển sang **Theo dõi** (biểu đồ loss,
  ETA, log rút gọn — các panel monitor của P006). Node Đo điểm xong → panel **Kết quả** (Scorecard/Pareto của P006).

## 4. Cổng & dây: kiểu dữ liệu
Bài học: ComfyUI và Blueprint mã hoá kiểu **chủ yếu bằng màu** (ComfyUI: "chỉ nối được cổng cùng màu"); Blueprint có cổng
khác kiểu cùng màu vẫn không nối được → màu một mình vừa không đủ cho người mù màu, vừa gây hiểu sai.
Tôi đã chạy validator CVD: chỉ **3 màu** an toàn khi mọi cặp có thể đứng cạnh nhau (canvas = mọi cặp). Do đó:

| Kiểu (nhãn hiển thị) | Hình cổng | Màu light / dark | Ví dụ |
|---|---|---|---|
| **Dữ liệu** (Dataset) | ■ vuông | aqua `#1baf7a` / `#199e70` | thư mục ảnh+QA, split train/val |
| **Model** (mọi model: gốc, đã train, đã nén) | ● tròn | blue `#2a78d6` / `#3987e5` | Qwen3-VL gốc, model sau LoRA, model FP8 |
| **Kết quả đo** (Report) | ◆ thoi | orange `#eb6834` / `#d95926` | điểm benchmark, số đo tốc độ |
| **Dịch vụ chạy thử** (Endpoint) | ▲ tam giác | ink trung tính (text.secondary) | model đang phục vụ để chat thử/đo |
| Giá trị đơn (số, chữ, ảnh mẫu) | • chấm nhỏ | muted | prompt, ngưỡng |
- **Map với type system của 009** (8 kiểu backend → 4 họ hiển thị; kiểu con hiện bằng chip + hình biến thể):
  | Kiểu 009 | Họ | Hình | Nhãn |
  |---|---|---|---|
  | Dataset | Dữ liệu | ■ | "Dữ liệu" |
  | Model | Model | ● | "Model" |
  | Adapter | Model | ◐ (nửa tròn) | "Phần học thêm (LoRA)" |
  | QuantizedModel | Model | ◉ (tròn viền đôi) | "Model · đã nén FP8/NVFP4/INT4" |
  | ExportArtifact | Model | ⬢ | "Model đã xuất" |
  | EvalReport, BenchReport | Kết quả đo | ◆ / ◇ | "Điểm benchmark" / "Tốc độ & bộ nhớ" |
  | Endpoint | Dịch vụ | ▲ | "Model đang chạy thử" |
  Token: `color.port.<dataset|model|report|endpoint|value>` (5 token màu) + `shape.port.<kiểu 009>`; **không** tạo 8 màu riêng (FAIL CVD all-pairs).
  Khi kéo dây, việc làm sáng cổng tương thích mới là cơ chế chính để phân biệt kiểu con, không phải màu.
- **Nhãn kiểu luôn hiện cạnh cổng** (không chỉ tooltip). Dây = màu cổng nguồn, 2px; dây Model của model đã nén có nét đôi (quan điểm).
- Phân loại con của Model (gốc / đã train / đã nén FP8…) hiện bằng **chip trong nhãn** ("Model · đã nén FP8"), không bằng màu mới.
  Đầu vào có thể đòi hỏi kiểu con (vd "Nén" nhận Model chưa nén) → kiểm tra ở bước nối.
- Lưu ý: blue/orange ở canvas = *kiểu dữ liệu*; ở panel Kết quả blue/orange = *model gốc / đã fine-tune* (P006). Hai ngữ cảnh
  tách bạch về vị trí; chấp nhận được (blue "Model" ≈ blue "base"), cần test với người dùng.

**Khi nối dây** (React Flow `isValidConnection`, hoặc tương đương trong lib R-1 chọn):
- Bắt đầu kéo → mọi cổng tương thích **sáng lên**, cổng không hợp **mờ đi** (feedback trước khi thả).
- Thả vào cổng sai → dây bật lại + toast 1 dòng: "Cổng **Nén** cần *Model*, bạn đang nối *Dữ liệu*. Gợi ý: nối từ cổng ● của
  node 'Fine-tune'." Không modal.
- Thả dây ra khoảng trống → mở **ô tìm node đã lọc theo kiểu tương thích** (pattern ComfyUI: kéo dây ra để thêm node phù hợp).

## 5. Thư viện node
- Nhóm theo 7 bước (khớp làn): Dữ liệu · Chuẩn bị · Huấn luyện · Chạy thử · Nén · Đo điểm · Xuất. Mỗi nhóm có icon riêng.
- Mỗi node: **tên hành động** + **1 câu mô tả đời thường** + nhãn kỹ thuật nhỏ + huy hiệu tài nguyên:
  ```
  ● Fine-tune nhẹ (LoRA)                                  ⏱ 1–3 giờ · RAM ~30–50 GB
  Dạy model trả lời theo dữ liệu của bạn, tốn ít bộ nhớ.   kỹ thuật: NeMo AutoModel · LoRA
  ```
- Tìm kiếm khớp cả tiếng Việt/Anh/từ đồng nghĩa ("nén" ↔ "quantize" ↔ "FP8"; "chấm điểm" ↔ "benchmark").
- Danh sách node v1 đề xuất (map với stage list của bao-research, tool trong ngoặc là "proposed" chờ ADR):
  | Bước | Node (nhãn người dùng) | Engine |
  |---|---|---|
  | Dữ liệu | Thư mục ảnh + câu hỏi/đáp án · Dataset Hugging Face · Dataset mẫu (demo) | HF datasets / JSONL |
  | Chuẩn bị | Kiểm tra & làm sạch dữ liệu · Chia train/val | (NeMo Curator, tuỳ chọn) |
  | Model | Chọn model gốc (danh sách đã kiểm license) | HF / NGC |
  | Huấn luyện | Fine-tune nhẹ (LoRA) · Fine-tune nhẹ tiết kiệm RAM (QLoRA) · Fine-tune toàn bộ (Advanced) · Học từ phản hồi (GRPO, Advanced) | NeMo AutoModel / NeMo-RL |
  | Chạy thử | Chat thử với model (ảnh/video) · So sánh 2 model cạnh nhau | vLLM |
  | Nén | Nén model (Nhanh/Cân bằng/Nhỏ nhất = FP8/NVFP4/INT4-AWQ) | ModelOpt |
  | Đo điểm | Đo điểm benchmark (chọn bộ: Tài liệu, Biểu đồ, Toán, Video…) · Đo tốc độ & bộ nhớ | lmms-eval / VLMEvalKit + perf |
  | Xuất | Lưu model (Hugging Face) · Xuất để chạy nhanh (vLLM/TensorRT-LLM) | HF safetensors / TRT-LLM |
- Preset ngôn ngữ thường thay tên kỹ thuật ở Simple: Nén **Nhanh nhất giữ chất lượng (FP8)** / **Cân bằng (NVFP4)** /
  **Nhỏ nhất (INT4-AWQ)**; benchmark theo **năng lực** (bao-research xác nhận 2026-10-12): "Đọc tài liệu & chữ trong ảnh" = DocVQA + OCRBench ·
  "Hiểu biểu đồ & sơ đồ" = ChartQA + AI2D · "Kiến thức & suy luận đa ngành" = MMMU · "Nhận thức ảnh tổng quát" = MMBench ·
  "Toán có hình" = MathVista · "Video" = Video-MME.
  **Không hiện điểm gộp** (khác thang/độ khó/số mẫu; trung bình che 1 benchmark tụt mạnh). Simple hiện **badge theo nhóm**:
  "↑ tốt hơn / ≈ như cũ / ↓ kém hơn" dựa trên delta_ci95 paired từng benchmark; có 1 benchmark ↓ ngoài CI → nhóm hiện ↓ (kèm tên benchmark).
  Badge = icon + chữ + màu `color.delta.*`; bấm vào → mở bảng điểm từng benchmark.

## 6. Kiểm tra trước khi chạy (pre-flight) — chống lỗi là tính năng số 1
Chạy liên tục khi sửa graph (debounce ~500 ms), hiển thị ở thanh đáy + dấu trên node; nút **Chạy** chỉ khoá với lỗi "chặn".

| # | Kiểm tra | Mức | Thông điệp mẫu |
|---|---|---|---|
| 1 | Cổng bắt buộc chưa nối / tham số thiếu | chặn | "Node **Fine-tune** chưa có dữ liệu. Nối cổng ■ Dữ liệu từ node ①." [Chỉ cho tôi] |
| 2 | Kiểu không khớp / chu trình | chặn | (đã chặn khi nối; kiểm lại khi mở file workflow) |
| 3 | Dữ liệu sai định dạng (thiếu ảnh, cột câu hỏi rỗng) | chặn | "12 / 3 400 mẫu không có ảnh. [Bỏ qua 12 mẫu] [Xem danh sách]" |
| 4 | **RAM dự kiến** vs MemAvailable (unified memory, máy dùng chung) | chặn nếu > available − biên an toàn; cảnh báo nếu sát | "Cần ~46 GB, máy đang trống 31 GB (các dịch vụ khác đang dùng). [Dùng QLoRA: ~22 GB] [Giảm độ phân giải ảnh]" |
| 5 | **Disk** dự kiến (checkpoint, model tải, kết quả) | chặn/ cảnh báo | "Cần ~60 GB, còn 40 GB. [Giải phóng 35 GB từ kết quả tạm]" (§9) |
| 6 | **Thời gian ước lượng** | thông tin | "Khoảng 2–3 giờ trên DGX Spark" (dải, không giả chính xác) |
| 7 | GPU đang bận (GPU_LOCK) | xếp hàng | "GPU đang được *bao-backend* dùng cho *sft-lora* (từ 14:02). Workflow sẽ chờ và tự chạy khi rảnh. [Xếp hàng]" — không nút dừng job người khác |
| 8 | Tải model/dataset lớn > 20 GB | cần duyệt | "Sẽ tải 33 GB. Cần quản trị viên đồng ý (quy định máy dùng chung)." |
| 9 | License model/dữ liệu (P003) | chặn / cảnh báo | "Model này chỉ cho nghiên cứu, không dùng thương mại." / "Bộ dữ liệu đánh giá đang nằm trong dữ liệu train — điểm sẽ bị ảo." |
| 10 | Tham số Advanced lệch xa mặc định an toàn | cảnh báo | "Learning rate 1e-2 cao bất thường (mặc định 1e-4) — dễ làm hỏng model." |
- **Ước lượng phải có nguồn**: RAM/thời gian/disk do backend tính (công thức + bảng đo thực tế trên GB10 từ bao-tester/bao-fixcode
  memguard), UI chỉ hiển thị **dải** + "độ tin cậy" (đo thật / ước lượng). Không có ước lượng → hiển thị "chưa ước lượng được", không bịa số.
- Mọi gợi ý sửa là **nút làm ngay** (đổi sang QLoRA, bỏ mẫu lỗi), có Undo.

## 7. Thông báo lỗi dễ hiểu (khi đang chạy)
Khuôn 4 phần (theo NN/g: ngôn ngữ người đọc được, chỉ rõ vấn đề, gợi ý xây dựng, không đổ lỗi):
```
┌ ✕ Fine-tune dừng ở bước 1 240 / 5 000 ─────────────────────────────────────────────┐
│ Chuyện gì xảy ra: Máy hết bộ nhớ (RAM dùng chung với các dịch vụ khác).             │
│ Vì sao: Ảnh độ phân giải cao (1344px) × batch 8 cần nhiều bộ nhớ hơn dự kiến.        │
│ Cách sửa:  [Chạy lại với batch 4]  [Giảm ảnh xuống 896px]  [Tiếp tục từ checkpoint 1 000]│
│ ▸ Chi tiết kỹ thuật (cho kỹ sư) — mã lỗi MEM-003 · log · lệnh tái hiện  [Sao chép]   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
- Lỗi hiện **trên chính node** (viền đỏ + icon ✕ + chữ "Lỗi") và trong panel phải; không chuỗi modal chặn (pitfall đã ghi nhận ở ComfyUI:
  dialog lỗi nối tiếp làm kẹt giao diện).
- Hệ mã lỗi **duy nhất** = catalog của bao-fixcode (`MEM-001…`, `DISK-001`, `DATA-001`, `CFG-001`, `SYS-000`…), action `kind: patch|resume|navigate` (+`confirm`).
  **Câu chữ (VI/EN) có 1 nguồn duy nhất**: `src/vlm_pipeline/i18n/{vi,en}.yaml` (bao-design sở hữu, runtime đọc được), khoá dạng
  `node.<id>.title|desc`, `error.<code>.title|what|why`, `action.<id>`, `consequence.<code>`; NodeSpec và `error_catalog.yaml` chỉ mang **key + params**
  (vd `{est_gb}`), backend merge theo locale. Mã/match kỹ thuật vẫn do fixcode giữ trong `error_catalog.yaml`.
- Stack trace không bao giờ là thông điệp chính; luôn có thể xem trong "Chi tiết kỹ thuật".

## 8. Template gallery — "mở là chạy"
```
Bắt đầu nhanh                                                      🔍  Lọc: [Tất cả|Fine-tune|Nén|Đánh giá]
┌──────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐
│ [sơ đồ thu nhỏ 5 node]    │ │ [sơ đồ thu nhỏ 3 node]    │ │ [sơ đồ thu nhỏ 4 node]    │
│ Dạy model đọc tài liệu    │ │ Nén model & đo lại điểm   │ │ So sánh 2 model trên ảnh  │
│ của bạn (LoRA)            │ │ (FP8 / NVFP4)             │ │ của bạn                   │
│ ⏱ 2–3 h · RAM ~35 GB      │ │ ⏱ 40 phút · RAM ~25 GB    │ │ ⏱ 10 phút · RAM ~20 GB    │
│ ● Cơ bản · Dữ liệu mẫu ✓  │ │ ● Cơ bản                  │ │ ● Cơ bản                  │
│ [Dùng template]  [Xem]    │ │ [Dùng template]  [Xem]    │ │ [Dùng template]  [Xem]    │
└──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘
```
- Mỗi template: kết quả đầu ra (bằng lời), thời gian/RAM/disk dự kiến, độ khó, **chạy được ngay với dataset mẫu nhỏ** (smoke ~5–10 phút),
  bước "thay dữ liệu của bạn" được đánh dấu bằng node viền nét đứt "👉 Thay bằng dữ liệu của bạn".
- **Chế độ chạy thử**: mọi template có tham số chung `run_mode: smoke|full` — smoke chạy **chính graph đó** với model nhỏ (~0.8B),
  vài bước train, benchmark giới hạn ~20 mẫu, đo tốc độ 1 lần (bao-tester). Nút trên thẻ: [Chạy thử nhanh ~N phút] · [Chạy đầy đủ].
  Thời gian smoke thực tế trên GB10 gồm cả nạp model/JIT/khởi động vLLM → hiển thị theo estimator, không hứa "5 phút".
- Số ⏱/RAM/disk trên thẻ **lấy từ `estimate()`** với tham số mặc định (dải [lo,hi] + `source`); `unknown` → "chưa có số đo". Không viết tay.
- Template nén/đo lại **tự chứa benchmark bản gốc BF16 cùng cấu hình** để có Δ paired; template đánh giá model gốc hiện số công bố cạnh số đo (gate G1).
- Model/dataset trong template = AssetRef pin revision (P003); "Dữ liệu mẫu" đi kèm app cần bao-security xác nhận license redistribute.
- Mở template → tạo **bản sao** (template gốc không bị sửa) + tour 3 bước (tuỳ chọn, bỏ qua được).
- Template v1 đề xuất: (1) Fine-tune LoRA VLM trên dữ liệu ảnh+QA, (2) Quantize FP8/NVFP4 + benchmark lại, (3) Full pipeline
  data→LoRA→quantize→benchmark→export, (4) Chỉ đánh giá model gốc trên bộ benchmark, (5) Chat thử / so sánh 2 model.
  Template là file workflow trong repo (`templates/*.vlmflow.json`), có bộ test bắt buộc của bao-tester (validate 0 lỗi trên profile GB10, MEM-001 trên profile RAM thấp, smoke CPU FakeWorker, smoke GPU nightly, schema mới nhất, DATA-* khi thay dữ liệu sai format, file gốc không đổi byte khi user sửa bản sao).
- Màn hình trống lần đầu = gallery, không phải canvas trắng (chống "blank canvas problem" — InstructPipe 2024).

## 9. Panel "Dung lượng" (theo yêu cầu bao-cleanup, P005 §6 R2)
- Vị trí: chip **💾** trên header → mở **drawer bên phải** (không rời canvas); cũng mở từ gợi ý của pre-flight #5.
- Nhóm theo ngôn ngữ người dùng: **Model đã tải** · **Model bạn đã train** · **Dữ liệu** · **Kết quả tạm (có thể tính lại)**.
  Không hiện optimizer/DCP/đường dẫn ở Simple; Advanced mới hiện chi tiết.
- Mỗi mục: tên · dung lượng · dùng lần cuối · "được dùng bởi workflow…" · nút **Xoá** (hoặc 🔒 + lý do theo `lock_reason_code`: "Đang được workflow X dùng" / "Đã ghim (KEEP)" /
  "Adapter Z còn cần model gốc này" / "Được bảo vệ theo quy định an toàn"). **Model đã train không khoá mặc định** — người dùng xoá được, chỉ cần gõ tên.
- Dữ liệu: shape v2 `items[]` của bao-cleanup (P005); hậu quả bằng lời sinh từ `consequence_code + params` (copy VI/EN do design viết).
- Dialog xác nhận (sau khi hệ thống chạy dry-run):
```
┌ Xoá 3 mục, giải phóng 41.2 GB? ───────────────────────────────────────┐
│ • Qwen3-VL-8B (Model đã tải, 17 GB)                                    │
│   → Workflow "Đọc hoá đơn" sẽ phải tải lại 17 GB lần chạy sau.         │
│ • Kết quả tạm của run sft-0412 (22 GB) → Có thể tính lại, ~1 giờ.       │
│ • Dataset mẫu cũ (2.2 GB)                                              │
│ ⚠ Phải tải lại 17 GB. Gõ  Qwen3-VL-8B  để xác nhận: [____________]    │
│                                         [Huỷ]   [Xoá vĩnh viễn] (khoá) │
└────────────────────────────────────────────────────────────────────────┘
```
  Gõ tên **chỉ khi** mục không tính lại được (model đã train) hoặc phải tải lại > 20 GB; mục tính lại được dù lớn → chỉ nút xác nhận.
  Tối đa 1 ô gõ mỗi dialog (`confirm_text` = mục rủi ro nhất, server trả). Nút xoá mặc định **không** được focus;
  xoá xong hiện "Đã xoá lúc X" (đọc chung event `hf_model_removed | dataset_removed | cache_evicted | checkpoint_pruned`). Panel team (P006 annex (i)) vẫn read-only.

## 10. Simple / Advanced
- **Toggle toàn cục** trên header + **"▸ Nâng cao (n)" trong từng node** (progressive disclosure, người dùng chủ động, đảo được).
- Simple: 2–4 tham số/ node với preset (Nhanh / Cân bằng / Chất lượng) + giải thích bằng lời; Advanced: tham số đầy đủ, tên kỹ thuật,
  xem YAML/CLI tương đương (Minh copy chạy CLI được — giữ "1 lệnh CLI reproducible" của CLAUDE.md).
- Giá trị đã sửa ở Advanced **vẫn hiển thị ở Simple** dưới dạng huy hiệu "⚙ 3 tuỳ chỉnh" — không để thay đổi ẩn.
- **"Bung" node to thành các node con** (subgraph) để tuỳ biến sâu: không có ở v1 (PM 2026-10-12), nhưng node type schema phải chừa chỗ
  (node có thể có `children`/`expand_to` graph con; tham số node cha map xuống node con) — đã báo bao-backend.
- Node chỉ có ở Advanced (Full SFT, GRPO, subgraph mịn) hiện mờ trong thư viện Simple kèm "bật Nâng cao để dùng".

## 11. Design system — bổ sung cho canvas
- Trạng thái node (khớp 009 F5) — mỗi cái = màu + icon + chữ: chưa đủ ▢ (muted) · sẵn sàng ○ · đang chờ ⧗ · ↺ dùng lại · ◐ đang chạy x% · ✓ xong (good) ·
  ⏸ tạm dừng (warning, MEM-002) · ✕ lỗi (critical) · – bỏ qua (muted).
- **Subgraph: v1 không có**, chỉ group trực quan (khung màu nhạt có tên) — đồng ý bao-frontend; template đảm nhận vai trò "gói sẵn". Subgraph → v1.1.
- Kế thừa tokens P006 (`design/tokens/`). Thêm: `color.port.dataset|model|report|endpoint|value`, `canvas.grid`, `canvas.lane`,
  `node.state.*` (ready/queued/running/done/error/cached/blocked — màu status + icon + chữ), `node.width` (240px), `shadow.node`.
- Node: header có icon nhóm + tên; thân: tóm tắt tham số Simple; chân: huy hiệu tài nguyên + trạng thái. Nhãn kiểu cổng luôn hiện.
- Accessibility canvas: mọi node/dây focus được bằng Tab, Enter chọn, mũi tên di chuyển (React Flow hỗ trợ sẵn); **thêm chế độ
  "Danh sách bước"** (outline dạng văn bản tuyến tính của graph, sửa được) cho người dùng bàn phím/screen reader và cho Lan khi graph phức tạp
  (quan điểm — React Flow chưa xác nhận hỗ trợ tạo dây bằng bàn phím).
- Song ngữ VI/EN từ đầu: mọi chuỗi qua file copy (`design/copy/*.yaml`), không hard-code.

## 12. Ảnh hưởng tới các proposal khác (R-4)
- **P006**: dashboard không còn là app riêng; Runs/Monitor/Scorecard/Pareto/Error analysis trở thành **panel trong app** (mở từ node hoặc
  tab "Lịch sử chạy"). Tokens, badge, quy tắc chart giữ nguyên. Chủ trương "UI read-only, hành động = CLI" **bị thay**: app giờ khởi chạy job
  (qua executor của 009), vẫn giữ: không kill job người khác, GPU lock, duyệt tải > 20 GB.
- **Stack (P006 phương án A/B)**: canvas node-graph cần React (React Flow) hoặc LiteGraph (ComfyUI) → phương án A (Gradio thuần) gần như
  loại; quyết định cuối thuộc R-1/R-2.
- **Backend (008/009)**: cần `error_code` chuẩn, ước lượng RAM/disk/time per node, cache key per node, node type schema (cổng có kiểu + kiểu con).

## 13. Câu hỏi mở
1. (R-1) Fork ComfyUI/LiteGraph hay React Flow tự xây? Ảnh hưởng UX: LiteGraph có sẵn search-on-drag, reroute, subgraph; React Flow dễ theo
   design system/a11y hơn. Design không chặn lựa chọn nào, nhưng các yêu cầu §4, §6, §11 phải làm được.
2. (backend) Executor có hỗ trợ cache theo node và resume từ checkpoint không (ảnh hưởng thông điệp "Tiếp tục từ checkpoint")?
3. (backend/tester/fixcode) Nguồn số ước lượng RAM/thời gian trên GB10 cho từng node + model size.
4. (PM/user) Người dùng thật có ai ngoài team? (Tạm: VI mặc định + EN, Simple mặc định — PM 2026-10-12, chờ user xác nhận.)
5. ~~(research) Gom benchmark theo năng lực~~ — đã chốt, xem §5.

## 14. Kế hoạch (sau khi PM chốt cùng R-1/R-2)
1. Test giấy/wireframe với 1–2 người không biết code (PM nhờ user giới thiệu) — 5 task: mở template, thay dữ liệu, sửa lỗi nối, đọc kết quả, xoá dung lượng.
2. Hi-fi cho: canvas + node + cổng, thanh kiểm tra, dialog lỗi, gallery, drawer Dung lượng (light/dark).
3. Copy deck VI/EN cho node + lỗi. 4. Review chéo với frontend khi build.

## Nguồn
- ComfyUI links (màu theo kiểu, chỉ nối cùng màu, reroute): https://docs.comfy.org/development/core-concepts/links
- ComfyUI subgraphs (gom node, expose widget cho template): https://blog.comfy.org/p/subgraphs-are-coming-to-comfyui
- ComfyUI lỗi missing node / dialog lỗi nối tiếp: https://www.rundiffusion.com/troubleshooting-workflows-in-comfyui ·
  https://wonderfullauncher.com/docs/comfyui-prompt-has-no-outputs
- Unreal Blueprint: exec pin trắng, data pin màu theo kiểu: https://docs.unrealengine.com/4.26/ProgrammingAndScripting/Blueprints/Scripting ·
  forum màu không phải luật cứng: https://forums.unrealengine.com/t/ue4-node-lines-colour-meanings/404255
- LLaMA-Factory LlamaBoard (Train/Eval/Chat/Export, zero-code): https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html
- InstructPipe (blank-canvas problem với novice, 2024): https://arxiv.org/pdf/2312.09672
- Novice: control-flow vs data-flow, 145 vấn đề usability VPL: https://bura.brunel.ac.uk/handle/2438/9217
- React Flow accessibility (Tab/Enter/mũi tên): https://reactflow.dev/learn/advanced-use/accessibility
- NN/g error messages: https://www.nngroup.com/videos/error-message-communication-guidelines/ · progressive disclosure:
  https://www.nngroup.com/videos/progressive-disclosure/
- Inline validation ~500 ms: https://guide.yukon.ca/en/content-and-design/forms/user-experience-ux-standards-forms/how-help-people-recover-form-error
- GB10 memory (nvidia-smi không báo, dùng /proc/meminfo): https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- Validator màu: skill `dataviz` `validate_palette.js` (chạy 2026-10-12: 3 màu cổng PASS adjacent; 4 màu FAIL all-pairs → endpoint dùng ink trung tính).
