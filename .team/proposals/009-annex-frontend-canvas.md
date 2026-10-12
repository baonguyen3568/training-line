# Proposal 009 — Phụ lục F: Canvas/editor, node library, template gallery, live run view (bao-frontend)
Trạng thái: DRAFT · 2026-10-12 · Đi kèm `009-node-graph-architecture.md` (bao-backend: node spec, graph JSON, executor, API).
Phụ thuộc: R-1 (bao-research: fork vs tự xây), R-3 (bao-design: UX no-tech). Chưa chốt — phụ lục này là input để chốt.

## F1. Lựa chọn thư viện canvas — bằng chứng (kiểm tra 2026-10-12)
| | **React Flow (`@xyflow/react`)** | **LiteGraph (`@comfyorg/litegraph`)** | **Fork ComfyUI_frontend** | Rete.js v2 |
|---|---|---|---|---|
| Version / cập nhật | 12.12.0, npm modified 2026-10-11 | 0.17.2 — **npm DEPRECATED** ("Package no longer supported"), release cuối 2025-08-06; bản gốc jagenjo 0.7.18 đứng từ 2024. ComfyUI hiện dùng litegraph nằm trong frontend GPL | v1.4x, push 2026-10-12, ~2k★ | 2.0.6, 2025-06 |
| License | MIT (npm) | MIT (npm) | **GPL-3.0** (GitHub API) — ComfyUI core cũng GPLv3 | MIT |
| Render | DOM, mỗi node là React component | Canvas2D | Vue 3 + LiteGraph | DOM qua plugin React/Vue |
| UI giàu trong node (form, preset, chart nhỏ, badge lỗi) | dễ — node là React, dùng chung component dashboard | khó — widget vẽ tay trên canvas, a11y kém | có sẵn widget kiểu ComfyUI, nhưng hướng ảnh/gen-AI | dễ |
| Typed port / validate nối dây | tự làm qua `isValidConnection` + Handle (ít code) | có sẵn typed slots | có sẵn | first-class |
| Quy mô graph | ổn tới hàng nghìn node khi memo + viewport culling (docs) | tốt nhất cho hàng trăm-nghìn node | như LiteGraph | như React Flow |
| Khớp stack đã chốt (FastAPI + React SPA, P006) | khớp hoàn toàn; tái dùng panel monitor/benchmark | phải bọc trong React | **đổi sang Vue**, bỏ toàn bộ hướng P006 | khớp |
| Executor | không có (backend tự xây — đúng ý P009) | `graph.start()` trong browser (không dùng) | executor ComfyUI: Python in-process, hướng chạy ngắn | plugin |

> **Đính chính 2026-10-12:** bản trước ghi @comfyorg/litegraph "vẫn cập nhật 2026-10-02" — SAI; ngày đó là lúc npm đánh dấu deprecated
> (`npm view @comfyorg/litegraph deprecated time`). Khớp với R-1 của bao-research. LiteGraph vì vậy là lựa chọn yếu hơn nữa.

Nguồn: npm registry (`npm view`), GitHub API repo Comfy-Org/ComfyUI_frontend (license GPL-3.0), reactflow.dev performance guide,
pistack.xyz 2026-09-09 so sánh React Flow/Rete/LiteGraph, README @comfyorg/litegraph.

### Khuyến nghị bao-frontend: **React Flow (`@xyflow/react` 12.x), tự xây editor; học ý tưởng + format từ ComfyUI, không fork code.**
Lý do (theo mức quan trọng):
1. **License**: ComfyUI_frontend + ComfyUI là GPL-3.0 → fork kéo cả app vào copyleft khi phân phối. Cần bao-security/PM/user
   xác nhận chấp nhận được không; nếu không, fork bị loại ngay. React Flow/LiteGraph đều MIT.
2. **Người dùng no-tech cần UI giàu trong node**: preset, slider, cảnh báo RAM/thời gian ước lượng, lỗi tiếng Việt ngay trên node,
   progress + mini loss chart khi chạy. DOM node (React) làm việc này rẻ; canvas widget thì đắt và kém accessibility.
3. **Quy mô graph nhỏ**: workflow training ~5–30 node, không phải hàng nghìn → lợi thế hiệu năng Canvas2D của LiteGraph không dùng tới.
4. **Mismatch executor**: ComfyUI executor chạy node trong cùng process Python, phù hợp gen ảnh vài giây; job của ta chạy hàng giờ
   trong container NGC, cần GPU lock, resume, retention → đằng nào cũng phải viết lại phần lõi (P008/P009 backend).
5. Giữ nguyên quyết định P006 (FastAPI + React SPA, 1 process, 127.0.0.1:7870); panel monitor/scorecard/Pareto tái dùng trực tiếp.
Rủi ro: tự làm typed port, undo/redo, copy-paste, group/subgraph, minimap (React Flow có `MiniMap`, `Controls`, `NodeResizer`; undo/redo tự làm
bằng history của store). Ước lượng effort editor MVP ~2 tuần (chưa đo, ước lượng chủ quan).
Nếu R-1 cho thấy fork ComfyUI lợi hơn hẳn và license ok → phương án B2: ComfyUI custom nodes + frontend extension; mình sẽ đánh giá lại.

## F2. Kiến trúc frontend
```
React SPA (Vite, TS) — served by FastAPI at /
├── store: zustand (graph, selection, history undo/redo, run state)
├── canvas: @xyflow/react  — NodeShell (generic) render từ NodeSpec, không hardcode node
│     ├── Handle per port, màu theo port type (tokens bao-design), isValidConnection = typeCompat(out, in) (subtype ok)
│     └── layout tự động khi mở template: @dagrejs/dagre (MIT) — KHÔNG dùng elkjs (EPL-2.0/GPL)
├── inspector: form sinh từ JSON Schema của params + `ui:` hints (preset, slider, advanced ẩn)
├── node library: GET /api/node-types (search, nhóm theo stage data/train/quantize/eval/export)
├── template gallery: GET /api/templates
├── validation overlay: POST /api/graphs/validate (debounce 500ms) → lỗi gắn theo node/port
├── live run view: WS /api/runs/{id}/events → trạng thái node, progress, metrics mini
└── panels (từ P006): run monitor, scorecard, Pareto, error analysis, demo (/demo Gradio)
```
- **Backend là nguồn sự thật** (đồng ý với bao-backend): UI chỉ biết `NodeSpec` generic. Thêm node mới = thêm spec ở backend, UI không đổi.
- Form engine: tự viết renderer nhỏ cho tập con JSON Schema (string/number/int/bool/enum/array of enum/file-ref) + `ui:` hints,
  thay vì RJSF — bundle nhẹ, kiểm soát UX no-tech, ít deps (security). Tập con được liệt kê trong P009 §Node spec.

## F3. Hợp đồng cần từ backend (đề xuất cho §API của bao-backend)
- `NodeSpec`: `type_id, version, category, icon, inputs[{name, type, optional}], outputs[...], params_schema (JSON Schema),
  ui_schema ({order, widget, preset, advanced}), estimate: bool, runner` — **chỉ mang i18n key** (`node.<type_id>.title|desc|port.<n>|param.<n>.label|help`);
  copy VI/EN 1 nguồn `src/vlm_pipeline/i18n/{vi,en}.yaml` (bao-design sở hữu); API resolve theo `?locale=` + giữ key để fallback.
  Mã lỗi theo catalog bao-fixcode (MEM-001, DISK-001…), copy lỗi `error.<code>.title|what|why`.
- `PortType` registry: `{id, parent (subtype), color_token, label_key}` — UI tô Handle & dây theo `color_token`.
- Graph JSON (workflow file): `{schema_version, id, name, nodes[{id, type_id, version, params, position:{x,y}}],
  edges[{id, from:{node,port}, to:{node,port}}], ui:{viewport, groups[], notes[]}}` — `position`/`ui` là phần UI riêng,
  backend bỏ qua khi tính cache key (không để kéo node làm chạy lại job).
- Validate response: `{ok, errors[{node_id?, port?, code, severity(error|warn|info), message_vi, fix_hint_vi}], estimate:{ram_gb_peak, disk_gb, duration_min, gpu_exclusive}}` — mỗi ước lượng là **dải + source** (đo thật | ước lượng | chưa có); UI hiện dải, không bịa số.
- Run events (WS) dùng lại contract events/metrics hiện có + `{event:"node_status", node_id, status(queued|cached|running|done|failed|skipped|interrupted), progress, eta_s, cached_from, resumable, checkpoint, message_key}` — `cached_from` → nhãn "↺ dùng lại kết quả <ts>"; `resumable`+`checkpoint` → nút "Tiếp tục từ checkpoint".

## F4. Template gallery
- Template = graph JSON + `meta {title_vi, description_vi, difficulty, est_time, est_ram_gb, est_disk_gb, required_models[], tags, preview.png}`.
- Mở template = clone vào workspace (không sửa template gốc), auto-validate ngay, hiện "trước khi chạy cần: tải model X (Y GB), …".
- v1: 3–5 template do research/backend chốt (vd "Fine-tune VLM LoRA", "Quantize FP8 + benchmark", "So sánh base vs fine-tune").

## F5. Live run view
- Node đổi trạng thái (màu + icon + chữ, không chỉ màu): queued ▢ · cached ↺ · running ◐ (progress, ETA) · done ✓ · failed ✕ · skipped –.
- Dây phát sáng nhẹ khi artifact đang truyền; node train hiển thị mini loss sparkline (uPlot) — click mở panel Run monitor (P006).
- Lỗi: message_vi + fix_hint_vi trên node; stack trace chỉ trong "Chi tiết kỹ thuật" (gập).
- Header toàn cục từ P006 (GPU lock, RAM, disk) giữ nguyên trên canvas.

## F6. Chặn lỗi trước khi chạy (phần UI)
- Không cho nối dây khác kiểu (Handle từ chối + tooltip vì sao). Port bắt buộc chưa nối → viền đỏ node.
- Nút "Chạy" disabled khi `errors` có severity=error; warn → hộp thoại xác nhận bằng ngôn ngữ dễ hiểu.
- Ước lượng tài nguyên hiển thị cạnh nút Chạy; vượt MemAvailable hiện tại → chặn với lời giải thích (máy dùng chung).

## F7. Bảo mật & ràng buộc (kế thừa P006 §8; bao-security xác nhận 2026-10-12, P003 D6b/D11/D13)
- **Auth bắt buộc kể cả localhost** (máy dùng chung; user/container network-host gọi được 127.0.0.1): token ngẫu nhiên sinh lúc khởi
  động → mở `http://127.0.0.1:7870/?token=…` → đổi sang cookie `HttpOnly; SameSite=Strict` rồi xoá token khỏi URL (history.replaceState).
- Token: so sánh constant-time (`hmac.compare_digest`), cookie `Path=/`; sinh mới mỗi lần restart (session cũ vô hiệu), file token perm 0600;
  access log redact query `?token=`; **WebSocket xác thực bằng cookie** lúc handshake (không chỉ Origin). (SEC-REVIEW F7: APPROVE)
- Allowlist `Host` header (chống DNS rebinding); kiểm `Origin` mọi POST/PUT/DELETE và handshake WebSocket; header `X-VLM-CSRF` trên
  mọi request ghi (SPA gửi tự động qua fetch wrapper); **không bật CORS**.
- Secret (HF token…): field chỉ-ghi, UI chỉ hiện `hf_…abcd`; trong graph tham chiếu `secret://<name>`, không bao giờ giá trị thật.
- Workflow import (JSON hoặc nhúng PNG) = **untrusted**: schema `extra=forbid`, `type_id` trong allowlist, không tự cài node, path chỉ
  tương đối; export strip secret/path tuyệt đối/hostname. v1 không có node "Python tuỳ ý".
- localhost-only, CSP chặt, không CDN. Deps: @xyflow/react, zustand, @dagrejs/dagre (MIT) — mỗi package có entry `licenses.yaml`
  (kind: code), lockfile commit, `npm ci`; tránh elkjs (EPL/GPL).
- Action có tính phá huỷ trong error catalog (vd "Xoá checkpoint cũ") → mở **drawer "Dung lượng"** (P010 §9) với reason/filter điền sẵn,
  dữ liệu từ `GET /api/storage/plan` (dry-run, `plan_id` hết hạn 10'), xoá qua `POST /api/storage/apply {plan_id, item_ids[], confirm}`
  (server tính lại plan; không có endpoint xoá theo path). Ưu tiên gợi ý cache tính lại được trước checkpoint. Màn team (i) read-only.

## F1b. Phương án 3 (PM yêu cầu đánh giá công bằng): ComfyUI chạy process riêng + custom nodes — góc nhìn frontend
Mô hình: ComfyUI (không sửa) chạy container `vlm-comfyui` 127.0.0.1:78xx; ta viết **custom nodes** (Python) gọi orchestrator của ta
(P008) qua HTTP; UI = ComfyUI_frontend nguyên bản + **frontend extension** (JS, `app.registerExtension`) cho các panel riêng.
| Khía cạnh | Đánh giá |
|---|---|
| Thời gian ra canvas | **nhanh nhất** — canvas, kéo thả, lưu/tải workflow JSON, queue, undo, group, subgraph có sẵn |
| Typed port / widget | có sẵn (type string trên slot; widget INT/FLOAT/COMBO/STRING) — đủ cho params đơn giản |
| UI no-tech (preset, ước lượng RAM, lỗi tiếng Việt trên node, fix-hint) | phải làm qua extension: vẽ thêm lên canvas node / DOM widget; giới hạn bởi API extension, dễ vỡ khi ComfyUI frontend nâng version (release rất dày, v1.4x) |
| Panel monitor/scorecard/Pareto (P006) | extension sidebar/bottom-panel tab nhúng bundle React riêng, hoặc mở SPA riêng ở tab khác → **2 UI, 2 design system** (PrimeVue/Tailwind của Comfy vs tokens của ta) |
| Job dài (giờ) | executor ComfyUI chạy hàng đợi tuần tự; custom node train phải **block chờ** orchestrator hàng giờ → khoá cả queue; hoặc trả về ngay "job handle" → mất ý nghĩa cache/trạng thái node của ComfyUI. Progress qua `PromptServer.send_sync` được, nhưng resume/interrupted (exit 75) không có khái niệm tương ứng |
| Validate trước khi chạy (ước lượng RAM/disk, máy dùng chung) | ComfyUI chỉ validate kiểu & input bắt buộc; phần ước lượng phải chèn vào extension trước khi queue |
| Thuật ngữ & UX | UI hướng gen ảnh (Queue Prompt, model loaders, menu nhiều mục) — phải ẩn/đổi tên nhiều cho no-tech |
| License | ComfyUI + frontend GPL-3.0. Chạy nguyên bản ở process riêng giảm rủi ro, nhưng **custom nodes import module ComfyUI** nhiều khả năng là tác phẩm phái sinh → cần bao-security/user xác nhận |
| Bảo trì | phụ thuộc lịch nâng cấp upstream; pin version + test extension mỗi lần nâng |
| arm64 / máy dùng chung | ComfyUI chạy được arm64 (máy đã có comfy-ui container), thêm 1 process Python + PyTorch (RAM ~1–3GB nếu không load model) |

Kết luận frontend: phương án 3 **thắng về tốc độ ra demo canvas** (vài ngày), **thua về UX no-tech, panel thống nhất, và job dài**.
PoC (nếu làm) theo bao-security: ngoài repo, container `vlm-comfyui` pin bản vá mới nhất, **không cài ComfyUI-Manager**
(CVE-2025-67303, CVE-2026-22777), bind 127.0.0.1, không mount models/ của container comfy-ui sẵn có, giữ GPU_LOCK nếu load model.
Hợp lý nếu mục tiêu ngắn hạn là demo/PoC để user xác nhận cảm giác "kéo thả"; nếu đi sản phẩm thì React Flow tự xây.
Đề xuất kết hợp: dùng **format workflow JSON giống ComfyUI** (nodes/links/widgets_values) ở mức ý tưởng để user quen, và có thể làm
PoC 2–3 ngày trên ComfyUI để user cảm nhận trước khi đầu tư editor riêng — PoC không ghép vào code chính (tránh GPL lẫn vào repo).

## F6b. Ghi chú hiện thực cho UX P010 (bao-design)
- Làm sáng cổng tương thích khi kéo dây: `useConnection()` của React Flow + `isValidConnection`; thả ra khoảng trống: `onConnectEnd`
  → mở search lọc theo kiểu của handle nguồn. Khả thi, chi phí thấp.
- Làn bước ①…⑦: nền vẽ bằng custom background; vị trí x = làn theo `category` của NodeSpec (làn ① = "Nguồn": dữ liệu trên, model gốc dưới), y = thứ tự trong làn → "Sắp xếp gọn"
  không cần dagre cho trường hợp thường (dagre giữ cho graph lệch làn).
- **Tạo dây bằng bàn phím**: React Flow không hỗ trợ sẵn → trong inspector, mỗi cổng vào có dropdown "Nhận từ: [node · cổng ▾]"
  (chỉ liệt kê nguồn tương thích). Rẻ, dùng được cho cả chuột lẫn bàn phím/screen reader; chế độ "Danh sách bước" v1 = outline
  dùng chung component này.
- Thumbnail template sinh từ graph JSON (mini SVG), không cần file PNG.
- Giả định màn hình desktop ≥ 1280px cho editor; < 1280px chỉ xem (read-only).

## F6c. Áp patch từ gợi ý lỗi
Theo P010 (design): patch đổi tham số → áp thẳng + hiện diff + Hoàn tác 1 click; patch tốn kém/không đảo được/đổi cấu trúc graph →
xem trước + xác nhận. Action có field `confirm: bool` (bao-fixcode) quyết định; action phá huỷ luôn `kind: navigate` → dry-run.

## F7b. Phân phối
User sẽ cho người khác dùng miễn phí (PM 2026-10-12) → có phân phối → rủi ro GPL của custom node ComfyUI là thật; phương án 3 chỉ nên dùng cho PoC ngoài repo.

## F8. Câu hỏi mở
1. (PM/user/security) Chấp nhận GPL-3.0 nếu fork ComfyUI? — quyết định này chặn/mở phương án fork.
2. (research R-1) Có app open source MIT/Apache nào đã làm node-graph cho training mà ta tái dùng được (LlamaBoard là Gradio, không phải canvas)?
3. (design R-3) Có cần subgraph/group ở v1 không? (đề xuất: group trực quan có, subgraph để v1.1).
