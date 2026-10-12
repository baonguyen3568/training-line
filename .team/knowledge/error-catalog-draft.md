# Error catalog — bản nháp v0 (bao-fixcode, 2026-10-12)

> Mục đích: biến lỗi kỹ thuật (KB `.team/knowledge/fixcode.md`) thành thông báo **người không biết code đọc hiểu và tự sửa được**.
> Trạng thái: DRAFT. Chờ bao-design chốt format hiển thị (giọng văn, độ dài, nút hành động) và bao-frontend chốt contract.
> **Shape chốt**: catalog không chứa text; exception/Issue/event chỉ mang code + params; render ở rìa. File `error-samples-v0.jsonl` có `title_vi/what_vi/why_vi` chỉ là fixture đã render cho UI, không phải format của events.jsonl.
> **Câu chữ** VI/EN chuyển sang `src/vlm_pipeline/i18n/{vi,en}.yaml` (bao-design sở hữu); catalog chỉ giữ code/severity/match/actions + key copy `error.<CODE>.title|what|why`, `action.<id>` + params (`{est_gb}`). Cột câu chữ trong bảng dưới là nguồn để design chuyển sang i18n.
> **Hệ mã (CHỐT 2026-10-12)**: `code` = `<NHÓM>-<NNN>` (MEM, DISK, GPU, DATA, MODEL, NET, CFG, SEC, CACHE, SYS) — đã được 009/010 dùng. Mỗi mã có `slug` SCREAMING_SNAKE (vd `UMA_OOM`) chỉ để kỹ sư đọc log/test cho dễ; UI, i18n key và contract dùng `code`. Mã đã phát hành không đổi nghĩa/không tái dùng; bỏ thì đánh dấu deprecated.
> Action: `{label_vi, kind: patch|resume|navigate, target, patch, confirm}` — `confirm=true` khi tốn kém / không đảo được / đổi cấu trúc graph (vd resume chạy lại job GPU dài); patch tham số `confirm=false` (áp thẳng + diff + Hoàn tác).
> Contract bản ghi lỗi (đã thống nhất với bao-frontend): `{code, severity, node_id, port?, params, why_vi, actions[{label_vi, kind: patch|resume|navigate, target, patch?:[{node_id,param,value}]}], technical_detail}`. `patch` = áp vào graph sau khi user xác nhận; `resume` = tiếp tục từ checkpoint (có thể kèm patch); `navigate` = mở màn hình/node (`target`), **mọi action phá huỷ (xoá) bắt buộc là navigate tới màn dry-run, không bao giờ là patch**; mã preflight = mã validate của P009.
> Sau khi chốt sẽ chuyển thành `src/vlm_pipeline/common/error_catalog.yaml`. Cột "Bắt trước" = preflight phát hiện được trước khi bấm Run.

| Mã | Mức | Tiêu đề (user thấy) | Chuyện gì xảy ra / vì sao (≤2 câu) | Hành động gợi ý (nút) | Bắt trước | Dấu hiệu kỹ thuật (match) | KB |
|---|---|---|---|---|---|---|---|
| MEM-001 `MEM_INSUFFICIENT` | error | Không đủ bộ nhớ cho bước này | Bước này cần khoảng {need} GB nhưng máy chỉ còn {avail} GB trống (máy đang dùng chung với ứng dụng khác). | "Dùng LoRA thay vì train toàn bộ" · "Giảm kích thước ảnh" · "Giảm batch" · "Chọn model nhỏ hơn" | ✅ | preflight `NO_MEM` | §2.2 |
| MEM-002 `MEM_PREEMPTED` | warn | Đã tạm dừng để bảo vệ máy | Bộ nhớ còn trống xuống dưới mức an toàn nên app đã lưu tiến độ và tạm dừng, tránh làm treo máy. | "Tiếp tục từ chỗ đã lưu" · "Tiếp tục với cấu hình nhẹ hơn" | — | exit `MEM_PREEMPT` (75) | §2.2 |
| MEM-003 `UMA_OOM` | error | Hết bộ nhớ khi đang chạy | Model dùng nhiều bộ nhớ hơn dự kiến ở giữa quá trình (thường do ảnh lớn hoặc câu quá dài). | "Giảm độ dài tối đa" · "Bật tiết kiệm bộ nhớ (gradient checkpointing)" · "Giảm batch" | — | `torch.OutOfMemoryError`, `CUDA out of memory`, `NV_ERR_NO_MEMORY` | §2.2 |
| MEM-004 `SERVE_MEM_TOO_HIGH` | error | Phục vụ model chiếm quá nhiều bộ nhớ | Thiết lập phục vụ model đang giữ quá nhiều bộ nhớ cho bộ đệm. | "Dùng mức an toàn (mặc định)" · "Giảm độ dài hội thoại tối đa" | ✅ | vLLM `gpu_memory_utilization` > 0.3; `No available memory for the cache blocks` | §2.2 |
| DISK-001 `DISK_FULL` | error | Không đủ dung lượng ổ đĩa | Cần khoảng {need} GB để lưu model/checkpoint nhưng chỉ còn {avail} GB. | "Xoá checkpoint cũ (xem trước)" · "Lưu ít checkpoint hơn" | ✅ | preflight `NO_DISK`, `No space left on device` | P005 |
| SYS-001 `GPU_LOCK_BUSY` | info | Máy đang bận chạy tác vụ khác | Chỉ chạy được 1 tác vụ nặng một lúc; {who} đang chạy "{job}" từ {since}. | "Xếp hàng chờ" · "Huỷ" | ✅ | preflight `LOCK_BUSY` | CLAUDE.md GPU lock |
| GPU-001 `GPU_ENV_INCOMPATIBLE` | error | Phần mềm không hợp với GPU | Một thành phần được cài cho phiên bản CUDA cũ hơn GPU này hỗ trợ. | "Dùng môi trường chuẩn của app" · "Gửi báo cáo lỗi" | ✅ (env check) | `libcudart.so.12`, `no kernel image is available`, `+cpu` torch | §2.1 |
| GPU-002 `GPU_COMPILE_FAILED` | error | Không biên dịch được thành phần tăng tốc | Bộ biên dịch GPU không nhận diện được chip GB10 (sm_121). | "Tắt tăng tốc biên dịch (chậm hơn nhưng chạy được)" · "Gửi báo cáo lỗi" | — | `sm_121a is not defined`, `ptxas fatal`, `PermissionError: ''` trong triton | §2.1 |
| GPU-003 `QUANT_SLOW_ON_DEVICE` | warn | Kiểu nén này chạy chậm trên máy | NVFP4 trên GB10 có thể chậm hơn FP8; chất lượng tương đương. | "Chuyển sang FP8" · "Vẫn dùng NVFP4" | ✅ | quantize node `nvfp4` + device cap (12,1) | §2.1, §2.7 |
| DATA-001 `DATA_IMAGE_TOKEN_MISMATCH` | error | Số ảnh trong dữ liệu không khớp | Mẫu dữ liệu #{idx} có {n_img} ảnh nhưng đoạn văn chỉ có {n_tok} vị trí ảnh (hoặc bị cắt mất do quá dài). | "Xem mẫu lỗi" · "Bỏ qua mẫu lỗi" · "Tăng độ dài tối đa" | ✅ (validate data node) | `Image features and image tokens do not match` | §2.5 |
| DATA-002 `DATA_BAD_IMAGE` | warn | Có ảnh bị hỏng hoặc sai định dạng | {n} ảnh không mở được hoặc định dạng lạ (CMYK, RGBA, quá nhỏ). | "Tự chuyển đổi" · "Bỏ qua các ảnh này" · "Xem danh sách" | ✅ | `PIL.UnidentifiedImageError`, `cannot identify image file` | §2.5 |
| DATA-003 `DATA_NO_LEARNING_SIGNAL` | warn | Model có vẻ không học được gì | Loss gần 0 ngay từ đầu, thường do dữ liệu đánh dấu sai phần cần học. | "Kiểm tra dữ liệu mẫu" · "Gửi báo cáo lỗi" | ✅ (label ratio check) | loss < 0.01 ở step đầu; tỉ lệ `labels != -100` = 0 | §2.5 |
| MODEL-001 `CKPT_LOAD_KEY_MISMATCH` | error | Không đọc được model sau khi chuyển đổi | Tên các phần trong file model không khớp với định dạng mong đợi (thường do khác phiên bản thư viện). | "Thử lại với phiên bản khuyến nghị" · "Gửi báo cáo lỗi" | — | `KeyError ... visual`, `Missing key(s)`, `Unexpected key(s)`, `size mismatch` | §2.6 |
| MODEL-002 `MODEL_VERIFY_FAILED` | error | Model sau chuyển đổi cho kết quả sai | Bước kiểm tra tự động thấy model mới trả lời khác hẳn model gốc. | "Xem so sánh" · "Dùng lại model gốc" · "Gửi báo cáo lỗi" | — | node kiểm tra: cosine < 0.999 / key thiếu | §2.6, R6 |
| NET-001 `DOWNLOAD_FAILED` | error | Không tải được model/dữ liệu | Không kết nối được tới Hugging Face/NGC hoặc chưa đăng nhập. | "Kiểm tra kết nối" · "Nhập token (Cài đặt)" | ✅ | `401`, `403 gated repo`, `ConnectionError`, `LocalEntryNotFoundError` | — |
| CFG-001 `GRAPH_PORT_TYPE_MISMATCH` | error | Hai bước này không nối được với nhau | Đầu ra của "{a}" là {type_a} nhưng "{b}" cần {type_b}. | "Thêm bước chuyển đổi phù hợp (xem trước)" · "Xem hướng dẫn" | ✅ | graph type check | P009 |
| DATA-004 `DATA_MISSING_IMAGE` | error | Thiếu ảnh trong dữ liệu | {n} mẫu dữ liệu trỏ tới ảnh không tồn tại. | "Xem danh sách" · "Bỏ qua các mẫu này" | ✅ | `FileNotFoundError` trên đường dẫn ảnh | §2.5 |
| SYS-002 `GPU_LOCK_TIMEOUT` | warn | Chờ máy rảnh quá lâu | Workflow đã chờ {minutes} phút nhưng máy vẫn bận với tác vụ khác. | "Tiếp tục chờ" · "Huỷ" | — | queue wait > timeout | CLAUDE.md GPU lock |
| SEC-003 `LICENSE_BLOCKED` | error | Giấy phép không cho phép dùng thế này | Giấy phép của {name} không cho phép {use}. (bao-security sở hữu nội dung) | "Chọn model/dữ liệu khác" | ✅ | license check | security-license-check |
| CFG-002 `GRAPH_CYCLE` | error | Các bước đang nối thành vòng tròn | Dây nối tạo thành một vòng: {nodes}. Mỗi bước phải chạy sau bước trước, không quay lại được. | "Chỉ ra dây gây vòng" | ✅ | validate: cycle detected | P009 |
| CFG-003 `INPUT_MISSING` | error | Bước này còn thiếu đầu vào | "{node}" cần "{port}" nhưng chưa có dây nối vào. | "Chọn nguồn cho đầu vào này" | ✅ | validate: required port unconnected | P009 |
| CFG-004 `NODE_TYPE_UNKNOWN` | error | Workflow dùng bước app không có | Bước "{node_type}" không có trong app hoặc không được phép dùng. App không tự cài thêm. | "Xoá bước này" · "Chọn bước thay thế" | ✅ | validate: unknown/disallowed node type | P009 |
| CFG-005 `PARAM_INVALID` | error | Một thiết lập không hợp lệ | "{param}" của "{node}" là {value}, nhưng phải {rule}. | "Đặt về mặc định" | ✅ | validate: pydantic ValidationError on params | P009 |
| CFG-006 `EDGE_DANGLING` | error | Có dây nối tới bước không tồn tại | Một dây trỏ tới "{node_id}" nhưng bước này đã bị xoá hoặc đổi tên. | "Xoá dây lỗi" | ✅ | validate: edge endpoint missing | P009 |
| CFG-007 `WORKFLOW_FILE_INVALID` | error | Không mở được file workflow | File không đúng định dạng workflow hoặc có thông tin lạ ({field}). | "Mở file khác" · "Xem chi tiết kỹ thuật" | ✅ | parse/schema error, extra="forbid" | P009 |
| SEC-001 `PATH_OUTSIDE_WORKSPACE` | error | Đường dẫn nằm ngoài thư mục làm việc | "{path}" nằm ngoài thư mục dự án nên app không được đọc/ghi ở đó. | "Chọn thư mục trong dự án" | ✅ | resolved path not under workspace root | P003 |
| SEC-002 `SECRET_IN_WORKFLOW` | error | Workflow đang chứa mật khẩu hoặc token | Thiết lập "{param}" có vẻ chứa token. Hãy lưu token trong Cài đặt, workflow chỉ tham chiếu tới nó. | "Chuyển sang Cài đặt bảo mật" | ✅ | secret regex on params (không lưu giá trị) | P003 |
| CACHE-001 `CACHE_ARTIFACT_INVALID` | warn | Kết quả cũ bị hỏng, đang chạy lại | Kết quả đã lưu của "{node}" bị thiếu hoặc không khớp, nên app chạy lại bước này. | "Xem chi tiết kỹ thuật" | — | executor: cache hit nhưng path mất / manifest hash lệch → miss (nội bộ executor, P009) | P009 |
| SYS-003 `WORKER_CRASHED` | error | Bước này dừng đột ngột | Tiến trình chạy "{node}" bị dừng bất ngờ (mã thoát {exit_code}) mà không báo lý do. | "Chạy lại" · "Gửi báo cáo lỗi" | — | worker exit ≠ 0/75 và không có event error. **Phân loại trước khi gán SYS-003**: exit 137/SIGKILL + `memory.events` oom_kill tăng hoặc `docker inspect .State.OOMKilled=true` → **MEM-003**; exit 75 → MEM-002; còn lại (139 SIGSEGV, 134 SIGABRT, …) → SYS-003 kèm `signal` | KB §2.2 |
| SYS-000 `UNKNOWN_ERROR` | error | Đã xảy ra lỗi chưa được nhận diện | App chưa biết lỗi này. Tiến độ đã được lưu nếu có. Bạn có thể gửi báo cáo để team sửa. | "Gửi báo cáo lỗi" · "Xem chi tiết kỹ thuật" | — | fallback | — |

## Câu hỏi cho bao-design
1. Giọng văn: "bạn" hay trung tính? Có nên hiện con số GB không, hay quy thành "cần thêm ~X%"?
2. Vị trí: badge đỏ trên node + panel bên phải? Toast cho MEM-002?
3. Nút hành động có auto-fix (vd "Dùng LoRA") → áp thẳng lên graph rồi cho user xem diff, hay chỉ mở node cần sửa?
4. "Chi tiết kỹ thuật" thu gọn: hiện stack trace + mã lỗi + nút copy?

## Quyết định hiển thị — bao-design (2026-10-12), chi tiết UX ở `.team/proposals/010-nocode-ux.md` §6–§7
1. **Giọng văn**: xưng "bạn", câu chủ động, không đổ lỗi ("Máy không đủ bộ nhớ", không phải "Bạn chọn cấu hình sai"). Tiêu đề ≤ 8 từ, nói *hệ quả*
   chứ không nói cơ chế. Giải thích ≤ 2 câu. **Hiện số GB tuyệt đối** (cần {need} / còn {avail}) — người dùng so được với thanh RAM trên header;
   không quy ra %. Mỗi nút hành động nên kèm hiệu quả nếu ước lượng được: "Dùng QLoRA (~22 GB)". Thuật ngữ kỹ thuật chỉ trong ngoặc, sau từ thường:
   "tiết kiệm bộ nhớ (gradient checkpointing)" ✓.
2. **Vị trí**:
   - Bắt trước (preflight): badge trên node (viền + icon + chữ "Lỗi"/"Cảnh báo") + dòng trong **thanh kiểm tra** đáy canvas; click → focus node + panel phải.
   - Lúc chạy: node chuyển trạng thái ✕ Lỗi + **thẻ lỗi 4 phần trong panel phải** (Chuyện gì / Vì sao / Cách sửa [nút] / ▸ Chi tiết kỹ thuật). Không modal, không chuỗi dialog.
   - MEM-002 (tạm dừng bảo vệ máy): **không dùng toast** (biến mất, dễ bỏ lỡ) → trạng thái node "⏸ Tạm dừng" + banner bền trên canvas đến khi người dùng xử lý.
   - Toast chỉ cho phản hồi tức thời không chặn: CFG-001 khi thả dây sai.
3. **Auto-fix**: áp thẳng lên graph khi chỉ đổi tham số (patch) → node nhấp nháy, panel hiện diff "batch 8 → 4", nút **Hoàn tác** 1 click, rồi chạy lại preflight.
   Ngoại lệ phải xác nhận trước: hành động tốn kém/không đảo được (tải lại model, xoá, chạy lại từ đầu), và thay đổi cấu trúc graph (thêm node chuyển đổi)
   → hiện node "bóng mờ" xem trước rồi [Áp dụng]/[Huỷ]. Nút không auto-fix được → "Mở node" (focus + mở đúng tham số).
4. **Chi tiết kỹ thuật** (gập mặc định; Advanced mode nhớ trạng thái mở): mã lỗi, node/run_id, phiên bản engine/container, 30 dòng log cuối
   (đã lọc secret/đường dẫn $HOME theo P003), lệnh CLI tái hiện, nút **"Sao chép báo cáo lỗi"** (markdown gom tất cả). Không đổ full stack trace lên UI —
   có link "mở file log đầy đủ".
- Vị trí file (CẬP NHẬT sau góp ý frontend): `error_catalog.yaml` giữ mã/match/actions(kind, patch, confirm) + **key copy**; câu chữ VI/EN ở `src/vlm_pipeline/i18n/{vi,en}.yaml` (design sở hữu, dùng chung cho node + lỗi + hậu quả);
  docs (`docs/errors/`) sinh tự động từ file đó; test fault-injection của bao-tester đọc cùng file. (Huỷ ý `design/copy/errors.*.yaml` trong P010.)
- (đã áp vào bảng) Sửa câu chữ nhỏ: CFG-001 nút "Thêm bước chuyển đổi" → "Thêm bước chuyển đổi phù hợp (xem trước)"; SYS-000 thêm câu "Bạn có thể gửi báo cáo để team sửa."
