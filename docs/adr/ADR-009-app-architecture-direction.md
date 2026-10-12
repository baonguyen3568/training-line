# ADR-009 — Hướng kiến trúc app node-graph

- Trạng thái: **ACCEPTED (hướng đi)**; chi tiết thiết kế tiếp tục trong Proposal 009 / 010 — 2026-10-12
- Nguồn: R-1 (bao-research), P009 (bao-backend + bao-frontend), P010 (bao-design), P003 (bao-security)
- Đồng thuận: research, backend, frontend, design, security; cleancode (chính sách bên thứ 3). Chốt bởi bao-pm + user.

## Quyết định
- **Tự xây, ghép khối có sẵn, không fork ComfyUI.** Lý do kỹ thuật: executor của ComfyUI chạy in-process, 1 prompt/lần, không resume, và giả định VRAM rời. Lý do license xem ADR-003.
- **Editor**: React Flow (@xyflow/react 12, MIT). Node làm bằng DOM. LiteGraph bị loại (đã deprecated).
- **Mô hình node / giao thức**: học theo ComfyUI (typed schema, `/prompt` + WS events, tách file template và plan, cache theo fingerprint) nhưng viết lại.
- **Executor**: FastAPI + SQLite + docker SDK. Mỗi node nặng = 1 job container `vlm-*`. Output là artifact handle. Cache content-addressed trên disk (ADR-005). Prefect 3 là phương án nâng cấp.
- **Engine (plugin, nhiều backend)**: NeMo AutoModel (chính), LLaMA-Factory (Apache-2.0, backend thứ 2), ModelOpt, vLLM ≥0.31.0, lmms-eval.
- **UX** (user chốt): mặc định **tiếng Việt** (có EN), chế độ **Simple**. Màn đầu là template gallery. Node "to" (1 node = 1 việc). Dataflow + số bước.
- **Custom node bên thứ 3: CHO PHÉP NGAY TỪ v1** (user chốt, khác khuyến nghị của team). Bắt buộc có cơ chế an toàn: xem mục dưới.

## Điều kiện cho custom node v1 (bao-security + bao-backend thiết kế, PM duyệt)
- Custom node chạy **trong container riêng** (không trong process server). Mặc định network none, chỉ mount artifact input/output.
- Manifest khai báo quyền (network, GPU, đường dẫn). Khi cài, UI hiện cảnh báo rõ bằng lời thường + nguồn + license.
- Pin version/hash khi cài. Không auto-update. Có nút gỡ.
- Node bên thứ 3 có license riêng; app không phân phối kèm.

## Còn mở
- Hệ kiểu pin (R-1 và P009 còn lệch nhẹ): backend + research thống nhất.
- Mã lỗi thống nhất (design + fixcode).

## Amendment A1 (2026-10-12) — Custom node sandbox v1 (P003 D16, P004 §8.5, P009 §2.4)
Đồng thuận: security, backend, frontend, design, tester, fixcode, cleancode.
- Manifest `vlm-node.toml` thuần data. Server **không import** code plugin. Không nạp JS/CSS bên thứ 3 vào app.
- Plugin chạy trong container sandbox: v1 **network none**, không mount tuỳ ý, không secret. `trust_level` = first_party | reviewed | unverified, do server tính.
- Port type tập đóng; plugin chỉ khai subtype của type core. `plugin_api` có semver.
- Cài đặt: luôn qua dialog (P010 §15). Quyền cao / GPU + nguồn chưa xác minh → phải gõ tên node. Workflow chia sẻ KHÔNG mang URL/commit để tự cài; không có cài hàng loạt.
- Output của plugin được kiểm bằng allowlist parse định dạng. Plugin hỏng → `unavailable`, app vẫn khởi động.
- **Chính sách riêng cho host dev dùng chung**: chờ user chốt (xem DECISIONS.md).
