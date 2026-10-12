# ADR-005 — Storage & Retention Policy

- Trạng thái: **ACCEPTED (có điều kiện)** — 2026-10-12
- Nguồn: `.team/proposals/005-storage-retention-policy.md` (v3, tác giả bao-cleanup)
- Đã review: bao-backend, bao-security, bao-research, bao-tester, bao-frontend, bao-cleancode
- Chốt bởi: bao-pm

## Quyết định
Chấp nhận toàn bộ D1–D8 của Proposal 005 v3. Tóm tắt:
- Cache dự án ở `.cache/` trong project root. `~/.cache/huggingface` toàn cục chỉ đọc.
- Layout `outputs/runs/<run_id>/` theo D2; file contract của dashboard nằm trong whitelist không bao giờ xoá.
- Checkpoint: giữ 2 bản gần nhất + bản consolidated cuối. Chỉ strip optim khi `status=completed`.
- Docker: label `project=vlm-training`, prefix `vlm-`, builder riêng `vlm-builder`. Image NGC chỉ xoá theo quy trình 2 bước trong D7.10.
- Quota warn 500GB / critical 800GB (user xác nhận 2026-10-12).
- Pre-flight D8 là bắt buộc trước khi lấy GPU_LOCK. Không `drop_caches`/sudo; khi từ chối chạy thì báo PM.
- Ngưỡng bộ nhớ (dùng chung với Proposal 007, 1 code path `common/memguard.py`):
  - Preflight: `MemAvailable ≥ est_peak + max(10GB, 20%×est_peak)`, sàn cứng 20GB, `SwapFree ≥ 2GB`.
  - Runtime watchdog: warn khi `MemAvailable < 14GB`, abort có kiểm soát khi `< 8GB`. Fixcode đo dao động RAM 24-48h rồi hiệu chỉnh theo dữ liệu.
  - Phân công: bao-fixcode viết `memguard.py`, bao-cleanup viết phần disk/lock + `scripts/cleanup/preflight.py`, review chéo.

## Điều kiện / mục còn mở
1. bao-backend xác minh ngữ nghĩa `checkpoint.save_consolidated` trong container được pin ở ADR-001. Kết quả cập nhật vào bảng D3.
2. ~~Quota 500GB kiểm tra lại theo danh sách model~~ → XONG (P005 §7): ~430GB tối đa kể cả cache node 200GB, nằm dưới mức warn 500GB. Mỗi model 27-35B thêm 55-72GB.

## Hệ quả
- Phase 2: bao-cleanup viết `scripts/cleanup/{disk_report,retention,preflight}.py`, bao-tester viết test, bao-cleancode + bao-security review.
- Mọi job nặng phải khai báo `est_peak_gb` trong `run.json` (mặc định 40GB).
