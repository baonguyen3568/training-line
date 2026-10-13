# ADR-005 — Storage & Retention Policy

- Trạng thái: **ACCEPTED** — 2026-10-12 (2 điều kiện đã đóng)
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
  - Ngoại lệ khi đang save checkpoint: memguard gia hạn tới 300s (tránh hỏng checkpoint), trừ khi `MemAvailable < 4GB` thì abort ngay.
  - Phân công: bao-fixcode viết `memguard.py`, bao-cleanup viết phần disk/lock + `scripts/cleanup/preflight.py`, review chéo.

## Điều kiện / mục còn mở
1. ~~Xác minh save_consolidated~~ → XONG (S-3, 26.08): `final` được hỗ trợ và là mặc định (`true` = every); max_recent_checkpoints=2 chạy đúng; checkpoint dở có marker `.incomplete`.
2. ~~Quota 500GB kiểm tra lại theo danh sách model~~ → XONG (P005 §7): ~430GB tối đa kể cả cache node 200GB, nằm dưới mức warn 500GB. Mỗi model 27-35B thêm 55-72GB.

## Hệ quả
- Phase 2: bao-cleanup viết `scripts/cleanup/{disk_report,retention,preflight}.py`, bao-tester viết test, bao-cleancode + bao-security review.
- Mọi job nặng phải khai báo `est_peak_gb` trong `run.json` (mặc định 40GB).

## Amendment A1 (2026-10-13, USER) — Trần bộ nhớ 80%
User: "cứ chơi đi, không quá 80% VRAM". Team dùng GPU tự do (bỏ cơ chế giữ GPU cho user), nhưng:
- **TỔNG bộ nhớ đang dùng trên thiết bị ≤ 80%.** UMA (GB10): `MemTotal − MemAvailable ≤ 0.8 × MemTotal`, tức MemAvailable ≥ ~25.5GB trên máy này. GPU rời: VRAM used ≤ 80%.
- Preflight: `used_hiện_tại + est_peak ≤ 80%`. Runtime: warn ở 75%, abort khi vượt 80% (fast-fall dự báo chạm 80%).
- Đây là ngưỡng CHẶT HƠN ngưỡng 8GB cũ, và thay thế ngưỡng đó. Ngưỡng là setting: mặc định 80, chỉ được chỉnh chặt hơn. Owner: bao-fixcode (memguard).
- **Xác nhận user (2026-10-13):** 80% là TỔNG bộ nhớ của cả máy, tính cả service khác. Chi tiết triển khai (bao-fixcode):
  - `max_used_frac` mặc định 0.80, chỉ chỉnh chặt hơn (lỏng hơn → CFG-005). GB10: dùng = MemTotal − MemAvailable. GPU rời: VRAM (D-4).
  - Preflight: `MemAvailable ≥ max(est_peak + max(10GB, 20%×est_peak), 20GB, est_peak + (1−max_used_frac)×MemTotal)`; SwapFree ≥ 2GB. Từ chối → MEM-001.
  - Runtime: warn > 75% (~30.4GB available), abort > 80% (~24.3GB), grace 60s / 300s khi đang save; fast-fall dự báo chạm 80% trong 5s → kill ngay; emergency < 4GB kill ngay. run.json ghi `memguard_abort_gb`.
  - Job không đủ ngân sách thì **xếp hàng chờ** RAM (không fail ngay); quá lâu → SYS-002 cho user biết.
