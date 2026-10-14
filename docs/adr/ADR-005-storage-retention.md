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
- **SỬA LẠI theo user (2026-10-13, thay cho cách hiểu "tổng cả máy" ở trên):** "đừng quá 80% phần VRAM CÒN LẠI".
  - Ngân sách job = `0.80 × MemAvailable` đo lúc preflight (GPU rời: 0.80 × VRAM trống). Chừa 20% phần còn trống làm đệm cho máy dùng chung.
  - Preflight: `est_peak ≤ max_free_frac × MemAvailable_now` (mặc định 0.80, chỉ chỉnh chặt hơn), cộng thêm SwapFree ≥ 2GB. Thiếu ngân sách → xếp hàng chờ (SYS-002 nếu quá lâu).
  - Runtime: gọi A0 = MemAvailable lúc job bắt đầu. Warn khi MemAvailable < 0.25 × A0; abort có kiểm soát khi MemAvailable < 0.20 × A0 (đã tiêu > 80% phần còn lại), hoặc khi fast-fall dự báo chạm mức đó trong 5s.
  - Sàn tuyệt đối vẫn giữ (cái nào chặt hơn thắng): abort khi < 8GB, emergency kill khi < 4GB.
  - Triển khai (34d4f7e): preflight `est_peak ≤ 0.8 × MemAvailable` VÀ `est_peak + 8GB ≤ MemAvailable`; cgroup là ngân sách thứ 2. Margin 10GB và sàn 20GB cũ đã BỎ. Runtime: warn < max(14GB, 0.25·A0), stop < max(8GB, 0.2·A0), emergency 4GB.
- Amendment A2 (2026-10-13): run `interrupted` quá 7 ngày (từ ended_at) mà không ai resume → xử lý như `failed`: ckpt thường xoá tự động; LATEST = PERMANENT_LOSS, chỉ xoá tay. UI nhắc trước hạn 1 ngày.

## Amendment A3 (2026-10-14, PM; đồng ý: bao-fixcode, bao-security) — Miễn sàn swap cho job nhỏ có cgroup cứng
- Job **không dùng GPU**, có cgroup memory cứng **≤ 4 GB** (`SWAP_EXEMPT_MAX_GB`) và **không có swap** (`--memory-swap = --memory`, `--pids-limit 512`) được bỏ qua sàn `SwapFree ≥ 2 GB`. MemAvailable / ngân sách 80% phần còn trống / abort_gb giữ nguyên.
- Điều kiện ở 1 chỗ: `jobrun.swap_exempt(job)`; argv docker và ngưỡng đều suy từ cùng `ContainerJob`.
- Lý do: host dùng chung, SwapFree dao động 1–4 GB do service khác, khiến job tải chờ mãi. Job miễn mà vượt cap thì bị OOM-kill trong cgroup riêng, không kéo host xuống.
- Job train/eval/quantize/serve giữ nguyên sàn swap. Khi chờ swap, UI/CLI báo `waiting_for: swap` (SYS-002 riêng), không ghi "memory".
- Test: `test_swap_exempt_jobs_get_no_swap_and_a_pids_limit`, `test_only_the_swap_floor_is_dropped_for_an_exempt_job`, `test_the_runner_gives_small_cpu_jobs_no_swap`.
