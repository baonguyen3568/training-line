# 017 — Một job mỗi run dir: attempt dir cho retry/resume, mount thu hẹp (M3-W)

- Tác giả: bao-backend · 2026-10-13 · Trạng thái: PM đồng ý hướng đi 2026-10-13 (Q3 chốt); Q1 chờ security, Q2 chờ cleanup
- Liên quan: ADR-015, P015 §3.2, `common/run_claims.py` (7f18102), `common/host_paths.py`, `runner/protocol.py`

## 1. Vấn đề
Security yêu cầu (M3): **không bao giờ cấp quyền ghi (rw) cho một thư mục mà một job trước đó đã từng ghi**. Lý do:
job trước có thể đã cài sẵn symlink hoặc file độc vào đó. Cơ chế đã có là `claim_run(run_id)` (O_EXCL, nằm ngoài
run dir, không bao giờ xoá); chưa nối vào `run_job`.

Hiện có 3 chỗ dùng lại run dir, nên nếu nối claim ngay thì chúng hỏng:

| Chỗ | Hành vi hiện tại |
|---|---|
| Graph node train / merge / quantize / trial (`executor.py:316`) | run dir `<nid>-<key16>` cố định theo cache key. Retry/resume chạy lại TRONG cùng thư mục, AutoModel tự resume từ ckpt trong đó (`resume_from_step`, CACHE-001) |
| bench (`bench_worker`) | 1 run dir node, nhiều job (`<dir>-<scenario>`): run_id ≠ tên run dir |
| eval (`eval/node_worker`) | job ghi vào `outputs/eval-cache/<key>` (key theo nội dung: base + ảnh + decoding + worker + image id), cố ý dùng lại giữa các run. Lần trước fail thì lần sau ghi lại vào CÙNG thư mục |

CLI (`train/smoke.py`, `jobrun_cli`) đã đúng: run_id mới mỗi lần, resume = `--restore-from` (ckpt của run cũ).

## 2. Đề xuất
1. **Attempt dir**: graph node dùng `<nid>-<key16>` cho lần đầu, `<nid>-<key16>-a2`, `-a3`… cho mỗi lần chạy lại
   (RUN_ID regex thêm `(-a[0-9]{1,3})?`). Executor chọn attempt kế tiếp chưa claim.
2. **Resume = input ro**: lần chạy lại của node train tìm ckpt mới nhất ở attempt trước. Ckpt đó được mount **ro** và đưa vào
   recipe qua `checkpoint.restore_from` (giống `smoke.py --restore-from`). Run dir mới chỉ chứa output của lần này.
   UI vẫn hiện "tiếp tục từ step N" như hiện nay. `resume_from_step` lấy từ attempt trước.
3. **Eval cache**: job ghi vào attempt dir của chính nó; chỉ khi COMPLETED thì host đổi tên (rename) sang
   `eval-cache/<key>`. Thư mục cache chỉ đọc sau đó; cache hit dùng ro, không bao giờ cho job ghi lại.
4. **Bench**: mỗi scenario một run dir con riêng `<node dir>/<scenario>` với run_id = tên dir đó; hoặc đổi luật protocol
   thành `run_dir.name == run_id` OR `run_dir.parent.name` là node dir của request. Mình chọn cách đầu (đơn giản, khớp luật).
5. **Nối claim** trong `run_job`, trước khi gọi launcher: `check_mount_source(run_dir)` → `claim_run` → mount.
   `LocalDockerLauncher` và runner cùng một đường.
6. **Mount thu hẹp** (P015 §3.2): job chỉ thấy run dir của mình (rw); `src/` (ro); input cụ thể (ro: ckpt attempt
   trước, dataset, model trong cache, upload); `.cache/hf` (ro, trừ thư mục modules nếu engine cần ghi → tmpfs).
   Bỏ mount cả project root.

## 3. Ảnh hưởng tới các bên
- **Cleanup/retention**: một node có thể có nhiều attempt dir. Retention coi mỗi attempt là một run. Attempt cũ có ckpt
  đang làm input của attempt mới phải được giữ khi attempt mới còn chạy; lease index đã có cơ chế này, cần xác nhận.
- **Frontend**: lịch sử node hiện 1 run dir; sẽ có nhiều attempt. API runs trả attempt mới nhất và danh sách attempt.
- **Tester**: E2E kiểm retry tạo `-a2` và resume đúng step.
- **Eval / bench / quantize worker**: run dir con theo luật mới.

## 4. Câu hỏi mở
- Q1 (security): mount ckpt attempt trước ro vào job mới. Job mới đọc file do job cũ viết, nhưng ckpt là safetensors +
  JSON, engine đọc như dữ liệu. Chấp nhận được không?
- Q2 (cleanup): giữ attempt cũ bao lâu khi attempt mới đã completed?
- Q3 (PM) — CHỐT: tối đa 10 attempt mỗi node; lần thứ 11 dừng, báo lỗi rõ kèm gợi ý "đặt lại bước này" (xoá các attempt qua panel Dung lượng).

## 5. Kế hoạch
Code sau khi proposal được duyệt. Thứ tự: (1) attempt dir + claim trong run_job (dev launcher) → (2) bench/eval run dir
con → (3) resume qua input ro → (4) mount thu hẹp → (5) runner server. Mỗi bước có security review riêng.
