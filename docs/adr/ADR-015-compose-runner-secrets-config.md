# ADR-015 — Docker compose, job runner, secrets & cấu hình

- Trạng thái: **ACCEPTED** — 2026-10-13
- Nguồn: `.team/proposals/015-compose-job-runner-secrets.md` (bao-backend + bao-security), §6.1 bảo vệ token, §7.1–7.2.
- Chốt bởi: bao-pm theo yêu cầu user (M3: Docker toàn bộ, compose, 1 file `.env.local`, amd64+arm64; D-7: cấu hình env/DB + trang Cài đặt).

## Quyết định
- **Tên app (hằng số duy nhất):** `ntc-training`, theo tên sản phẩm user chốt "NTC-Training". Policy admin: `~/.config/ntc-training/policy.toml` (thay `~/.config/vlm-app/`).
- **Compose:** app (API + UI dist), site tĩnh, `vlm-runner` sidecar, vLLM khi cần. Mọi cổng bind 127.0.0.1. Build buildx `linux/amd64,linux/arm64`.
- **Job runner:** sidecar `vlm-runner` có API kiểu hoá qua unix socket, chỉ create/stop/inspect/logs/remove trên container có label dự án, không exec/pull/build. App KHÔNG có docker.sock, docker CLI hay token. Có một interface `JobLauncher` (local cho dev, runner cho compose) dùng chung `docker_args`.
- **Mount job thu hẹp:** run dir (rw), cache và data (ro), code (ro). Job không bao giờ thấy `.env*` hay `~/.config/ntc-training`.
- **`.env.local`:** ĐẶT Ở GỐC DỰ ÁN như user yêu cầu (gitignored, quyền 0600), là nguồn cấu hình hạ tầng duy nhất (port, data root, profile, khoá mã hoá…). `.env.local.example` được commit. Secret trong đó đưa vào container qua compose `secrets:`, chỉ cho runner/job download. `.env` cũ được migrate rồi bỏ.
- **Cấu hình user (D-7):** lưu trong DB app, chỉnh qua trang Cài đặt. Secret mã hoá AEAD, chỉ ghi (trả last4), có audit log. `admin_policy` chỉ đọc qua API.
- **Data root:** mặc định trong thư mục dự án (`.cache/`, `outputs/`, `data/`), đổi được qua `.env.local`. Quy tắc an toàn của team áp dụng cho repo + data root đã cấu hình + `~/.config/ntc-training`.
- Bắt buộc 6 nhóm gate + 5 bảo vệ token có test hồi quy (P015 §6.1).

## Điều kiện bắt buộc vì `.env.local` nằm trong repo (bao-security, chấp nhận)
1. Mỗi Dockerfile có `.dockerignore` chứa `.env*` + `**/.env*`. check.sh kiểm build context không chứa file env. Security kiểm `docker history`/layer sau mỗi build.
2. Không service compose nào bind gốc repo (kể cả compose dev); chỉ bind thư mục con. Có test parse `docker compose config`.
3. Git: `.gitignore` `.env.*` (đã có) + gitleaks + scan_tokens tracked.
4. Người/agent: không đọc/in `.env*`. README cảnh báo không zip/chia sẻ thư mục dự án khi còn `.env.local`.
- Amendment (2026-10-13): HF token có quyền write → CẢNH BÁO (không từ chối), vì user chấp nhận giữ token hiện tại (U3). UI cảnh báo 1 lần lúc lưu + link hướng dẫn token Read; audit `warning:write_scope`; 5 bảo vệ bù (P015 §6.1) bắt buộc. Khoá AEAD: ~/.config/ntc-training/secrets.key.
