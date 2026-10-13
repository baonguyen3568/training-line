# ADR-019 — Mở app ra LAN qua nginx

- Trạng thái: **ACCEPTED** — 2026-10-13 (user chọn mở LAN; PM chốt thiết kế của security)
- Nguồn: `.team/proposals/019-lan-exposure-nginx.md` (bao-security, 5dbcaa6). Sửa đổi P003 D13 + ADR-015.

## Quyết định
- nginx là cổng vào DUY NHẤT từ LAN: bind **192.168.2.178:4000** (không 0.0.0.0), `allow 192.168.2.0/24; deny all`. Chạy trên compose bridge (không network host). Image pin digest, non-root, read-only, cap_drop ALL.
- Server của dự án ở dải **127.0.0.1:4001–4050** (4001 app compose, 4002 site, 4003 app dev, 4010–4019 dev server). Ở 1.0, dev server không được proxy.
- **TLS bắt buộc**: CA nội bộ có Name Constraints (chỉ IP/hostname máy này). HTTP vào port 4000 → 301. HSTS chỉ khi Host là tên.
- Host/Origin allowlist đúng các origin LAN + localhost; default_server trả 444. Cookie `__Host-` Secure/HttpOnly/SameSite=Strict, session idle 12h / tối đa 7 ngày, có "đăng xuất mọi thiết bị".
- **Đăng nhập đổi**: link dạng `#t=…` (fragment), trang nhỏ POST vào /api/session. `?token=` chỉ còn nhận từ 127.0.0.1. Rate-limit 5/phút/IP, khoá 15 phút sau 20 lần sai.
- Header bảo mật; body 2MB (21MB cho /api/uploads, không buffer); WebSocket; access log dùng `$uri` (không query, không cookie).

## Rủi ro còn lại (user đã được báo)
1. Ai có phiên là điều khiển toàn bộ dự án; link lộ thì mất quyền cho tới khi xoay token. 1.0 chưa có tài khoản riêng/audit theo người (→ roadmap).
2. User phải cài CA nội bộ; bấm bỏ qua cảnh báo cert có thể bị MITM.
3. Docker bỏ qua UFW. Tuỳ chọn mạnh hơn là luật DOCKER-USER iptables (cần sudo, user quyết).
4. Bug của app giờ khai thác được từ LAN.
5. Lạm dụng GPU/disk chỉ bị chặn bởi lock/quota, không theo danh tính.
