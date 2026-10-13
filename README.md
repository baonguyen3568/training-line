# NTC-Training

Ứng dụng huấn luyện Vision-Language Model (VLM) trên GPU NVIDIA bằng giao diện kéo thả: lấy và xử lý dữ liệu →
fine-tune → chạy thử → nén → đánh giá → xuất model, không cần viết code. Engine bên dưới là stack NVIDIA
(NeMo AutoModel, TensorRT Model Optimizer, vLLM), bắt đầu từ DGX Spark (GB10, aarch64).

## Cần có

- Linux có GPU NVIDIA, driver đã cài, **NVIDIA Container Toolkit** (runtime `nvidia` của Docker).
- Docker Engine và Docker Compose v2 (`docker compose version`).
- Tài khoản dùng app nằm trong nhóm `docker`. Không chạy bằng `root` hay `sudo`.
- Dung lượng trống: vài chục GB cho image và model; nhiều hơn nếu train model lớn.

## Cài đặt (một lệnh)

```bash
git clone <repo> ntc-training && cd ntc-training
scripts/install.sh            # kiểm tra máy, tạo thư mục + khoá, build 2 image, khởi động
scripts/install.sh --dry-run  # chỉ in ra sẽ làm gì, không thay đổi gì
```

Lệnh cài:

- tạo `.env.local` từ `.env.local.example` nếu chưa có (cổng `VLM_PORT`, thư mục dữ liệu `VLM_DATA_ROOT`);
- tạo `outputs/ .cache/ data/ uploads/` trong thư mục dữ liệu;
- tạo `~/.config/ntc-training/secrets.key` (khoá mã hoá token, 0600) và `policy.toml` (chính sách quản trị, có thể để trống);
- ghi `.env.compose` (chỉ các giá trị tính được, không chứa mật khẩu), build và chạy `docker compose up -d`.

Chạy lại lệnh cài bất cứ lúc nào để cập nhật: file đã có sẽ không bị ghi đè.

Nâng cấp từ bản cài trước ngày 2026-10-13: hai mạng nội bộ của app (`edge` cho nginx, `egress` cho job tải về) giờ có
dải địa chỉ cố định. Nếu `docker compose up` báo mạng `ntc-training_egress` hoặc `ntc-training_edge` "sai cấu hình",
hãy chạy `docker compose --env-file .env.compose down` một lần (chỉ dừng app, không xoá dữ liệu) rồi chạy lại lệnh cài.

### Token Hugging Face (chuyển từ bản cũ)

Token nhập trong app: **Cài đặt → Token**. Token được mã hoá bằng `secrets.key`; app không đọc token từ file `.env`.
Nếu trước đây bạn để `HF_TOKEN=` trong file `.env` của thư mục dự án, hãy dán token đó vào Cài đặt một lần, rồi xoá
dòng `HF_TOKEN` khỏi `.env` để token không còn nằm dạng chữ thường trên đĩa.

## Mở app

Sau khi cài, lấy đường link có mã đăng nhập một lần:

```bash
docker compose --env-file .env.compose logs app | grep 'open http'
```

Link có dạng `http://127.0.0.1:4001/#t=…`. Mã nằm sau dấu `#` nên trình duyệt không gửi nó lên máy chủ; trang đăng
nhập đọc mã rồi đổi thành phiên đăng nhập. Mỗi lần app khởi động lại sẽ có mã mới. Nút **Cài đặt → Đăng xuất mọi
thiết bị** cũng đổi mã ngay.

Mặc định app chỉ nghe trên `127.0.0.1`.

### Mở cho máy khác trong mạng LAN (tuỳ chọn, ADR-019)

1. Trong `.env.local`, đặt `VLM_LAN_BIND=<IP LAN của máy này>` (ví dụ `192.168.2.178`), rồi chạy lại `scripts/install.sh`.
   Lệnh cài sẽ:
   - tạo một CA nội bộ có Name Constraints, chỉ ký được cho IP/tên của máy này;
   - tạo chứng chỉ máy chủ (gia hạn tự động khi chạy lại lệnh cài);
   - chạy nginx trên `https://<IP>:4000`. Chỉ các máy trong `VLM_LAN_ALLOW` (mặc định: cùng dải /24) vào được.
2. Cài file `~/.config/ntc-training/tls/ca.crt` vào từng thiết bị sẽ mở app (mục "chứng chỉ tin cậy" của hệ điều hành
   hoặc trình duyệt). Đừng bấm "bỏ qua cảnh báo chứng chỉ": làm vậy là mất lớp bảo vệ chống nghe lén.
3. Mở link `https://<IP>:4000/#t=…` (cũng lấy bằng lệnh `logs app` ở trên).

Rủi ro còn lại, cần biết trước khi bật:

- Ai có link hoặc phiên đăng nhập là điều khiển được toàn bộ dự án: job, dữ liệu, export. Chưa có tài khoản riêng
  cho từng người. Chỉ gửi link cho người tin cậy. Lộ link thì vào **Đăng xuất mọi thiết bị**.
- Link đăng nhập in trong `docker logs` của app. Ai có quyền docker trên máy này đọc được cho tới lần đổi mã tiếp theo.
- Docker bỏ qua tường lửa UFW: chỉ nginx (bind đúng IP LAN, kèm danh sách dải mạng được phép) chặn truy cập.
- Lỗi của app giờ có thể bị khai thác từ trong mạng LAN.

## Dừng, khởi động lại

```bash
docker compose --env-file .env.compose stop     # dừng
docker compose --env-file .env.compose up -d    # chạy lại
```

Dữ liệu, model đã tải và kết quả nằm trong thư mục dữ liệu, không mất khi dừng hay cập nhật app.

## An toàn

- Chỉ dịch vụ `runner` được dùng Docker socket. App và các job huấn luyện không có quyền đó, không chạy bằng root,
  và job chỉ thấy đúng những thư mục nó cần (đọc-chỉ cho dữ liệu đầu vào).
- Token (Hugging Face…) nhập trong **Cài đặt** của app, được mã hoá bằng `secrets.key`; không bao giờ ghi vào log
  hay workflow.
- Mỗi lúc chỉ một job nặng dùng GPU (khoá GPU dùng chung với các script trên máy).
- Không nén hay chia sẻ cả thư mục dự án khi đang có `.env.local`.
- Máy dùng chung: quản trị viên giữ `~/.config/ntc-training/policy.toml` của từng người dùng, thuộc root và 0644
  (`sudo chown root:root …/policy.toml && sudo chmod 644 …/policy.toml`). App chỉ đọc file này, nên người dùng không
  tự nới chính sách được.

## Cho người phát triển

```bash
uv sync                      # môi trường Python 3.12
bash scripts/check.sh        # ruff + mypy + pytest + kiểm tra frontend
bash scripts/check.sh --clean  # như trên trên một bản sạch của HEAD
```

Tài liệu kiến trúc: `docs/adr/`. Quy ước làm việc của nhóm: `CLAUDE.md`.
