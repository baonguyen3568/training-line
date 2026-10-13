---
title: Bảo mật & báo cáo lỗ hổng
description: "Cách {{PRODUCT_NAME}} được thiết kế để an toàn, và cách báo lỗ hổng cho chúng tôi."
---

# Bảo mật & báo cáo lỗ hổng

## Báo cáo lỗ hổng
Nếu bạn tìm thấy lỗ hổng bảo mật, xin **đừng** tạo issue công khai. Hãy gửi riêng tới:
**[www.whatareme@gmail.com](mailto:www.whatareme@gmail.com)**

Xin gửi kèm:
- phiên bản (hoặc commit) đang dùng và hệ điều hành / phần cứng;
- các bước tái hiện, kèm proof-of-concept nếu có;
- ảnh hưởng mà bạn đánh giá.

Cam kết của chúng tôi: xác nhận đã nhận trong **3 ngày làm việc**, đánh giá ban đầu trong **10 ngày làm việc**, và
thông báo khi có bản sửa. Chúng tôi đề nghị công bố phối hợp: chờ bản sửa hoặc tối đa **90 ngày** kể từ khi báo,
tuỳ điều kiện nào đến trước. Nếu bạn muốn, chúng tôi ghi công bạn trong Changelog.

Chúng tôi sẽ không có hành động pháp lý với người nghiên cứu bảo mật thiện chí: thử trên máy của chính mình, không truy
cập dữ liệu của người khác, không làm gián đoạn dịch vụ, và báo cho chúng tôi trước khi công bố.

### Trong phạm vi
- Ứng dụng {{PRODUCT_NAME}}: server local, giao diện, cách chạy job trong container, xử lý workflow/plugin, các bước tải
  và kiểm tra model.
- Website này.

### Ngoài phạm vi
- Lỗ hổng của phần mềm bên thứ ba (PyTorch, NeMo, vLLM, Hugging Face, Docker, driver NVIDIA…). Hãy báo cho dự án đó;
  chúng tôi sẽ cập nhật phiên bản khi có bản vá.
- Tấn công cần quyền đọc/ghi sẵn trên máy của người dùng (người đó vốn đã kiểm soát máy).
- Thiếu header bảo mật mà không kèm kịch bản khai thác.

## Ứng dụng được bảo vệ như thế nào
- **Chỉ trên máy bạn:** server chỉ nghe tại `127.0.0.1`. Mỗi lần khởi động tạo một mã truy cập mới. Mã này được đổi
  thành cookie phiên `HttpOnly`, `SameSite=Strict`. Server kiểm tra Host, Origin và token CSRF, chặn website khác điều
  khiển ứng dụng (CSRF, DNS rebinding).
- **Job không có mạng:** train, đánh giá và quantize chạy trong container `--network none`. Token (Hugging Face, NGC,
  W&B) không bao giờ được truyền vào job.
- **Chuỗi cung ứng:**
  - container được ghim bằng digest;
  - thư viện Python/npm được ghim phiên bản kèm hash, và chỉ dùng bản đã phát hành ít nhất 7 ngày;
  - model được ghim theo commit, chỉ tải định dạng `safetensors`;
  - code đi kèm model (`trust_remote_code`) phải qua review và được ghim bằng hash.
- **Plugin / node bên thứ ba:** chạy trong container cách ly, không có mạng, phải khai báo quyền. Node chưa được kiểm
  duyệt thì bị chặn mặc định trên máy dùng chung.
- **Xoá dữ liệu:** chỉ xoá trong thư mục dự án, luôn xem trước (dry-run). Model đã train không bao giờ bị tự động xoá.
- **Không telemetry.**

## Khuyến nghị khi dùng
- Dùng token Hugging Face **chỉ đọc**, lưu trong `.env`. Không dán token vào workflow hay chia sẻ file `.env`.
- Cập nhật NVIDIA Container Toolkit và driver lên bản mới nhất có bản vá bảo mật.
- Chỉ cài plugin từ nguồn bạn tin cậy.
- Không mở cổng ứng dụng ra mạng (vd qua reverse proxy) khi chưa có lớp xác thực riêng.
