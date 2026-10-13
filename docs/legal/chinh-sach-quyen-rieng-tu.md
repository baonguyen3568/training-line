---
title: Chính sách bảo mật
description: "{{PRODUCT_NAME}} chạy trên máy của bạn và không thu thập dữ liệu của bạn."
effective_date: "{{TODO_USER: ngày hiệu lực (U10)}}"
---

# Chính sách bảo mật

**Tóm tắt:** {{PRODUCT_NAME}} là phần mềm chạy trên máy của bạn (local-first). Chúng tôi không có máy chủ nhận dữ liệu
của bạn và không thu thập dữ liệu sử dụng (không telemetry). Dữ liệu, model và kết quả train đều nằm trên ổ đĩa của bạn.

## 1. Ai chịu trách nhiệm
{{TODO_USER: tên cá nhân/đơn vị vận hành, địa chỉ, email liên hệ (U7)}}.
Chính sách này áp dụng cho (a) ứng dụng {{PRODUCT_NAME}} bạn cài trên máy và (b) website giới thiệu này.

## 2. Ứng dụng {{PRODUCT_NAME}}

### 2.1 Chúng tôi không thu thập gì
Ứng dụng **không** gửi về cho chúng tôi dữ liệu, ảnh, workflow, log, thông tin máy hay thống kê sử dụng.
Không có tài khoản người dùng và không có máy chủ trung gian.

### 2.2 Dữ liệu được lưu ở đâu
Mọi thứ nằm trong thư mục cài đặt trên máy bạn:
- dữ liệu bạn đưa vào, model đã tải, checkpoint, kết quả benchmark, log chạy job;
- cơ sở dữ liệu nội bộ của ứng dụng (danh sách lần chạy, bộ nhớ đệm kết quả).

Bạn xoá được bất cứ lúc nào qua bảng **Lưu trữ** trong ứng dụng, hoặc bằng cách xoá thư mục.
Ứng dụng không tự xoá model đã train.

### 2.3 Ứng dụng kết nối Internet khi nào
- **Khi chạy job** (train, đánh giá, quantize): job chạy trong container **không có mạng**. Tính năng gửi dữ liệu sử
  dụng của các thư viện bên trong bị tắt (Hugging Face telemetry tắt, Weights & Biases ở chế độ offline).
- **Khi cài đặt hoặc chuẩn bị**: máy bạn tải trực tiếp
  - container từ NVIDIA NGC (`nvcr.io`),
  - model và dataset từ Hugging Face Hub,
  - thư viện từ PyPI/npm.

  Các dịch vụ này nhận địa chỉ IP và thông tin kỹ thuật của yêu cầu tải, theo chính sách bảo mật riêng của họ.
  Chúng tôi không đứng giữa và không nhận được thông tin đó.
- Giao diện ứng dụng chỉ mở trên `127.0.0.1` (chính máy bạn). Thiết bị khác trong mạng không truy cập được.

### 2.4 Token và khoá truy cập
Nếu bạn dùng token Hugging Face hoặc NGC để tải model bị giới hạn (gated), token được lưu trong file `.env` trên máy
bạn. Token không được ghi vào log và không được truyền vào các job train/đánh giá. Hãy dùng token **chỉ đọc**
(read-only).

### 2.5 Cookie và bộ nhớ trình duyệt của ứng dụng
Xem [Chính sách cookie](./chinh-sach-cookie.md). Tóm tắt: có 1 cookie phiên để bảo vệ ứng dụng, 2 mục lưu trong
trình duyệt cho giao diện, và không có cookie quảng cáo hay theo dõi.

## 3. Website này
- Không dùng công cụ phân tích (analytics), pixel quảng cáo hay script của bên thứ ba. Font, icon và hình ảnh đều được
  tải từ chính website.
- Website không đặt cookie theo dõi. Website không có form thu thập thông tin.
- Như mọi website, máy chủ lưu trữ ({{TODO_USER: tên nhà cung cấp hosting (U9)}}) có thể tự động ghi log kỹ thuật:
  địa chỉ IP, thời gian, trang được mở, loại trình duyệt. Log này phục vụ vận hành và chống tấn công, giữ theo chính
  sách của nhà cung cấp đó. Chúng tôi không dùng log để nhận diện cá nhân.
- Khi bạn bấm link sang website khác (GitHub, Hugging Face, NVIDIA…), chính sách của website đó được áp dụng.
  Trình duyệt được yêu cầu không gửi địa chỉ trang trước đó (Referrer).

## 4. Khi bạn liên hệ với chúng tôi
Nếu bạn gửi email (kể cả báo lỗ hổng), chúng tôi chỉ dùng địa chỉ email và nội dung thư để trả lời bạn.
Chúng tôi không chia sẻ chúng cho bên thứ ba và xoá khi không còn cần cho việc xử lý.

## 5. Quyền của bạn
Theo pháp luật về bảo vệ dữ liệu cá nhân ({{TODO_USER: luật áp dụng (U8), đề xuất: Luật Bảo vệ dữ liệu cá nhân
2025 và Nghị định 13/2023/NĐ-CP của Việt Nam}}), bạn có quyền biết, truy cập, chỉnh sửa và yêu cầu xoá dữ liệu cá nhân.
Vì chúng tôi gần như không giữ dữ liệu nào của bạn (ngoài email bạn tự gửi), hãy liên hệ theo mục 1 nếu cần.

## 6. Trẻ em
Ứng dụng là công cụ kỹ thuật, không dành cho trẻ em và không cố ý thu thập dữ liệu của trẻ em.

## 7. Thay đổi chính sách
Nếu một phiên bản ứng dụng thay đổi bất kỳ điều nào ở trên (vd thêm kết nối mạng), chúng tôi cập nhật trang này
**trước** khi phát hành phiên bản đó, ghi ngày hiệu lực mới và nêu thay đổi trong Changelog. Ứng dụng sẽ không bao giờ
bật telemetry mà không hỏi bạn trước.
