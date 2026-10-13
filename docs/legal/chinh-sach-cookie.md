---
title: Chính sách cookie
description: "Website không dùng cookie theo dõi. Ứng dụng chỉ dùng 1 cookie bảo mật."
effective_date: "{{TODO_USER: ngày hiệu lực (U10)}}"
---

# Chính sách cookie

Chúng tôi không dùng cookie quảng cáo, cookie phân tích hay cookie của bên thứ ba, ở cả website lẫn ứng dụng.
Vì vậy bạn không thấy hộp thông báo "chấp nhận cookie".

## Website
| Tên | Loại | Mục đích | Thời hạn |
|---|---|---|---|
| *(không có cookie)* | | | |
| `starlight-theme` | localStorage | Ghi nhớ giao diện sáng/tối bạn chọn ở trang Tài liệu | Tới khi bạn xoá dữ liệu trình duyệt |
| `sl-sidebar-state` | sessionStorage | Giữ trạng thái mở/đóng và vị trí cuộn của mục lục Tài liệu | Mất khi đóng tab |

## Ứng dụng (chạy trên máy bạn, địa chỉ `127.0.0.1`)
| Tên | Loại | Mục đích | Thời hạn |
|---|---|---|---|
| `vlm_session` | Cookie thiết yếu (HttpOnly, SameSite=Strict) | Xác nhận trình duyệt đang mở là trình duyệt bạn dùng để khởi động ứng dụng; chặn website khác điều khiển ứng dụng thay bạn | Hết khi đóng trình duyệt; đổi mỗi lần khởi động lại ứng dụng |
| `vlm.ui.mode` | localStorage | Ghi nhớ chế độ Đơn giản / Nâng cao | Tới khi bạn xoá dữ liệu trình duyệt |
| `vlm.draft.v1` | localStorage | Tự lưu bản nháp workflow chưa lưu, để không mất việc khi lỡ đóng tab | Xoá khi bạn lưu hoặc bỏ bản nháp |

Các mục này chỉ nằm trong trình duyệt trên máy bạn, không gửi tới chúng tôi hay bên thứ ba nào.
Cookie `vlm_session` là bắt buộc để ứng dụng chạy an toàn. Hai mục localStorage là tiện ích: nếu trình duyệt chặn
localStorage, ứng dụng vẫn chạy, chỉ không nhớ được lựa chọn và bản nháp.

## Cách xoá
Xoá "cookie và dữ liệu trang web" cho website này hoặc cho `127.0.0.1` trong cài đặt trình duyệt.
