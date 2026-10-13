# docs/legal — nội dung trang chính sách (nguồn sự thật)

Owner: bao-security (M2-8). Site (`site/`) **import** các file này, không chép tay.
Bằng chứng cho từng mệnh đề nằm ở `.team/security-reviews/2026-10-13-m2-site-security-spec.md` §5.

| File | Trang site |
|---|---|
| `chinh-sach-quyen-rieng-tu.md` | Chính sách bảo mật (quyền riêng tư) |
| `dieu-khoan-su-dung.md` | Điều khoản sử dụng |
| `chinh-sach-cookie.md` | Chính sách cookie |
| `bao-mat-va-bao-cao-lo-hong.md` | Bảo mật & báo cáo lỗ hổng (bản tiếng Anh: `/SECURITY.md`) |
| `giay-phep-va-ghi-cong.md` | Giấy phép & ghi công |

Quy tắc:
- `{{TODO_USER: …}}` là chỗ user phải chốt (U7–U10). `scripts/security/site_check.py` sẽ fail build site nếu còn sót.
- PR nào đổi hành vi được mô tả ở đây (thêm cookie, kết nối mạng, telemetry, nơi lưu dữ liệu) thì phải sửa file tương ứng
  trong cùng PR. Mục này nằm trong checklist security review.
- `{{PRODUCT_NAME}}` được thay bằng tên sản phẩm khi user chọn (M2-1).
