# M2-1: các hướng thẩm mỹ để user chọn (NTC Training)

bao-design · 2026-10-13.
- Quy trình: impeccable new-work, seed 00ab95ce, mode Persuade cho landing.
- Không có công cụ sinh ảnh nên mọi mock đều là HTML (code-led).
- Mở bằng trình duyệt: `design/m2/directions/<hướng>/index.html?v=landing|dashboard` (D4 có thêm `gallery|settings`).
- Mọi dashboard đều tối, icon trước chữ, panel dock kiểu Unreal (theo yêu cầu PM/user). Ẩn dụ của mỗi hướng chỉ nằm ở landing và ở chi tiết.

| # | Hướng | Một câu | Ảnh | Rủi ro thật |
|---|---|---|---|---|
| D1 | **Sơn mài** (hướng bốc thăm) | Train model như làm tranh sơn mài: phủ lớp rồi *mài* để hiện hình. Hero cho rê chuột mài lớp son, lộ pipeline dát vàng | `son-mai/shots/` | Màu son gần đỏ báo lỗi; chất "nghệ thuật" có thể lấn tính kỹ thuật; chữ trên nền vỏ trứng rạn hơi khó đọc |
| D2 | **Bản đồ tuyến** (mình xếp cao nhất) | Workflow = tuyến metro: bước là ga, dây là tuyến. Màu tuyến = màu cổng sẵn có của app | `ban-do-tuyen/shots/` | Ẩn dụ metro cho pipeline khá quen; vàng biển báo dễ bị đọc thành cảnh báo; trên mobile bản đồ phải cuộn ngang |
| D3 | **Bảng giờ tàu** (đối thủ "competitive": rõ nhất về sản phẩm) | Bảng lật chữ ở nhà ga: các bước/lần chạy là chuyến tàu, đèn hổ phách = "đang chờ, máy bận" | `bang-gio/shots/` | Người dùng ít "thấy mình" trong đó; lật chữ nhiều sẽ ồn |
| D4 | **Chuẩn doanh nghiệp** (lối ra tiêu chuẩn, PM yêu cầu) | SaaS tối kiểu Linear/Vercel, editor kiểu Unreal/Figma; chính sản phẩm là hero | `chuan/shots/` | Quen thuộc, ít nhận diện riêng |

Chỉ D4 có đủ các yêu cầu bổ sung trong ngày:
- màn Cài đặt (mục bị khoá hiện ổ khoá + popover);
- thư viện node lớn có tìm kiếm;
- subgraph đang bung;
- gallery 8 pipeline tham chiếu có nhãn nguồn và license; độ khó, thời gian ghi "chưa đánh giá/chưa có số đo".

D1–D3 sẽ được bổ sung các phần này nếu user chọn.

## Khuyến nghị của design
- **Dashboard:** dùng khung của **D4** dù landing chọn hướng nào. Đó là thứ user tả ("clean như Unreal"), và D2 cũng đã hội tụ về cùng khung này.
- **Landing và site:** chọn **D2 Bản đồ tuyến** nếu muốn site và app chung một ngôn ngữ màu/hình (cổng ■●◆ = tuyến), dễ hiểu nhất với người no-code. Chọn **D4** nếu ưu tiên "chuẩn doanh nghiệp" an toàn. D1 đẹp và khác biệt nhất, nhưng rủi ro đỏ son / đỏ lỗi.

## Tên & logo
User đặt tên **NTC Training** (PM đang xác nhận chính tả). Mỗi mock có một wordmark nháp, chữ "NTC" là phần chính:
- D1: ô son viền vàng;
- D2: biển tuyến vàng;
- D3: 3 ô lật N·T·C;
- D4: ô chàm, chữ N hình học.
Logo sẽ chốt theo hướng được chọn.

## Không bịa
- Số trạng thái (42%, RAM 74/121, 14/9 GB) và lệnh cài đặt đều có nhãn "minh hoạ".
- Không có testimonial, logo khách hàng, số sao.
- NVIDIA chỉ xuất hiện ở dạng mô tả sự thật; "Built on NVIDIA Cosmos" có trên thẻ template Cosmos.
- Font OFL tự host, kèm file OFL (`design/m2/fonts/`). Icon Lucide kèm LICENSE.

## Còn mở (chuyển sang bản dựng thật)
Hook impeccable bắt được các lỗi sau trong mock: chữ < 11px, tracking quá chặt, padding chật, nhảy cấp heading. Ngoài ra còn bố cục mobile của D1/D2, và nhãn bị cắt trong thư viện D4. Frontend sẽ sửa khi dựng hướng được chọn.
