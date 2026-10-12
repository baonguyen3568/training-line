# Kế hoạch thử nghiệm với người dùng no-code (P010 §14) — bản 1

Tác giả: bao-design · 2026-10-12 · Trạng thái: DRAFT. PM sẽ hỏi user về người tham gia khi vertical slice S-5 chạy được.

## Mục tiêu
Kiểm tra 4 giả thuyết UX quan trọng nhất của P010 trước khi build tiếp:
1. Gallery → "Dùng mẫu này" → "Chạy thử nhanh" là đủ để người không biết code bắt đầu, không cần hướng dẫn.
2. Người dùng hiểu cổng/dây (màu + hình + nhãn) đủ để nối hoặc sửa 1 dây sai.
3. Thanh kiểm tra và thẻ lỗi 4 phần giúp người dùng **tự sửa** lỗi (vd thiếu RAM, thiếu đầu vào) mà không hỏi ai.
4. Badge "Bên thứ 3 – chưa xác minh" và dialog cài làm người dùng dừng lại đọc, chứ không bấm qua theo phản xạ.

## Người tham gia
- 3–5 người **không viết code**, có nhu cầu thật với ảnh + câu hỏi (vd nghiệp vụ, kế toán, nghiên cứu ứng dụng). 5 người thường đủ để thấy phần lớn vấn đề lớn của một vòng thử.
- Không phải thành viên team. Không cần biết ML.
- Cần user/PM giới thiệu người, và đồng ý ghi màn hình (chỉ để phân tích nội bộ, xoá sau khi tổng hợp).

## Thiết lập
- Chạy app trên localhost (127.0.0.1:7870) qua SSH tunnel; tài khoản hoặc máy riêng cho buổi thử.
- **Không chạy job GPU thật** trong buổi thử: dùng FakeWorker hoặc chế độ smoke giả lập (bao-tester), nên không cần GPU_LOCK và không ảnh hưởng máy dùng chung.
- Dữ liệu mẫu đã được security duyệt license; không dùng dữ liệu của người tham gia.
- Ngôn ngữ: tiếng Việt, chế độ Simple (mặc định đã chốt).

## Kịch bản (think-aloud, ~40 phút / người)
Người điều phối chỉ đọc nhiệm vụ, không gợi ý. Tối đa 8 phút mỗi nhiệm vụ.

| # | Nhiệm vụ (đọc cho người tham gia) | Thành công khi | Đo |
|---|---|---|---|
| T1 | "Bạn muốn dạy model đọc hoá đơn tốt hơn. Hãy bắt đầu từ một mẫu có sẵn và cho nó chạy thử." | Mở template mặc định "Đọc hoá đơn → JSON" (Nemotron-Parse v1.1, fine-tune toàn bộ ~0.9B) → bấm "Chạy thử nhanh" | thời gian, số lần bấm sai, có tìm được gallery không |
| T2 | "Thay dữ liệu mẫu bằng thư mục ảnh hoá đơn này." | Thay node "👉 Thay bằng dữ liệu của bạn" và nối lại đúng | có nhận ra node nét đứt không; dùng kéo dây hay dropdown "Nhận từ" |
| T3 | (Chuẩn bị sẵn 1 dây nối sai + 1 đầu vào thiếu) "Workflow này không chạy được. Hãy làm cho nó chạy." | Đọc thanh kiểm tra → sửa cả 2 lỗi | có đọc thẻ lỗi không; dùng nút sửa ngay hay tự sửa |
| T4 | (Giả lập MEM-001) "Máy báo không đủ bộ nhớ. Bạn sẽ làm gì?" | Hiểu thiếu RAM là do **máy dùng chung đang bận**, không phải lỗi của mình, và chọn "Xếp hàng — tự chạy khi đủ bộ nhớ" (hoặc "Xem ứng dụng nào đang dùng bộ nhớ"). Template Parse đã ở cấu hình tiết kiệm nhất (bao-research): không có nút giảm ảnh hay giảm batch | hiểu "máy dùng chung" không; có lo sợ không |
| T5 | (Mở workflow chia sẻ có node bên thứ 3 chưa xác minh, xin GPU) "Đồng nghiệp gửi bạn workflow này. Hãy mở và chạy." | Người dùng **dừng lại**, đọc màn "Không có trong nguồn tin cậy", và nói ra được rủi ro. Không có cách cài 1 chạm | có đọc không; diễn đạt lại rủi ro bằng lời của họ |

| T5b | (1 phút) Cho xem màn "⛔ Bị chặn trên máy này theo chính sách quản trị" của node đó. "Màn này nói gì? Bạn sẽ làm gì tiếp?" | Hiểu là do chính sách của máy, không phải lỗi của mình, và biết hỏi quản trị viên | diễn đạt lại |

- **T5 dùng màn "chưa xác minh + gõ tên"** trên instance thử, với `policy_block` tắt trong cấu hình FakeWorker. Lý do: đây là giả thuyết chính cần kiểm, tức người dùng có dừng lại đọc không. Trên máy dev thật, theo D16, node đó sẽ bị `policy_block`; màn này được kiểm riêng ở T5b.
- **T4**: template mặc định là fine-tune toàn bộ, không có LoRA/QLoRA → nút sửa phải là hành động có thật của recipe (đang hỏi bao-research). Số "~X GB" trên nút phải lấy từ estimator đã hiệu chỉnh sau S-3 (`action_estimates` trong fixture). Nếu S-3 chưa có số thì hiện "chưa có số đo" và bỏ phần đo "hiểu con số", không đặt tay.
- Sau **mỗi** nhiệm vụ: hỏi **SEQ** (Single Ease Question, 1–7): "Nhìn chung, nhiệm vụ này dễ hay khó?"
- **Log sự kiện local** (chỉ trên instance thử, không telemetry): số issue sau mỗi lần validate, action đã bấm, node được focus, thời gian mở dialog cài trước khi quyết định, có cuộn xem phần quyền không. Dùng làm thước đo khách quan bên cạnh quan sát, để so được giữa các vòng thử. Fixture và oracle: `tests/fixtures/usability/` (bao-tester).

Cuối buổi: 3 câu hỏi.
- "Điều gì khó hiểu nhất?"
- "Màu và hình trên các cổng có giúp gì không?"
- "Bạn có dám tự chạy cái này một mình không?" (thang 1–5)

## Chỉ số
- Tỉ lệ hoàn thành từng nhiệm vụ không cần trợ giúp (đối chiếu `expected_issues.json` + log sự kiện).
- SEQ trung bình theo nhiệm vụ.
- Thời gian hoàn thành.
- Số lỗi nghiêm trọng (bế tắc hoặc hiểu sai nguy hiểm, vd cài node lạ không đọc).
- Trích dẫn nguyên văn của người tham gia.
- Ngưỡng cần thiết kế lại (quan điểm): T1 hoặc T3 dưới 3/5 người thành công; hoặc ở T5 có bất kỳ ai cài được node mà không đọc cảnh báo.

## Đầu ra
`design/research/usability-round-1.md`: các vấn đề xếp theo mức nghiêm trọng, có video timestamp nội bộ, kèm đề xuất sửa. Gửi PM, frontend, và chủ của từng phần liên quan.

## Phụ thuộc
- Slice S-5 chạy được đến validate overlay. T4 và T5 cần lỗi giả lập và dialog cài; nếu chưa có thì dùng prototype giấy hoặc HTML tĩnh cho 2 nhiệm vụ đó.
- bao-tester: fixture ở `tests/fixtures/usability/` (t1, t3, t4, t5 + `t5_app_config.json` policy tắt, `t5b_app_config.json` policy bật). Override chỉ được chấp nhận khi `worker: fake` (worker thật → ValueError). Còn thiếu: log sự kiện local (frontend).
- PM/user: người tham gia và đồng ý ghi màn hình.
