# Landing 1.0: copy cho các phần còn thiếu
bao-design · 2026-10-13. Người dựng: bao-frontend. Hero và phần "Chặn lỗi" giữ nguyên.
Thứ tự trang: Hero → Chặn lỗi → **Template** → **Chỉ chạy lại phần đã đổi** → **Hỏi đáp** → **Kết**.
Phần giải thích phụ dùng `.help` + popover giống hero. Không thêm animation mới: khoảnh khắc chính của trang vẫn là EditorPreview (quy tắc 5).

## 3. Template
- H2: **Mở một mẫu, bấm Chạy, rồi sửa theo ý bạn.**
- Lead: Mỗi mẫu là một workflow dựng bằng chính các bước trong app. Số đo bên dưới lấy từ lần chạy thật trên DGX Spark; mẫu nào chưa đo thì ghi rõ.
- **3 thẻ** lấy từ `reference-pipelines.json`, đúng thứ tự trong file. Chỉ lấy pipeline có `status: available`, nên hiện chỉ có 2 thẻ.
  Không bịa thẻ thứ ba cho đủ bộ. Nếu còn chỗ, ô cuối là link "Xem tất cả pipeline tham chiếu →" `/template`.
  - Mỗi thẻ có: tiêu đề, MiniDiagram (`steps`), giấy phép (badge), độ khó.
  - Mỗi thẻ có một dòng số đo:
    - Có `measured`: "Dạy model: ~{train_minutes → dur_*} · RAM đỉnh {peak_ram_gb} GB · {metric} {value}".
      Nút "?" mở popover ghi `config`, `n` + CI95, và nguồn. Popover nói "Đo trên một máy DGX Spark", không viết "trên máy bạn".
    - `measured: null`: chữ "Chưa có số đo" ở màu `text.muted`, không có icon cảnh báo.
- Định dạng điểm: tên chỉ số hiển thị "F1 theo trường" (field_f1), giữ 2 chữ số thập phân như nguồn (70.97). Popover ghi khoảng tin cậy 95%: 66.33–75.38 trên 100 ảnh CORD test.
  Không làm tròn thành "71%".

## 4. Chỉ chạy lại phần đã đổi
- H2: **Đổi một bước, chỉ bước đó chạy lại.**
- Body: App nhớ kết quả của từng bước. Bạn đổi thiết lập ở bước đo điểm thì phần dạy model không chạy lại; workflow chỉ chạy từ chỗ bạn sửa trở về sau.
- Popover "?": Một bước chạy lại khi đầu vào hoặc thiết lập của nó đổi, hoặc khi kết quả đã lưu bị hỏng; trường hợp hỏng, app báo trước. Link "Đọc thêm" → `/tinh-nang#chỉ-chạy-lại-phần-đã-thay-đổi`.
- Minh hoạ: 4 node nhỏ, 3 node đầu ở trạng thái `cached` (icon + chữ "Dùng lại kết quả"), node cuối `running`. Dựng bằng chính component node của EditorPreview, không vẽ ảnh tĩnh.

## 5. Hỏi đáp (accordion `<details>`, tối đa 7 câu)
1. **Tôi không biết viết code, có dùng được không?**
   Được. Mở một mẫu, chọn thư mục dữ liệu của bạn rồi bấm Chạy. Chế độ Đơn giản chỉ hiện vài thiết lập quan trọng, mỗi thiết lập đều có mặc định an toàn.
2. **Cần máy như thế nào?**
   Bản 1.0 được thử trên NVIDIA DGX Spark. Máy x86-64 có GPU NVIDIA cài được bằng Docker, nhưng chưa được thử với GPU thật. Xem [Yêu cầu hệ thống](/docs/cai-dat/#yeu-cau-he-thong).
3. **Dữ liệu của tôi có bị gửi đi đâu không?**
   Không. App chạy trên máy của bạn. App chỉ kết nối ra ngoài để tải model hoặc bộ dữ liệu bạn chọn, ví dụ từ Hugging Face.
4. **Có mất phí không?**
   Không. App miễn phí, mã nguồn theo giấy phép Apache-2.0. Mỗi model và bộ dữ liệu có giấy phép riêng; app cho bạn xem trước khi tải. Xem [Miễn phí & mã nguồn mở](/ma-nguon-mo/).
5. **Dạy một model mất bao lâu?**
   Tuỳ model, lượng dữ liệu và số lượt học. App ước lượng thời gian trước khi chạy. Ví dụ: mẫu Đọc hoá đơn mất khoảng 2 giờ trên DGX Spark.
   Kèm "?" hiện: 116 phút, 3 lượt trên 800 ảnh CORD-v2. Nguồn `configs/reference_pipelines.yaml`.
6. **Dạy xong thì dùng model ở đâu?**
   Bạn chạy thử ngay trong app, hoặc xuất model ra một thư mục theo định dạng Hugging Face, kèm văn bản giấy phép cần thiết, để dùng ở máy khác.
7. **Dùng được những model nào?**
   Các model trong danh mục của app, đã được kiểm tra giấy phép và nguồn tải, ví dụ Nemotron Parse, Qwen3.5, Gemma 4. Xem [Catalog model](/docs/catalog-model/).

Câu 6 chỉ đúng khi bước xuất và bước Chạy thử đã có trong template (luồng B/C). Trước khi publish, tester đánh dấu đã kiểm câu 6. Nếu chưa kiểm được thì bỏ câu 6, không sửa thành câu mơ hồ.

## 6. Kết
- H2: **Miễn phí, mã nguồn mở, chạy trên máy của bạn.**
- Nút chính: "Tải về & cài đặt" (`installHref`). Nút phụ: "Xem template" → `/template`.
- Không thêm câu kêu gọi kiểu "Tham gia cùng hàng nghìn người dùng" (không có số thật).
