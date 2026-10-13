# D4 — "Chuẩn doanh nghiệp" (lối ra tiêu chuẩn, làm đúng chuẩn)
Mock: `index.html` với 4 view: `?v=landing` · `?v=dashboard` (`&fly=1` mở thư viện) · `?v=gallery` · `?v=settings`. Ảnh chụp ở `shots/`. **User đã chọn D4 cho cả site và dashboard.** Thanh chọn view nằm ở dải 24px trên cùng, không che giao diện.

**Luận điểm:** chuẩn SaaS doanh nghiệp, không ẩn dụ. Sản phẩm thật là hero, chi tiết chính xác tới từng đường viền 1px. Mức hoàn thiện tham chiếu: Linear/Vercel/Modal cho site, Unreal Editor 5/Figma cho editor.
**Tên & logo:** **NTC Training**.
- Logo chữ: ô vuông bo góc màu tím chàm, bên trong là chữ "N" hình học; cạnh đó là "NTC" đậm và "Training" nhạt hơn.
- Logo là SVG inline. Chữ N vẫn đọc rõ ở 16–32px: 24px trên thanh điều hướng, 32px trên toolbar dashboard.
- Không dùng màu xanh NVIDIA, không có hình gần giống logo NVIDIA.

**Bảng màu** (tối, xám trung tính hơi lạnh, 4 bậc nền):
- Nền: `#09090b` / `#0f1013` / `#15171b` / `#1c1e23`; đường viền `#23262c` / `#30343b`.
- Chữ: `#ededf0` / `#a7acb6` / `#858b95` (tương phản ≥ 4.5:1 trên `#0f1013`).
- Màu thương hiệu tím chàm `#6e56cf`, chữ trắng trên nền này đạt 5.4:1; chữ/link dạng accent dùng `#b3a6ff`.
- Màu cổng giữ theo design system: Dữ liệu `#199e70` ■ · Model `#3987e5` ● · Kết quả `#d95926` ◆.
- Màu trạng thái: tốt `#0ca30c`, cảnh báo `#fab219`, lỗi `#d03b3b`, luôn kèm icon + chữ.

**Chữ:** Be Vietnam Pro cho toàn bộ giao diện; JetBrains Mono chỉ dùng cho lệnh và con số. Cả hai đều OFL-1.1, tự host ở `../../fonts`. Icon dùng Lucide (ISC, `icons/LICENSE`).

**Khung đầu tiên của landing:**
- Tiêu đề: "Dạy AI đọc hoá đơn, biểu đồ và ảnh của bạn. Chỉ bằng kéo thả."
- Hành động chính ở dạng dùng được ngay: ô lệnh cài kèm nút sao chép (minh hoạ) và nút "Tải về & cài đặt".
- Một dòng sự thật: "Miễn phí · Apache-2.0 · Chạy trên NVIDIA NeMo AutoModel · Tối ưu cho DGX Spark (?)".
- Bên dưới là **editor thật** (iframe của chính view dashboard), không phải ảnh chụp.

**Một khoảnh khắc chuyển động:** khi editor cuộn vào khung nhìn, dây "Dữ liệu học" tự nối. Node Fine-tune chuyển từ "Chưa đủ" sang "Sẵn sàng", thanh kiểm tra từ "✕ 1 lỗi" sang "✓ Mọi thứ đã sẵn sàng", nút Chạy bật lên. Nếu người dùng bật `prefers-reduced-motion`, hiện thẳng trạng thái cuối.

**Dashboard: icon trước, mọi công cụ trong một màn hình:**
- Dải icon bên trái rộng 48px: thư viện, danh sách bước, pipeline, tìm, dung lượng; mỗi icon có tooltip.
- Thư viện mở thành panel nổi 304px cạnh dải icon (phím `/`, đóng bằng Esc), nên canvas được trọn chiều rộng.
- Tab dưới chỉ có icon, kèm tooltip và số đếm.
- Chế độ Đơn giản/Nâng cao là một nút icon có trạng thái bật/tắt.
- Toàn bộ dashboard chỉ còn một nhãn "minh hoạ" kiểu nét đứt cạnh các con số RAM/ổ đĩa.

**M4 (Simple giữ node to, Advanced có thêm chiều sâu):**
- Toolbar chỉ có icon, kèm tooltip:
  - nhóm file (mới/mở/lưu), sửa (hoàn tác/làm lại), xem (sắp xếp gọn/vừa khung/zoom);
  - công tắc Đơn giản/Nâng cao;
  - chip GPU/RAM/ổ đĩa (icon + số);
  - icon Pipeline tham khảo, icon Cài đặt;
  - nút Chạy/Dừng là nút duy nhất có chữ.
- Thư viện bên trái:
  - ô tìm đang gõ "lora" → "2 kết quả", nhãn làn hiện bên phải;
  - phần "Tất cả bước 23", nhóm thu gọn được, có số đếm từng nhóm;
  - icon cờ-lê đánh dấu bước chỉ có ở chế độ Nâng cao; icon chồng lớp đánh dấu bước gồm nhiều bước con.
- **Subgraph mở:** "Fine-tune toàn bộ" là một khung nét đứt, có header gồm trạng thái và nút gập.
  - Bên trong: Nạp model → Bộ xử lý ảnh/tokenizer → Vòng lặp train → Lưu checkpoint → Đánh giá nhanh.
  - Cổng ngoài nằm trên mép khung; tên cổng hiện qua tooltip.
  - "Đo điểm benchmark" để gập, mang icon "gồm 3 bước" để đối chiếu.
- Inspector theo đường dẫn "Fine-tune toàn bộ › Vòng lặp train", hiện các tham số Nâng cao thật với giá trị mặc định: batch tổng 8 / mỗi lần 1, weight decay 0.01, tiết kiệm bộ nhớ bật, giữ 2 điểm khôi phục.

**Pipeline tham khảo (`?v=gallery`, mở từ icon trên dải trái):** 8 thẻ, gồm 2 mẫu của app, 5 nguồn ngoài (DGX Spark playbook, NeMo AutoModel, LLaMA-Factory, ModelOpt PTQ, lmms-eval) và 1 thẻ "Ứng viên — chưa phát hành" làm mờ (Hiểu ảnh và video, giữ "Built on NVIDIA Cosmos", nút Mở bị khoá). Trang chủ không quảng bá Cosmos. Mỗi thẻ có:
- sơ đồ mini (SVG tự vẽ từ danh sách bước);
- nguồn ghi bằng chữ, không logo;
- giấy phép: chỉ ghi khi chắc (Apache-2.0 cho NeMo AutoModel, LLaMA-Factory, ModelOpt; model của mẫu app lấy theo catalog), còn lại "xem nguồn";
- độ khó: "Cơ bản" chỉ cho 3 mẫu app (`templates.json` ghi `basic`), còn lại "chưa đánh giá";
- thời gian luôn "chưa có số đo".
HF TRL bị cắt để giữ tối đa 8 thẻ.

**Cài đặt (`?v=settings`, mở từ icon bánh răng):**
- Bố cục: thanh mục bên trái (icon + nhãn ngắn); bên phải là các hàng nhãn + điều khiển.
- Giải thích chỉ nằm sau "?".
- **Mục khoá** chỉ đọc, có icon ổ khoá; popover nói ai khoá và vì sao:
  - 1 job nặng mỗi lúc;
  - ngưỡng bộ nhớ (ADR-005);
  - hạn mức 500/800 GB;
  - xoá > 20 GB cần duyệt (approval_policy threshold);
  - quy tắc lưu giữ 7/3/14 ngày;
  - chỉ nhận token "Read";
  - node bên thứ 3 không dùng Internet;
  - node chưa xác minh dùng GPU bị chặn.
- **Chỉnh được:** trần bộ nhớ đệm 200 GB (P005 R1), số điểm khôi phục 2, preset 3 lượt / Vừa / Tất cả mẫu (theo default trong code).
- Hạn mức RAM mỗi job và lịch dọn dẹp hiện "chưa đặt".
- Token Hugging Face: ô che `hf_••••••••` + nhãn "✓ Đã kết nối".
- Thanh lưu chỉ hiện khi có thay đổi.

**Đi xuyên các bề mặt:** cùng token màu, header và kiểu cổng cho landing → docs (Starlight dùng chung token) → dashboard. Popover "?" là cơ chế chung cho chữ phụ trên mọi trang.

**Rủi ro thật:** quen thuộc, ít nhận diện riêng. Trông giống các SaaS tối khác (Linear/Vercel); dấu ấn riêng chỉ nằm ở chi tiết (cổng hình + màu, logo NTC).

**Chưa xong:**
- Dây "Dữ liệu kiểm tra" từ bước Chia sang Đo điểm vẫn chạy ngang sau khung subgraph; app thật cần auto-layout hoặc định tuyến dây.
- Số trên mock là minh hoạ (RAM 74/121 GB, 14/9 GB, 42%). Dashboard chỉ còn một nhãn "minh hoạ" cạnh RAM/ổ đĩa; thẻ lỗi và hero trên trang chủ có nhãn riêng.
