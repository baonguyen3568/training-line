# Luồng H — Mang model/data của bạn (ADR-021): spec UX
bao-design · 2026-10-14. Người làm: bao-frontend. Copy: bao-design (key bên dưới); mã lỗi: bao-fixcode (§6).
Tinh thần D4 giữ nguyên: icon-first, giải thích phụ nằm trong "?", không thêm ghi chú nhỏ trên mặt node.
Chỉ dùng icon đã vendor (Lucide 1.51.0): shield-check, badge-question-mark, search, download, ban, external-link, loader-circle.

## 1. Bước "Chọn model gốc": hai nguồn
Inspector của `model.hf_pick` có một segmented control ngay dưới tiêu đề:
**[shield-check] Đã kiểm chứng | [search] Từ Hugging Face** (`byo.source.catalog` / `byo.source.hub`).
- **Đã kiểm chứng:** giữ đúng ô chọn catalog hiện tại.
- **Từ Hugging Face:**
  1. Ô nhập repo id (font mono, placeholder `byo.model.placeholder`). Bấm Enter hoặc nút **Kiểm tra** (`byo.check`) mới gọi server, không kiểm theo từng phím gõ.
  2. "Phiên bản (revision)" nằm trong `<details>` `byo.revision` (mặc định là bản mới nhất). Server ghim thành commit sha và hiện 12 ký tự đầu, font mono.
  3. **Đang kiểm tra:** loader-circle + `byo.checking`.
  4. **Kết quả hợp lệ:** một thẻ nhỏ, theo thứ tự:
     - repo (mono) + badge **Chưa kiểm chứng** (§2);
     - dòng thông số `byo.model.facts`: họ kiến trúc · ~{params} tham số · tải ~{size_gb} GB;
     - giấy phép: tên + `license.badge.*`, chi tiết xem lúc tải (G-5);
     - nếu gated: `byo.gated` cùng link mở flow `gated.*` đã có.
  5. Nút chính **Dùng model này** (`byo.model.use`). Bấm xong, node nhận model. Nếu model chưa có trên máy thì mặt node hiện trạng thái `not_downloaded` + nút tải, theo đúng luồng G-5; dialog tải hiện giấy phép và bắt tick đồng ý.
  6. **Kết quả lỗi:** một thẻ lỗi inline. Chữ lấy từ error catalog (§6), không dùng câu chung. Không có nút "Dùng".
- Model tự thêm không được đưa vào catalog. App nhớ 5 repo dán gần nhất ngay dưới ô nhập (`byo.recent`), đọc từ workflow đang mở và localStorage; người dùng bấm vào để kiểm tra lại.

## 2. Badge "Chưa kiểm chứng"
- Nằm trên mặt node (chip nhỏ, icon badge-question-mark, mực `text.secondary`, không dùng màu cảnh báo) và trong thẻ kết quả ở §1.
- Chip này là chỗ DUY NHẤT trên mặt node có "?". Popover dùng `byo.unverified_help`.
- **Preflight:** model chưa kiểm chứng mà chưa từng chạy thử thì có một dòng info (status ink info) `byo.smoke_hint`, kèm nút **Chạy thử nhanh** (`run_smoke`, đã có). Đây chỉ là gợi ý, không chặn Chạy.
- **Ước lượng RAM/thời gian:** dòng `estimate.*` hiện thêm `byo.estimate_upper` trong "?". Số vẫn có nhãn "ước lượng"; không bao giờ ghi "Đo trên DGX Spark".

## 3. Model cần chạy mã riêng (trust_remote_code): bị chặn
Đây là một mã lỗi trong catalog (§6, fixcode). Câu chữ giải thích lý do bằng lời thường và cho 2 lối ra:
- chọn model khác, chuyển sang tab "Đã kiểm chứng";
- đề nghị thêm vào danh mục: action `request_catalog` mở `https://github.com/baonguyen3568/training-line/issues/new` trong tab mới, không có query (cùng điều kiện security như bug report).
Icon `ban`, màu `status_ink.serious`. Không nhắc tới "trust_remote_code" trong câu chính; tên kỹ thuật chỉ nằm trong "Chi tiết kỹ thuật".

## 4. Dataset từ Hugging Face: ghép cột + xem trước
Bước dữ liệu (`data.hf_load`) có cùng segmented `byo.source.*`. Tab "Từ Hugging Face":
1. Nhập repo id, rồi chọn config/split (`byo.data.config`, `byo.data.split`; chỉ hiện khi có nhiều hơn một lựa chọn). Bấm **Kiểm tra**.
2. Kết quả hợp lệ thì mở dialog **Ghép cột dữ liệu** (`byo.map.title`):
   - radio 2 kiểu (`byo.map.mode_qa` / `byo.map.mode_messages`);
   - kiểu Ảnh + câu hỏi + câu trả lời: 3 select `byo.map.image` / `byo.map.question` (tuỳ chọn) / `byo.map.answer`. Nếu bỏ trống câu hỏi thì dùng ô `byo.map.fixed_question`;
   - kiểu Hội thoại: 2 select `byo.map.messages` + `byo.map.image`;
   - server gợi ý sẵn từng select (`byo.map.suggested`, chữ nhỏ cạnh select). Cột sai kiểu (vd chọn cột chữ cho ảnh) thì báo lỗi ngay dưới select;
   - **Xem trước** (`byo.map.preview`): 3–5 mẫu, mỗi mẫu một hàng gồm thumbnail 64px + câu hỏi + câu trả lời (clamp 2 dòng). Mẫu nào lỗi thì hiện lý do thay cho thumbnail;
   - nút **Dùng cách ghép này** (`byo.map.apply`) + Huỷ (hairline).
3. Sau đó bộ kiểm tra chất lượng chạy như với data local (ảnh hỏng, trùng, rò rỉ sang dữ liệu kiểm tra, độ dài). Kết quả hiện trong thanh kiểm tra, không hiện trong dialog.
4. Không chạy loading script. Repo chỉ có script thì báo lỗi §6.

## 5. Gallery / pipeline tham chiếu
- Thẻ pipeline: thêm 1 câu vào popover "?" của dòng số đo (không thêm chữ trên mặt thẻ): `pipelines.swap_hint`.
- Mặt node `model.hf_pick` và `data.hf_load` của một template: không ghi gì thêm. Help của param `model`/`repo` có thêm vế về tab "Từ Hugging Face".

## 6. Mã lỗi cần fixcode thêm (đề xuất ý, câu chữ cuối do design viết khi có mã)
| Trường hợp | Ý câu chính | Lối ra |
|---|---|---|
| Repo không tồn tại / gõ sai | Không tìm thấy "{repo}" trên Hugging Face. | Kiểm tra lại tên |
| Repo riêng tư / gated thiếu token | Cần quyền tải. | Mở flow gated |
| Không phải model ảnh–chữ (VLM) | Model này không đọc được ảnh. | Chọn model khác |
| Kiến trúc chưa hỗ trợ | Engine hiện tại chưa dạy được họ model "{family}". | Chọn model khác / đề nghị |
| Cần chạy mã riêng | §3 | Chọn khác / đề nghị thêm vào danh mục |
| Quá lớn cho máy này | Cần khoảng {need_gb} GB, máy chỉ có {avail_gb} GB trống. | Model nhỏ hơn |
| Dataset chỉ có loading script | Bộ dữ liệu này cần chạy mã riêng để đọc. | Chọn bộ khác / tự tải về thư mục |
| Thiếu cột bắt buộc | Chưa ghép cột "{role}". | Mở lại Ghép cột |

## 7. Nghiệm thu (ảnh 7878, dark)
1. Tab Từ Hugging Face với kết quả hợp lệ (có badge Chưa kiểm chứng).
2. Lỗi bị chặn vì chạy mã riêng.
3. Dialog Ghép cột với phần xem trước.
4. Mặt node có chip Chưa kiểm chứng và preflight có gợi ý Chạy thử nhanh.
5. H-prompt (mock, 2026-10-14, design đã duyệt): ô Câu hỏi để trống thì hiện câu đã train làm gợi ý, kèm chú thích dưới ô; kết quả Chạy thử có dòng "Câu đã hỏi: … (câu lúc train)". Ảnh: `outputs/review/byo/10-trained-prompt-hint.png`, `11-trial-prompt-used.png`. Model một tác vụ (Nemotron Parse) không có ô Câu hỏi, đây là hành vi đúng.
