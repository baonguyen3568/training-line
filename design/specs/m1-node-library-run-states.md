# Spec nhanh M1-5: thư viện node, canvas trống, trạng thái node khi chạy
bao-design · 2026-10-12 · cho bao-frontend (M1-4). Spec cấp thực thi; nền tảng ở P010 §3–§6, §11.
Mục tiêu M1: người dùng **tự tạo** pipeline từ đầu, gồm kéo node → nối dây → bấm Chạy → thấy từng node chạy. User đánh giá app hiện tại là "chưa nối hay tạo pipeline gì hết, hơi thô".

## 1. Bố cục
```
┌ header ────────────────────────────────────────────────────────────────────────────┐
├──────────────┬─────────────────────────────────────────────────────┬───────────────┤
│ THƯ VIỆN 264px│  canvas (làn ①–⑦)                                   │ inspector 320px│
│ 🔍 Tìm bước…  │                                                     │ (node đang chọn│
│ ▾ ① Nguồn     │                                                     │  / gợi ý)      │
│   ● Chọn model│                                                     │               │
│   ■ Dataset HF│                                                     │               │
│ ▾ ② Chuẩn bị  │                                                     │               │
│ ...           │                                                     │               │
├──────────────┴─────────────────────────────────────────────────────┴───────────────┤
│ thanh kiểm tra + ▶ Chạy                                                            │
└────────────────────────────────────────────────────────────────────────────────────┘
```
- Thư viện **luôn ở bên trái** và thu gọn được (nút «, nhớ trạng thái). Inspector ở bên phải. Thư viện không thay chỗ inspector.

## 2. Thư viện node
**Nhóm** theo làn, đúng thứ tự ①–⑦. Tiêu đề nhóm dùng `canvas.lane.*` và mở/đóng được. Mặc định mở hết vì v1 chỉ có ~8 node.
Nhóm không có node nào thì ẩn. Node bên thứ 3 nằm trong nhóm riêng "🧩 Đã cài thêm" ở cuối (P010 §15).

**Mỗi mục** cao khoảng 52px:
```
● Fine-tune toàn bộ                      ← icon cổng ra chính (hình + màu họ cổng) + node.<id>.title
  Dạy model trả lời theo dữ liệu của…    ← node.<id>.desc, 1 dòng, cắt …, full trong tooltip
```
- Node chỉ có ở Advanced: ở Simple vẫn hiện nhưng **mờ** + nhãn `canvas.advanced_only` khi rê chuột. Không kéo được.
- Node `policy_block`: hiện "⛔ Bị chặn" và không kéo được.

**Tìm kiếm** (ô ở đầu, phím tắt `/`):
- lọc ngay khi gõ, khớp không dấu và không phân biệt hoa thường ("nen" khớp "Nén");
- khớp cả title, desc, type_id và từ đồng nghĩa (P010 §5: nén↔quantize↔FP8, chấm điểm↔benchmark, dạy↔fine-tune/train);
- khi đang tìm thì kết quả xếp phẳng, không theo nhóm, có ghi tên làn bên phải;
- không có kết quả: "Không có bước nào khớp. Thử từ khác." (`library.no_match`);
- `Esc` xoá ô tìm. `↓` / `Enter` chọn kết quả rồi thêm node vào giữa vùng đang nhìn, theo đúng làn.

**Thêm node** (3 cách, kết quả như nhau):
1. Kéo từ thư viện thả vào canvas.
2. Double-click hoặc Enter trên mục → thêm vào **đúng làn** của node, ở hàng trống kế tiếp.
3. Kéo dây từ một cổng ra rồi thả vào chỗ trống → mở ô tìm đã lọc theo kiểu tương thích (P010 §4). Chọn node thì node được thêm và **tự nối dây**.

**Khi kéo**:
- Bóng node có kích thước thật, mờ 70%. Làn mà node thuộc về được tô sáng (`canvas.lane_band` đậm hơn + viền `accent.focus` 2px). Các làn khác giữ nguyên.
- Thả **vào làn đúng**: node đặt tại chỗ thả, x khớp lưới 20px.
- Thả **vào làn khác**: vẫn nhận, nhưng node **tự trượt về làn đúng** (animation 150ms, tắt nếu `prefers-reduced-motion`). Toast 1 dòng: "Đã đặt *{node}* vào làn {lane}" (`library.snapped_to_lane`). Lý do: làn là gợi ý bố cục, không phải luật. Không bắt người dùng thả lại.
- Thả **ra ngoài canvas** (vào thư viện, inspector, header): huỷ, không có toast.
- Thả **chồng lên node khác**: dời xuống hàng trống gần nhất trong làn, không chồng nhau.
- Node mới thêm được chọn sẵn → inspector mở thiết lập của nó. Node nhấp nháy viền 1 lần (600ms).

**Bàn phím / a11y**: Tab vào thư viện, mũi tên lên/xuống, Enter để thêm. Mục có `role="option"` trong `listbox`. Thêm xong thì focus chuyển sang node mới trên canvas.

## 3. Canvas trống lần đầu
Mở "Workflow trống" (`gallery.blank_workflow`), hoặc xoá hết node:
```
           ①  Nguồn  (làn được tô nhẹ)
   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   │  ⬚  Kéo "Dữ liệu" vào đây    │   ← ô gợi ý nét đứt nằm trong làn ①, căn giữa theo chiều dọc
   │  hoặc bấm để chọn             │
   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
   Mỗi workflow bắt đầu từ dữ liệu và một model gốc.   [Xem mẫu có sẵn]
```
- Bấm vào ô gợi ý → mở nhóm ① trong thư viện, focus ô tìm.
- Có node đầu tiên thì gợi ý chuyển sang bước kế tiếp. Gợi ý **chỉ hiện 1 câu một lúc**, đặt ngay cạnh cổng còn trống:
  1. Có Dữ liệu mà chưa có Model → "Thêm *Chọn model gốc*" (làn ①).
  2. Có Model + Dữ liệu mà chưa có bước train → "Kéo *Fine-tune* vào làn ③ rồi nối dây".
  3. Có node mà cổng bắt buộc chưa nối → nhấp nháy nhẹ cổng đó + "Nối ■ Dữ liệu vào đây" (chung CFG-003).
  4. Đủ một đường từ Nguồn → Train → bấm Chạy được → gợi ý tắt.
- Gợi ý tắt được ("Ẩn gợi ý", nhớ theo người dùng), và không bao giờ là modal.
- Copy cần thêm: `canvas.empty.drop_here`, `canvas.empty.or_click`, `canvas.empty.explain`, `canvas.empty.see_templates`, `canvas.hint.add_model`, `canvas.hint.add_train`, `canvas.hint.connect_port`, `canvas.hint.hide`. Đã có trong i18n (`canvas.empty.*`, `canvas.hint.*`).

## 4. Nối dây (nhắc lại P010 §4, mức thực thi)
- Bắt đầu kéo dây: cổng tương thích **phóng to 1.4× + viền sáng**, cổng không tương thích mờ 30%. Dây đang kéo dùng màu cổng nguồn.
- Thả vào cổng sai: dây bật về + toast `canvas.wire_rejected`. Không có modal.
- Một cổng vào chỉ có 1 dây. Nối dây mới vào cổng đã có dây → **thay dây cũ** (dây cũ mờ dần 150ms), kèm toast có nút Hoàn tác.
- Chọn dây → Delete để xoá. Rê chuột lên dây → dây dày lên 3px + hiện nút ✕ ở giữa.

## 5. Trạng thái node khi chạy
Mỗi trạng thái = **viền/thanh màu + icon + chữ** ở chân node. Không bao giờ chỉ dùng màu. Token: `color.node_state.*`, copy: `node_state.*`.

| Trạng thái | Chân node | Viền node | Thêm |
|---|---|---|---|
| not_ready | ▢ Chưa đủ | thường | cổng thiếu nhấp nháy khi bấm Chạy |
| ready | ○ Sẵn sàng | thường | |
| queued | ⧗ Đang chờ | thường, nét liền | số thứ tự chờ nếu có |
| cached | ↺ Dùng lại kết quả lúc {time} | thường | tooltip: "Không có gì thay đổi nên không chạy lại" |
| running | ◐ Đang chạy {percent}% | 2px `node_state.running` + **thanh tiến độ 3px** ở đáy node | ETA nếu backend có; node train hiện sparkline loss 40×16px; percent không biết → thanh chạy vô định, chữ "Đang chạy" |
| done | ✓ Xong · {duration} | thường | nút nhỏ "Xem kết quả" → panel monitor/kết quả |
| interrupted | ⏸ Tạm dừng | 2px warning | thẻ lỗi MEM-002 trong inspector; banner bền trên canvas |
| failed | ✕ Lỗi · {title ngắn} | 2px critical | inspector mở thẻ lỗi; node hạ nguồn → skipped |
| skipped | – Bỏ qua | thường, node mờ 60% | tooltip: "Bước trước bị lỗi" |
| blocked_policy | ⛔ Bị chặn | 2px critical | |

**Dây khi chạy**: dây ra từ node done chuyển nét liền đậm 2.5px. Dây vào node running có hạt sáng chạy dọc (tắt khi `prefers-reduced-motion`; khi đó thay bằng dây nét đứt động 0 → tĩnh). Các dây khác giữ nguyên.
**Toàn cục khi chạy**:
- Nút ▶ Chạy đổi thành **■ Dừng** (secondary; hỏi xác nhận nếu đã chạy > 1 phút). Thanh kiểm tra đổi thành tiến độ chung: "Bước {done_count}/{total_count} · {node đang chạy}".
- Canvas **khoá sửa cấu trúc** khi đang chạy: không thêm/xoá node hay dây; vẫn xem, chọn và cuộn được. Bấm vào thư viện thì toast "Đang chạy — dừng lại để sửa workflow".
- Xong toàn bộ: toast "Chạy xong sau {duration}" + nút "Xem kết quả". Có lỗi: thanh kiểm tra đỏ, focus node lỗi đầu tiên.
- Copy cần thêm: `run.stop`, `run.stop_confirm`, `run.progress`, `run.locked_while_running`, `run.finished`, `run.view_results`, `node_state.done_with_duration`, `node_state.skipped_tooltip`, `node_state.cached_tooltip` — đã có trong i18n (+ `run.finished_with_errors`, `workflow.*`, `library.*`).

## 6. Thao tác node (M1-4)
- Chọn node → Delete để xoá (Hoàn tác được). Ctrl/⌘+D nhân bản: bản sao lệch 20px, giữ tham số, không giữ dây. Menu chuột phải gồm Nhân bản / Xoá / Đổi tên hiển thị.
- Lưu và mở file workflow: nút trong header. Có thay đổi chưa lưu thì hiện chấm ● cạnh tên workflow. Đóng tab khi chưa lưu → hỏi xác nhận (beforeunload).

## 7. Tiêu chí review (mình sẽ chụp theo từng phần nhỏ)
1. Từ canvas trống, chỉ dùng chuột, tạo được Dữ liệu → Model → Fine-tune → Đo điểm và bấm Chạy được, **< 2 phút**, không cần đọc tài liệu.
2. Làm lại bước 1 chỉ bằng bàn phím (Tab, Enter, `/`, dropdown "Nhận từ").
3. Trạng thái từng node đọc được trong ảnh đen trắng (icon + chữ, không dựa vào màu).
4. Đủ cả light và dark.
