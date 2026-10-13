# P4-4 — Subgraph UX (spec cho D2)
bao-design · 2026-10-13. Dựa trên P016 §2, bản đã chốt: id phẳng `<group>__<inner>`, lỗi bên trong có `node_id` = group cấp ngoài cùng
và `inner_path` = mảng id theo cấp, lỗi của chính group là CFG-012. Người làm: bao-frontend. Mình review bằng ảnh chụp 7878.

## 0. Quyết định chính: **đi vào nhóm** (drill-in), không bung tại chỗ
Có hai cách hiện một nhóm khi mở ra:
1. Bung tại chỗ thành một khung chứa các bước con, kiểu ComfyUI group.
2. Đi vào một màn riêng của nhóm, có breadcrumb, kiểu Unreal "collapsed graph" / Houdini subnet.

Chọn **cách 2**, vì:
- Bung tại chỗ phá bố cục lane (Nguồn → … → Xuất): mỗi lần bung là các bước khác phải dịch chỗ, mắt người dùng mất dấu.
- Nhóm lồng được tới 3 cấp và phẳng ra được tới 500 node. Bung tại chỗ ở mức đó thì canvas không còn đọc được trong một màn hình (luật One-Screen).
- Người mới không bao giờ phải thấy bên trong; người giỏi có một chỗ riêng để chỉnh sâu.

Ghi chú về từ ngữ: UI gọi là **"nhóm"**; "subgraph" chỉ dùng trong code và tài liệu kỹ thuật.

## 1. Nhóm khi đang gập (trên canvas cha)
- Dùng cùng khung node thường (rộng, cổng, header), nhưng có 2 dấu hiệu nhóm. Theo luật Never-Colour-Alone, không dùng màu riêng.
  - Icon Lucide `layers` đặt trước tiêu đề.
  - Mép dưới có **một đường hairline lệch 3px** như một lá bài đặt chồng (border token `border.subtle`, theo luật Hairline-First). Không dùng bóng đổ.
- Chỉ hiện **cổng và tham số được expose**. Inspector của nhóm giống node thường; ở chế độ Nâng cao có thêm khối "Bên trong" liệt kê các bước con.
- Nút **Mở nhóm** (icon `square-arrow-down-right`, tooltip `canvas.group.open`) nằm ở góc phải header, chỉ hiện khi hover hoặc focus.
  Bấm đúp vào node hoặc nhấn Enter khi node đang được chọn cũng mở nhóm.
- **Trạng thái gộp** lấy từ các bước con: fold `run_status.nodes` theo tiền tố trước dấu `__` đầu tiên, đệ quy. Thứ tự ưu tiên, cái nào khớp trước thì hiện:
  `failed` > `blocked_policy` > `waiting_memory` > `running` > `interrupted` > `queued` > `not_ready` > `skipped` (chỉ khi mọi bước con đều skipped) >
  `done`/`cached` (chỉ khi mọi bước con đều xong; có ít nhất 1 bước chạy thật thì hiện `done`, không thì hiện `cached`) > `ready`.
  - Khi `running`: hiện tiến độ `canvas.group.progress` "{done_count}/{step_count} bước" thay cho %.
- **Trust:** nếu `trust_min` của nhóm thấp hơn "chính chủ" thì hiện cùng badge với node plugin. Nhóm không bao giờ nâng trust (P016).
- **Lỗi:** badge lỗi trên nhóm hiện số lỗi bên trong (`canvas.group.inner_errors`). Lỗi của chính nhóm (CFG-012) hiện như lỗi của node thường.

## 2. Bên trong nhóm
- **Breadcrumb** đặt ở toolbar, ngay sau tên workflow: `Workflow › Dạy model › Vòng học`. Mỗi phần bấm được để lên đúng cấp đó.
  Phím tắt lên một cấp: `Alt+↑`; `Esc` cũng lên một cấp nếu không có popover hay ô nhập nào đang mở. Dùng ký tự `›` để hiển thị; id thật vẫn dùng `__`.
- **Hai node biên** không xoá được:
  - **Đầu vào nhóm** (`canvas.group.boundary_in`) ở lane đầu tiên, mang các cổng vào đã expose.
  - **Đầu ra nhóm** (`canvas.group.boundary_out`) ở lane cuối.
  - Cả hai dùng cùng shape cổng theo kiểu dữ liệu. Nền là `surface.sunken` để phân biệt với bước thật.
- Lane layout và `fitView` (minZoom 0.75) áp dụng y như canvas gốc.
- **Nhóm dựng sẵn (first-party) chỉ xem được cấu trúc.**
  - Breadcrumb hiện icon `lock`, tooltip `canvas.group.readonly`.
  - Không kéo thêm bước, không nối hay gỡ dây.
  - Inspector của bước con vẫn mở được để xem. Tham số nào được expose thì hiện ghi chú `canvas.group.param_exposed` và một link quay về nhóm để sửa ở đó.
- **Nhóm tự tạo (`user.*`)** sửa được bên trong như canvas thường.
- Chuyển cảnh khi vào/ra: scale từ vị trí node nhóm, 180 ms, ease-out. Có `prefers-reduced-motion` thì cắt thẳng, không animation.
- **Focus:**
  - Khi vào nhóm, focus đặt vào bước con đầu tiên.
  - Khi ra, focus về lại node nhóm.
  - Breadcrumb dùng `nav aria-label` (`canvas.group.breadcrumb_label`), phần cuối có `aria-current="page"`.

## 3. Lỗi bên trong nhóm (G-3 / thanh kiểm tra)
- Thanh kiểm tra và thẻ lỗi hiện **đường dẫn** trước tiêu đề lỗi, ví dụ `Dạy model › Vòng học`, tính từ `inner_path` (lấy tiêu đề node theo i18n, không dùng id).
- Nút **Chỉ cho tôi** / action `open_inner_step` làm 3 việc: đi vào đúng cấp, chọn bước con, focus đúng `param` hoặc `port` nếu có.
- Inspector của nhóm có khối lỗi: dòng đầu là `canvas.group.inner_errors`, sau đó mỗi lỗi một dòng gồm đường dẫn, tiêu đề lỗi và nút `action.open_inner_step`.
- Ở canvas cha, dây nối vào cổng của nhóm hiện lỗi kiểu như với node thường. Ánh xạ cổng do `graph.expand` lo.

## 4. Tạo và tách nhóm (D2b: làm nếu còn thời gian trong 1.0)
- Chọn ≥2 bước → nút toolbar **Gộp thành nhóm** (icon `group`, `Ctrl+G`). App tạo `user.<slug>`, tự expose các cổng có dây đi ra ngoài vùng chọn, và đặt tiêu đề mặc định `canvas.group.default_title`.
- **Tách nhóm** (icon `ungroup`, `Ctrl+Shift+G`) chỉ áp cho nhóm `user.*`. Thao tác này đặt các bước con trở lại canvas cha.
- Tiêu đề do người dùng đặt là text không tin cậy: render dạng text thuần, giới hạn độ dài theo P016.

## 5. Thư viện
- Nhóm dựng sẵn nằm trong thư viện như bước thường, có icon `layers`.
- Node `available=False`: làm mờ, có icon `lock`, tooltip lấy `unavailable_reason`. Vẫn đặt lên canvas được và validate báo CFG-011.

## 6. Key i18n (design viết; frontend thêm cùng commit code)
| Key | vi | en |
|---|---|---|
| canvas.group.open | Mở nhóm | Open group |
| canvas.group.progress | {done_count}/{step_count} bước | {done_count}/{step_count} steps |
| canvas.group.inner_errors | {error_count} lỗi bên trong nhóm | {error_count} errors inside this group |
| canvas.group.inner_title | Bên trong | Inside |
| canvas.group.boundary_in | Đầu vào nhóm | Group inputs |
| canvas.group.boundary_out | Đầu ra nhóm | Group outputs |
| canvas.group.readonly | Nhóm dựng sẵn chỉ xem được. Chỉnh thiết lập ở bước nhóm bên ngoài. | Built-in group: view only. Change its settings on the group step outside. |
| canvas.group.param_exposed | Chỉnh ở bước nhóm | Set on the group step |
| canvas.group.breadcrumb_label | Vị trí trong workflow | Location in workflow |
| canvas.group.root | Workflow | Workflow |
| canvas.group.up | Lên một cấp | Up one level |
| canvas.group.create | Gộp thành nhóm | Group steps |
| canvas.group.ungroup | Tách nhóm | Ungroup |
| canvas.group.default_title | Nhóm mới | New group |
| action.open_inner_step | Mở bước bị lỗi | Open the failing step |

`inner_errors` hiện cả khi chỉ có 1 lỗi: tiếng Việt không chia số nhiều. Bản EN dùng biến thể `inner_errors_one` = "1 error inside this group" khi `error_count` = 1.

## 7. Nghiệm thu (design review)
Ảnh chụp 7878 (dark), mỗi ảnh là một màn:
1. Nhóm gập ở trạng thái thường và ở trạng thái `running` có tiến độ.
2. Nhóm gập có lỗi bên trong.
3. Bên trong một nhóm dựng sẵn: breadcrumb có khoá, có 2 node biên.
4. Sau khi bấm "Mở bước bị lỗi": đúng cấp, đúng bước, ô param được focus.
5. (Nếu làm D2b) Gộp rồi tách nhóm.

Tester: E2E đi vào và ra 3 cấp, kiểm tra focus trả về đúng chỗ, và Esc không thoát nhóm khi đang có popover mở.

## 8. Icon (Lucide 1.51.0, vendor vào `site/src/icons/lucide/`)
- Đã có: `layers`, `lock`, `chevron-right`.
- Cần vendor thêm từ đúng bản 1.51.0, giống các lần trước: `square-arrow-down-right` (Mở nhóm), `group`, `ungroup` (D2b).
  Nếu bản 1.51.0 thiếu icon nào thì báo mình chọn icon thay; không tự vẽ icon riêng.
