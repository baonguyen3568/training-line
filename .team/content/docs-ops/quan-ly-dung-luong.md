---
title: Quản lý dung lượng
description: App dùng ổ đĩa vào đâu, tự dọn những gì, và cách tự chọn mục cần xoá trong panel Dung lượng.
sources: [src/vlm_pipeline/app/storage.py, src/vlm_pipeline/artifacts/cache_retention.py, src/vlm_pipeline/i18n/vi.yaml#storage, docs/adr/ADR-005-storage-retention.md]
status_note: "API panel Dung lượng đã có (GET /api/storage/plan, POST /api/storage/apply); giao diện panel đang được làm — người dựng trang kiểm lại trước khi phát hành."
---

# Quản lý dung lượng

Train model tạo ra nhiều dữ liệu lớn: model tải về, checkpoint, model đã nén, kết quả từng node. Trang này giải thích
app giữ gì, tự dọn gì, và khi nào cần bạn chọn.

## Bốn nhóm dữ liệu

| Nhóm | Ví dụ | App tự xoá? |
|------|-------|-------------|
| **Model đã tải** | model gốc từ Hugging Face | Không. Xoá thì phải tải lại. |
| **Model bạn đã train** | kết quả của node Fine-tune | **Không bao giờ tự xoá.** Bạn xoá tay được, phải gõ tên để xác nhận. |
| **Dữ liệu** | bộ dữ liệu đã tải và đã xử lý | Không. |
| **Kết quả tạm (có thể tính lại)** | đầu ra của các node khác trong workflow | Có, khi bộ nhớ đệm vượt giới hạn. |

Lịch sử chạy (nhật ký, số đo, thông tin lần chạy) **luôn được giữ lại**, kể cả khi dữ liệu lớn của lần chạy đó đã bị
xoá. Nhờ vậy bảng điều khiển vẫn hiện "đã xoá lúc … bởi …" thay vì một liên kết hỏng.

## Bộ nhớ đệm kết quả node

Khi chạy lại một workflow, node nào có đầu vào và tham số không đổi sẽ dùng lại kết quả cũ thay vì chạy lại. Các kết
quả này nằm trong bộ nhớ đệm có giới hạn:

- Giới hạn mặc định: **200 GB hoặc 10% dung lượng trống của ổ đĩa**, lấy số nhỏ hơn.
- Khi vượt giới hạn, app xoá **kết quả tạm dùng lâu nhất trước**, đến khi về dưới giới hạn.
- App **không** xoá: model bạn đã train, mục đã ghim, mục mà một workflow đang chạy cần dùng, model gốc mà một phần học
  thêm (LoRA) còn cần để chạy, và mục có thông tin bị hỏng.
- Khi xoá, app chỉ xoá phần nặng (checkpoint, engine, file kết quả). Thư mục xuất bản (`export/`), nhật ký và số đo
  được giữ.

Nếu xoá hết những gì được phép mà vẫn vượt giới hạn, app **không** xoá thêm mà báo:
**"Bộ nhớ đệm đầy, cần bạn chọn mục xoá"** và mở panel Dung lượng.

## Panel Dung lượng

Panel chia thành hai phần:

- **App sẽ tự dọn khi cần**: kết quả tạm có thể tính lại.
- **Cần bạn chọn**: mục được bảo vệ, kèm lý do (ví dụ "Model bạn đã train — app không tự xoá").

Mỗi mục ghi dung lượng, lần dùng cuối và **hậu quả nếu xoá**, ví dụ:

- "Có thể tạo lại" — chạy lại workflow sẽ tạo lại.
- "… sẽ mất vĩnh viễn — không khôi phục hay tạo lại y hệt được."
- "Model vẫn dùng được, nhưng không thể train tiếp run … từ điểm này."

### Xoá một mục

1. Mở panel Dung lượng. App lập **danh sách xem trước**; chưa có gì bị xoá.
2. Chọn các mục muốn xoá.
3. Với mục được bảo vệ, **gõ đúng tên** hiển thị để xác nhận.
4. Bấm **Xoá vĩnh viễn**.

<!-- popover: "Vì sao danh sách bị làm mới?" -->
Danh sách xem trước chỉ có hiệu lực 10 phút. Quá thời gian đó, app kiểm tra lại vì trong lúc chờ có thể đã có workflow
bắt đầu dùng một mục. Mục nào trở nên đang được dùng hoặc được bảo vệ sẽ bị bỏ qua và ghi rõ lý do.

### Máy dùng chung: cần quản trị viên duyệt

Trên máy dùng chung (hồ sơ `shared_dev`), nếu tổng dung lượng bạn xoá trong 24 giờ vượt **20 GB**, app **không xoá**
mà báo "Cần duyệt trước khi dọn dung lượng" kèm lệnh để quản trị viên chạy:

```bash
uv run python scripts/cleanup/cache.py --items <danh sách mục> --apply --by <tên quản trị viên>
```

Không thể lách bằng cách chia nhỏ: các lần xoá trong 24 giờ được cộng dồn. Trên máy cá nhân, mặc định không cần duyệt.

## Dọn từ dòng lệnh (quản trị viên)

Mọi script dọn dẹp **chỉ xem trước** khi không có `--apply`, và bắt buộc ghi tên người thực hiện (`--by`):

| Lệnh | Việc |
|------|------|
| `uv run python scripts/cleanup/disk_report.py` | báo cáo dung lượng dự án theo từng nhóm, mức cảnh báo |
| `uv run python scripts/cleanup/retention.py` | xem checkpoint nào sẽ bị dọn theo [chính sách lưu giữ](./chinh-sach-luu-giu) |
| `uv run python scripts/cleanup/cache.py` | xem bộ nhớ đệm node và những mục sẽ bị xoá |
| `uv run python scripts/cleanup/worktrees.py` | dọn thư mục làm việc tạm của các lần chạy đã xong |

Thêm `--apply --by <tên>` để thực hiện. Mỗi lần xoá được ghi vào `outputs/system/cleanup.jsonl`.

## Giữ lại một lần chạy

Tạo file rỗng tên `KEEP`:

- trong thư mục lần chạy (`outputs/runs/<run>/KEEP`) — giữ toàn bộ lần chạy;
- trong một thư mục checkpoint — chỉ giữ checkpoint đó.

Xoá file `KEEP` khi không cần giữ nữa.
