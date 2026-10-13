---
title: Xử lý sự cố bộ nhớ, ổ đĩa và GPU
description: Ý nghĩa các thông báo về bộ nhớ, dung lượng và khoá GPU, và cách xử lý.
sources: [docs/adr/ADR-005-storage-retention.md#quyet-dinh, .team/contracts/S-4-preflight.md, src/vlm_pipeline/common/error_catalog.yaml, src/vlm_pipeline/i18n/vi.yaml]
---

# Xử lý sự cố bộ nhớ, ổ đĩa và GPU

Trước mỗi tác vụ nặng, app kiểm tra bộ nhớ, ổ đĩa và GPU. Phần lớn sự cố được chặn **trước khi chạy**, nên không làm
treo máy hay hỏng dữ liệu.

## "Không đủ bộ nhớ cho bước này" (MEM-001)

App ước lượng bước này cần nhiều bộ nhớ hơn mức máy đang còn trống.

**Cách xử lý:**
Thông báo có sẵn các nút sửa nhanh; mỗi nút đổi một tham số của node rồi để bạn chạy lại:
- **Dùng LoRA** hoặc **QLoRA** thay cho fine-tune toàn bộ (nếu model hỗ trợ).
- **Giảm kích thước ảnh** (một nửa số điểm ảnh tối đa).
- **Giảm batch** (một nửa).

Hoặc đợi ứng dụng khác trên máy dùng ít bộ nhớ hơn rồi chạy lại.

<!-- popover: "App tính thế nào?" -->
Cần còn trống ≥ mức dự kiến + max(10 GB, 20%), tối thiểu 20 GB, và swap trống ≥ 2 GB. Trên DGX Spark, CPU và GPU dùng
chung 128 GB nên ứng dụng khác trên máy cũng chiếm phần này.

App **không** tự giải phóng bộ nhớ bằng cách xoá bộ nhớ đệm của hệ điều hành hay dừng ứng dụng khác.

## "Đã tạm dừng để bảo vệ máy" (MEM-002)

Trong lúc chạy, bộ nhớ trống xuống dưới mức an toàn. App theo dõi liên tục:

- dưới **14 GB**: cảnh báo;
- dưới **8 GB**: lưu tiến độ và dừng có kiểm soát.

Nếu đang ghi checkpoint, app chờ tối đa 5 phút cho ghi xong để checkpoint không hỏng, trừ khi bộ nhớ trống dưới 4 GB
thì dừng ngay.

**Cách xử lý:** nếu model hỗ trợ chạy tiếp, chọn **Chạy tiếp từ checkpoint** hoặc **Chạy tiếp, tiết kiệm bộ nhớ hơn**
(bật gradient checkpointing: chậm hơn nhưng tốn ít bộ nhớ hơn). Nếu không, chạy lại khi máy rảnh hơn.

## "Hết bộ nhớ khi đang chạy" (MEM-003)

Model dùng nhiều bộ nhớ hơn dự kiến giữa chừng, thường do ảnh lớn hoặc câu quá dài.

**Cách xử lý (có nút sửa nhanh):** giảm một nửa độ dài chuỗi tối đa, bật gradient checkpointing, hoặc giảm một nửa
batch, rồi chạy lại.

## "Phục vụ model chiếm quá nhiều bộ nhớ" (MEM-004)

Thiết lập phục vụ model đang giữ quá nhiều bộ nhớ cho bộ đệm, có thể ảnh hưởng ứng dụng khác. Nút sửa nhanh: đặt về
mức an toàn (25% bộ nhớ cho phục vụ model) hoặc giảm một nửa độ dài tối đa của model.

## "Không đủ dung lượng ổ đĩa" (DISK-001)

Ổ đĩa không đủ chỗ cho các checkpoint sắp ghi. App cần:
**dung lượng 1 checkpoint × (số checkpoint giữ lại + 1) + 50 GB dự phòng.**

**Cách xử lý:** mở [panel Dung lượng](./quan-ly-dung-luong#panel-dung-luong) để xoá kết quả tạm, hoặc chỉ giữ 1
checkpoint gần nhất cho node train.

## "Bộ nhớ đệm đầy, cần bạn chọn mục xoá" (DISK-003)

Kết quả lưu tạm vượt giới hạn, nhưng những mục còn lại đều được bảo vệ nên app không tự xoá. Mở panel Dung lượng và
chọn mục muốn xoá.

## "Cần duyệt trước khi dọn dung lượng" (DISK-002)

Chỉ xuất hiện trên máy dùng chung khi tổng xoá trong 24 giờ vượt 20 GB. **Chưa có gì bị xoá.** Gửi lệnh hiển thị
trong thông báo cho quản trị viên.

## "Chưa dùng được GPU lúc này" (SYS-001)

Mỗi lúc chỉ một tác vụ nặng được dùng GPU. Có ba trường hợp:

| Thông báo | Ý nghĩa | Cách xử lý |
|-----------|---------|------------|
| … đang chạy "…" từ … | tác vụ khác đang dùng GPU | đợi trong hàng chờ, hoặc huỷ |
| Khoá máy … vẫn còn nhưng tiến trình đã dừng | khoá bị bỏ lại | báo quản trị viên kiểm tra; app không tự xoá khoá |
| Không đọc được file khoá máy | file khoá hỏng | báo quản trị viên; app coi như máy đang bận |

<!-- popover: "Vì sao app không tự gỡ khoá bị bỏ lại?" -->
Tiến trình có thể vẫn đang chạy ở nơi app không thấy (ví dụ trong container). Gỡ nhầm sẽ để hai tác vụ cùng dùng GPU
và có thể làm cả hai hỏng. Quản trị viên kiểm tra rồi mới gỡ.

## Tự kiểm tra từ dòng lệnh

```bash
uv run python scripts/cleanup/preflight.py --est-peak-gb 40 --ckpt-gb 9
uv run python -m vlm_pipeline.common.gpu_lock status
uv run python scripts/cleanup/disk_report.py --no-write
```

Cả ba lệnh chỉ đọc. Mã kết thúc của `preflight.py`: `0` đạt, `2` thiếu bộ nhớ/swap, `3` thiếu ổ đĩa, `4` GPU bận.
