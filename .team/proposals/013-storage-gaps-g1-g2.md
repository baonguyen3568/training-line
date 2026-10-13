# Proposal 013: storage gaps G-1 (panel API) và G-2 (retention theo lịch)

- Tác giả: bao-cleanup, 2026-10-13. Trạng thái: **ACCEPTED** (PM chốt 2026-10-13).

## Quyết định PM (2026-10-13)
- **Q1**: đồng ý. Scheduler tự áp dụng D3, set-aside đã superseded và evict cache tạm. **Không bao giờ** tự làm item
  PERMANENT_LOSS hay item bị chặn; các item đó vào mục "Cần bạn chọn".
- **Q2**: đồng ý. Scheduler tính vào quota 20 GB/24h của `shared_dev`, dừng khi chạm ngưỡng, không tự duyệt.
- **Q3**: **bỏ qua lượt khi GPU lock đang được giữ** (khác đề xuất), để tránh tranh IO/RAM với job train và để việc
  suy luận về an toàn đơn giản hơn. Lượt sau chạy lại.
- **Q4**: đồng ý. Bản cài cá nhân bật mặc định; `shared_dev` chỉ dry-run cho tới khi PM bật.
- **G-1**: bắt đầu ngay.
- Phạm vi: mở rộng ADR-005, không đổi các luật D3/D7 đã chốt. Chỉ quyết định **cái gì chạy tự động** và **API cho panel**.
- Reviewer: bao-backend (scheduler trong app), bao-security (đường xoá tự động), bao-frontend (schema API).

## Hiện trạng (2026-10-13, đã kiểm trong code)
- `artifacts/retention.py` (D3, set-aside) và `artifacts/cache_retention.py` (node cache) đã có plan/apply. Hiện chỉ có CLI
  gọi chúng: `scripts/cleanup/{retention,cache}.py`.
- `app/storage.py`: `GET /api/storage/plan` và `POST /api/storage/apply` chỉ phục vụ **node cache**. Không có gì
  gọi `evict` tự động.
- Frontend chưa có UI panel.

## G-2: retention theo lịch trong app

### Đề xuất
1. **Một luồng nền** trong process app (`app/storage_scheduler.py`). Chạy lần đầu 10 phút sau khi app khởi động, sau
   đó mỗi 6 giờ.
   - Đảm bảo mỗi lúc chỉ 1 lượt chạy bằng `flock` trên `outputs/system/retention.lock`, để không chạy trùng với
     CLI hay với một app thứ hai.
   - Tắt được bằng `VLM_RETENTION_SCHEDULE=off`.
2. **Chỉ tự áp dụng item an toàn**. Mỗi lượt gồm:
   - (a) node cache: `plan_cache(cap)` → `evict` các mục **không được bảo vệ** (kết quả tạm, tạo lại được);
   - (b) checkpoint D3 + set-aside: `retention.plan()` → `apply()` các item actionable **không có
     `consequence_code == PERMANENT_LOSS`**.
3. **Không bao giờ tự áp dụng**:
   - item PERMANENT_LOSS, ví dụ LATEST của run failed sau 7 ngày, hoặc set-aside `checkpoints.*`;
   - item bị chặn (`TRAINED_MODEL`, KEEP, lock…).

   Các item này chỉ hiện trong panel, mục "Cần bạn chọn".
4. Ghi log `by="app-scheduler"`, `manual=false`, vào `cleanup.jsonl` và `events.jsonl` như CLI.
5. Shortfall (DISK-003) hoặc mức warn/critical của `disk_report` → ghi event `storage_attention`. Frontend hiện chấm
   đỏ ở icon Dung lượng.

### Câu hỏi cần PM chốt
- **Q1. Luật D3 có tự áp dụng không?** D3 gồm strip optim của LATEST sau 7 ngày (hậu quả RESUME_IMPOSSIBLE), xoá
  checkpoint thường sau 7 ngày, xoá `.incomplete`. ADR-005 đã chấp nhận D3 nhưng lúc đó chỉ chạy bằng CLI.
  - **Đề xuất: CÓ**, trừ PERMANENT_LOSS.
  - Phương án thận trọng hơn: chỉ tự áp dụng `.incomplete` và set-aside đã superseded; còn lại vào panel.
- **Q2. Quota máy dùng chung.** `shared_dev` giới hạn 20 GB/24h xoá chưa duyệt.
  - **Đề xuất:** scheduler **cũng tính vào quota đó và dừng khi chạm ngưỡng**, phần còn lại để lượt sau hoặc chờ
    admin duyệt.
  - Không dùng `approved_by="policy"` vì như vậy là tự duyệt cho chính mình.
  - Bản cài cá nhân (`none`): không giới hạn.
- **Q3. Có chạy khi GPU đang train không?**
  - **Đề xuất: CÓ.** Retention đã chặn run đang giữ lock, cache có lease.
  - Rủi ro duy nhất là I/O xoá vài GB cùng lúc với lúc ghi checkpoint. Nếu backend thấy cần thì thêm: bỏ qua lượt khi
    `GPU_LOCK` held.
- **Q4. Mặc định bật hay tắt?**
  - **Đề xuất:** bật ở bản cài cá nhân.
  - Ở `shared_dev`: **bật nhưng ở chế độ dry-run** (chỉ ghi plan vào `outputs/system/retention_plan.json`) cho tới
    khi PM bật hẳn, vì máy dev đang có nhiều run của team cần giữ.

## G-1: API panel Dung lượng (schema v3)

`GET /api/storage/plan` (schema_version 3) giữ nguyên các trường v2. Thay đổi:
- `items[]` gồm **mọi nguồn**.
- Mỗi item có thêm `source`: `node_cache` | `checkpoint` | `set_aside`.
- Mỗi item có thêm `section`:
  - `auto`: scheduler sẽ tự làm;
  - `manual`: cần người chọn, gõ xác nhận khi `confirm_required`;
  - `kept`: chỉ hiển thị, kèm `lock_reason` + `lock_params`, `size_gb=null`, không nằm trong `plan_id`.

Frontend tự nhóm theo `section`. Ví dụ thật lấy từ code: `013-annex-storage-plan.v3.sample.json`.

Trường theo nguồn (khác v2):
- `item_id`: cache key; `run/checkpoints/epoch_…`; `graph:<node>/<entry>`.
- `display_key`: `storage.item.checkpoint_of_run` | `storage.item.set_aside_of_run` (params `run_name`, `checkpoint`).
- `action`: `delete` | `strip_optim` (retention).
- `confirm_text` = tên run (giống node cache).

`POST /api/storage/apply` không đổi hình dạng body:
- Server tách `item_ids` theo `source` của plan rồi gọi `cache_retention.evict` hoặc `retention.apply(confirmed=…)`.
  Phần retention gọi với `by="app-user"`.
- Quota DISK-002 tính chung cả hai nguồn.
- Response 200 thêm `pruned[]` (item retention đã làm) cạnh `evicted[]` (cache).
- DISK-002 `admin_command` có thể gồm 2 lệnh nối bằng ` && ` (cache.py, retention.py `--confirm … --approved-by`).

Hai chỗ cần sửa nhỏ để làm được G-1:
- `retention.apply` cần tham số `manual=True` để log giống cache. Hiện quota của retention đếm
  `checkpoint_pruned` không có `approved_by`, nên đã đúng.
- Kiểu `approval_policy` lấy từ `StorageSettings`.

## Thứ tự làm
1. G-1 API (không xoá gì mới, chỉ gom nguồn): làm ngay sau khi PM chốt; frontend gắn UI sau M2.
2. G-2 scheduler: sau khi PM chốt Q1–Q4. Security review bắt buộc vì đây là đường xoá tự động; backend review vì là
   thread trong app.
