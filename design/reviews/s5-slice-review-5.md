# Review UI — S-5 vòng 5: Simple/Advanced, inspector, Run monitor (bao-design, 2026-10-12)
Ảnh: `design/reviews/s5-r5/`. Verdict:
- **Run monitor: APPROVE.** M1 đã sửa: trục learning rate hiện 1.0e-4 / 2.0e-4. Legend hiện giá trị cuối.
- **Gallery: APPROVE.** Có thẻ Cosmos với "Built on NVIDIA Cosmos" và chip 🔑, chỉ một dòng "chưa có số đo".
- **Canvas/inspector: CHANGES REQUESTED**, có 1 lỗi chặn cho buổi thử (I1).

## Chặn
**I1. Dropdown "Nhận từ" không phân biệt được các cổng ra cùng kiểu.**
- "Nhận từ — Dữ liệu học" và "Nhận từ — Dữ liệu kiểm tra" đều hiện option giống hệt nhau: "Chia dữ liệu học / kiểm tra · Dữ liệu".
- Nhiệm vụ T2 (thay dữ liệu, nối lại) phụ thuộc vào đúng chỗ này: Lan không biết chọn cái nào là dữ liệu học, cái nào là dữ liệu kiểm tra.
- Sửa: label của option = tên node + **copy cổng của node nguồn**, vd `node["data.split"].port.train` = "Dữ liệu học". Không dùng nhãn kiểu `port.dataset`.

## Không chặn
1. Thanh kiểm tra hiện "Dự kiến: chưa có số đo · chưa có số đo · chưa có số đo". Khi cả 3 đều unknown thì hiện **một** câu: `estimate.source.unknown_all` ("chưa có số đo thời gian, RAM, ổ đĩa"). Nếu chỉ một vài cái unknown thì ghép nhãn từng phần ("RAM: chưa có số đo").
2. Node "Dataset Hugging Face" chưa hiện summary "cord-v2" như node model. Cần `x-ui.summary` cho `data.hf_load.repo`.
3. Thẻ "Nén model & đo lại điểm" vẫn hiện "~15–40 phút · RAM ~12–20 GB". Nếu chưa đo thì phải là unknown, giống 2 thẻ kia.
4. Copy `node["train.sft_vlm"].desc` trước đây so với LoRA ("tốn hơn LoRA"), gây rối cho template mặc định vốn không có LoRA. Đã đổi thành "Dạy model trả lời theo dữ liệu của bạn bằng cách cập nhật các phần model cần học."
5. Toggle Đơn giản/Nâng cao trên header rõ ràng. "▸ Nâng cao (1)" trong inspector đúng P010 §10. Help text dưới field đọc tốt.
6. Ở Simple, danh sách Lịch sử chạy cần kiểm tra lại: run spike MedPix phải bị ẩn. Ảnh lần này chụp ở Advanced vì script bấm nhầm toggle, nên mình chưa xác nhận được.

## Sau vòng này
Sửa xong I1 thì slice đủ cho T1–T3 của buổi thử. Mình sẽ báo PM.
