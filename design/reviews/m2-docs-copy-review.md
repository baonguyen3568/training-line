# Review copy tài liệu M2 (docs/*)

Người review: design (copy). Ngày 2026-10-13. Chỉ ghi nhận xét, không sửa trang. Đối chiếu: `design/m2/sitemap.md` (Quy tắc nội dung), `DESIGN.md`, `src/vlm_pipeline/i18n/vi.yaml`.

**Tóm tắt**
1. Kết luận chung: nội dung đúng, giọng bình tĩnh, có dấu đầy đủ, không có tên cũ "VLM Studio", không "Powered by/đối tác". Cần sửa trước khi phát hành: 1 trang là placeholder (`bat-dau`), 1 trang còn `{{TODO_USER}}` (`cai-dat`), và phần lớn trang dài hơn câu hỏi của nó.
2. Vấn đề xuyên trang 1: **trùng lặp nặng** quanh bộ nhớ/ổ đĩa/GPU. Ngưỡng bộ nhớ, "một tác vụ nặng", 500/800 GB, lệnh preflight, chu kỳ 6 giờ xuất hiện ở 3–5 trang, dễ lệch số khi một trang được cập nhật.
3. Vấn đề xuyên trang 2: **thuật ngữ nằm trong thân bài** (bf16/fp32, gradient checkpointing, QLoRA, optimizer, ANLS, nTED, bộ nhớ hợp nhất, container) ở các trang dành cho người không code; và nhiều Hint chứa thông tin bắt buộc (máy dùng chung) cần đưa ra thân bài.
4. Vấn đề xuyên trang 3: **thông tin mâu thuẫn hoặc có thể đã cũ**: "Bấm Dừng rồi chạy tiếp được" (faq) trái với "nếu model hỗ trợ train tiếp" (xu-ly, chinh-sach); catalog chưa ghi Cosmos là ứng viên; faq nói app chưa đính kèm giấy phép trong khi app đã có EXP-001; link app `/docs/ma-loi#<CODE>` chưa có trang đích.
5. Lặt vặt lặp lại: chữ "team" (ngôn ngữ nội bộ, nên đổi thành "chúng tôi"); "ước tính" trên docs trong khi UI ghi "ước lượng"; "RAM" và "bộ nhớ" dùng lẫn.

---

## Trang của research

### bat-dau — sửa lớn
- `description: Trang tài liệu mẫu của khung site.` và thân bài "Trang tài liệu mẫu (M2-3)…" → viết trang thật: tiêu đề "Bắt đầu: chạy workflow đầu tiên trong 10 phút" (số phút chỉ ghi khi đã đo, chưa đo thì bỏ), 3 bước: mở app → mở template "Đọc hoá đơn → JSON" → bấm Kiểm tra, rồi Chạy. Lý do: đang là placeholder, lại ở `order: 1` nên người đọc gặp đầu tiên.
- `order: 1` trùng với `cai-dat` (cũng `order: 1`) → đặt `bat-dau` = 0, `yeu-cau-he-thong` = 1, `cai-dat` = 2. Lý do: thứ tự sidebar không xác định.
- Hint "Node là gì?" giữ lại, nhưng thêm "Đọc thêm" → `/docs/node-va-cong/`. Lý do: theo quy tắc popover có link đọc thêm.

### node-va-cong — sửa nhỏ
- Bảng mã lỗi có `MEM-001`, `SYS-001` → chỉ giữ các mã CFG, thay hai dòng MEM/SYS bằng câu "Lỗi bộ nhớ, ổ đĩa, GPU: xem [Xử lý sự cố](/docs/xu-ly-su-co-bo-nho/)." Lý do: trùng với xu-ly-su-co-bo-nho và kiem-tra.
- Cột "Khi nào" nên kèm tiêu đề lỗi như app hiển thị, ví dụ `CFG-001` "Hai bước này không nối được với nhau", `CFG-003` "Bước này còn thiếu đầu vào". Lý do: người dùng tra theo câu họ thấy trên màn hình (vi.yaml).
- `CFG-010 | … (ví dụ dùng LoRA với Nemotron Parse, model chỉ hỗ trợ fine-tune toàn bộ)` → "Model không train được theo cách bạn chọn. Ví dụ: Nemotron Parse chỉ hỗ trợ fine-tune toàn bộ." Lý do: khớp tiêu đề vi.yaml, câu ngắn hơn.
- Hint "Node nặng và node nhẹ": "chạy trong container riêng trên GPU" → "chạy trên GPU, mỗi lúc một bước". Lý do: "container" là thuật ngữ, người đọc không cần biết.
- Heading cần anchor ổn định theo `type_id` (sitemap mục 6: node "?" → `/docs/node-va-cong#<type_id>`). Hiện bảng node không có anchor nào. Đề xuất mỗi node có id (vd `<span id="train.lora">`). Lý do: link từ app sẽ gãy.
- Bảng kiểu cổng 9 dòng ngay đầu trang → giữ 5 kiểu người dùng hay gặp (`Dataset`, `Model`, `Adapter`, `EvalReport`, `ExportArtifact`) trong thân bài, đưa 4 kiểu còn lại vào Hint "Các kiểu khác". Lý do: trang trả lời "nối node thế nào", bảng đầy đủ là thông tin phụ.

### kiem-tra-truoc-khi-chay — sửa nhỏ
- `DGX Spark dùng **bộ nhớ hợp nhất**: CPU và GPU dùng chung 128 GB.` → "Máy có 128 GB bộ nhớ, dùng chung cho cả máy và các ứng dụng khác." và đặt thuật ngữ vào `<Hint title="Bộ nhớ hợp nhất là gì?">`. Lý do: quy tắc 2.
- Khối công thức `> Cần trống ≥ mức cao nhất dự kiến + dự phòng (lớn hơn trong hai giá trị…)` → thân bài chỉ giữ ví dụ "Train Nemotron Parse v1.1 cần khoảng 20 GB, nên máy phải còn trống ít nhất 30 GB.", đưa công thức vào Hint "App tính mức dự phòng thế nào?". Lý do: công thức là giải thích phụ.
- Bảng "Bảo vệ khi đang chạy" (14/8/4 GB, 5 phút) → thêm một câu đầu: "Nếu app dừng bước của bạn, đó là để bảo vệ máy, không phải lỗi của bạn. Tiến độ được lưu nếu kịp." Lý do: quy tắc 7; hiện bảng đi thẳng vào số.
- Mục 5 "mỗi checkpoint … khoảng 8 GB" trong khi yeu-cau-he-thong và chinh-sach-luu-giu ghi "8,1 GB" → thống nhất "khoảng 8,1 GB". Lý do: lệch số giữa trang.
- Mục 4 GPU trùng xu-ly-su-co-bo-nho (SYS-001) và node-va-cong → giữ 1 câu + link sang xu-ly. Lý do: trùng lặp.
- Ghi chú cho tester: memguard còn quy tắc theo % tổng bộ nhớ (`for_total`, chọn mức chặt hơn). Trang chỉ ghi số tuyệt đối; tester xem có cần thêm "hoặc theo tỉ lệ, lấy mức chặt hơn" không.

### catalog-model — sửa nhỏ
- Tiêu đề `Catalog model` → "Các model có sẵn trong app". Lý do: tiêu đề phải trả lời câu hỏi; "catalog" là từ nội bộ.
- Dòng Cosmos Reason2 2B/8B, cột Ghi chú chỉ ghi "Ảnh và video; cần chấp nhận điều khoản" → "Ứng viên, chưa có template. Ảnh và video; cần chấp nhận điều khoản". Dòng Qwen3.5 2B đang trống → "Dùng cho template Hiểu ảnh". Lý do: quy tắc 6, ADR-001 am.2 (template 2 đã sang Qwen3.5-2B); trang hiện để Cosmos ngang hàng với model template.
- Cột "Nén FP8" dùng giá trị `Chính thức` cùng Hint "công cụ nén của NVIDIA (ModelOpt) hỗ trợ" → đổi thành "Hỗ trợ / Thử nghiệm / Không hỗ trợ". Lý do: "chính thức" đặt cạnh NVIDIA dễ đọc thành bảo chứng (quy tắc 6).
- Cột `RAM khi train` và nhãn `(ước tính)` → "Bộ nhớ khi train", "(ước lượng)". Lý do: UI hiển thị "ước lượng" (comment `ram_source` trong `configs/catalog/models.yaml`); docs nên dùng cùng chữ với app.
- `Mọi model trong bảng đều cho phép dùng thương mại…` lặp lại trong faq-ky-thuat → giữ ở catalog, faq chỉ link. Lý do: trùng lặp.
- Mục "Fine-tune toàn bộ hay LoRA?" có "khoảng 30% tham số; phần nhìn ảnh được giữ nguyên" → giữ một câu "Cập nhật phần lớn model", phần 30% đưa vào Hint sẵn có "Vì sao Nemotron Parse dùng fine-tune toàn bộ?". Lý do: chi tiết phụ.

### doc-diem-benchmark — sửa nhỏ
- Tiêu đề `Benchmark và cách đọc điểm` → "Cách đọc điểm benchmark". Lý do: một câu hỏi, tiêu đề trả lời nó.
- Bảng "Benchmark chung", cột `Chỉ số` (ANLS, "Độ chính xác nới lỏng (±5%)", "Điểm/1000") → bỏ cột khỏi thân bài, đưa vào Hint "Mỗi benchmark chấm thế nào?". Lý do: thuật ngữ cho kỹ sư.
- Dòng `nTED | Độ giống cấu trúc cây…` → chuyển vào Hint "Các chỉ số phụ". Thân bài chỉ giữ Field F1, JSON khớp hoàn toàn, Tỉ lệ bị cắt. Lý do: người đọc chính chỉ cần chỉ số chính.
- `team yêu cầu điểm của app lệch số công bố **không quá 1,5 điểm**… rồi mới coi đường đo là đáng tin` → "Chúng tôi chỉ coi cách đo của app là đáng tin khi điểm lệch số công bố không quá 1,5 điểm, cùng cài đặt." Lý do: "team" là ngôn ngữ nội bộ; câu quá dài.
- Mục "So với số hãng công bố" trùng câu FAQ "Vì sao điểm của tôi khác số hãng công bố?" và dòng gioi-han-da-biet "Chưa đối chiếu…" → trang này sở hữu chủ đề, FAQ chỉ còn 1 câu + link. Đồng thời ghi rõ ở đây trạng thái "chưa đối chiếu xong". Lý do: hiện người đọc trang này tưởng đã đối chiếu.
- Bullet "Sau khi nén, mức tụt chấp nhận được … 2–3 điểm với NVFP4" → nếu app chưa có NVFP4 cho người dùng thì ghi "(NVFP4: chưa có trong app)". Lý do: tránh hứa tính năng chưa có.

### phan-cung-va-bo-nho — sửa lớn
- Trang mở bằng H2 `## Máy đang dùng`, không có câu trả lời → thêm câu mở: "Máy có 128 GB nhưng thường chỉ còn trống khoảng 40 GB, vì máy chạy chung với ứng dụng khác. Vì vậy model cần chờ máy rảnh hoặc chọn model nhỏ hơn." Đổi tiêu đề thành "Vì sao máy 128 GB vẫn báo thiếu bộ nhớ?". Lý do: quy tắc 1. (Số 40 GB lấy từ `budget_note` trong `configs/catalog/models.yaml`, đã có trong sources.)
- Mục "Máy đang dùng" (GB10, aarch64, 128 GB, container NGC) trùng bảng yeu-cau-he-thong → xoá, thay bằng link. Lý do: trùng lặp.
- Mục `## Độ chính xác số học khi train` (bf16, fp32, 5,3% vs 63,8%) → chuyển sang faq-ky-thuat, gộp với câu "Val loss giảm đều…" (cùng chủ đề lỗi bf16). Lý do: nội dung kỹ sư, không trả lời câu hỏi của trang.
- `ví dụ lần đo train LoRA Qwen3.5 0.8B bắt đầu khi máy chỉ còn trống **42 GB**` → "Khi chúng tôi đo, máy chỉ còn trống khoảng 42 GB." Lý do: câu ngắn hơn; chi tiết model đã có ở bảng số đo.
- Bảng "Số đo thực tế": `308 giây cả lần chạy`, `20 giây cho mỗi bước 8 ảnh`, `7 ảnh mỗi phút` → giữ, nhưng cần tester kiểm nguồn của số Parse (19,8 GB, 20 giây, 7 ảnh/phút, 5,3%, 63,8%, 1,6 GB). `sources:` chỉ có `.team/knowledge/research.md` và file perf của Qwen, không có file perf của Parse. Lý do: quy tắc 4.
- Hint "Vì sao bộ nhớ đo được khác số trong log train?" nói về "tiến trình biên dịch phụ" → giữ nhưng rút còn 1 câu: "Chúng tôi đo bộ nhớ trống của cả máy, vì đó là con số quyết định máy có bị treo hay không." Lý do: ngắn, bỏ chi tiết kỹ thuật.

### gioi-han-da-biet — sửa nhỏ
- `Lần train đã sửa lỗi bf16 cho kết quả train tốt; đang chấm trên tập kiểm tra` → "Đang chấm điểm trên tập kiểm tra." Lý do: "lỗi bf16" là thuật ngữ; "kết quả train tốt" không có số.
- `NVIDIA chỉ hỗ trợ chính thức chạy suy luận (không phải train) dòng model này trên DGX Spark, và dòng Cosmos Reason2 đã chuyển sang bảo trì hạn chế` → "Hướng dẫn của NVIDIA cho DGX Spark chỉ có phần chạy thử, chưa có phần train. Dòng Cosmos Reason2 đang ở chế độ bảo trì hạn chế." Lý do: bỏ "chính thức" (quy tắc 6), câu ngắn. Cần thêm nguồn cho "bảo trì hạn chế" (P4-1 đã có trong sources, tester xác nhận).
- `Chưa chạy qua vLLM, đường phục vụ nhanh vẫn đang kiểm` → "Đang thử cách chạy nhanh hơn." Lý do: "vLLM" là thuật ngữ; Hint bên dưới đã giải thích.
- Bảng "Đo điểm": `Cần MMBench v1.1, máy chủ tải đang không phản hồi; tạm hoãn` → "Đang tạm hoãn vì nguồn tải dữ liệu chưa phản hồi." Lý do: tên phiên bản không giúp người đọc.
- Mục "Máy" trùng yeu-cau-he-thong ("chỉ một máy") và kiem-tra ("một tác vụ nặng") → giữ dòng "chưa hỗ trợ nhiều GPU/nhiều máy", bỏ dòng "một tác vụ nặng". Lý do: đó không phải giới hạn mới, mà là quy tắc đã nói ở trang khác.
- Ghi chú: trang này cần có ngày cập nhật ở đầu ("Cập nhật: 2026-10-13"). Lý do: description hứa "cập nhật theo tiến độ thật".

### faq-ky-thuat — sửa nhỏ
- `### Tôi bấm Dừng rồi chạy tiếp được không?` `Được.` → "Thường là được: app giữ checkpoint gần nhất và chạy tiếp từ đó. Một số model chưa hỗ trợ chạy tiếp; khi đó app train lại từ đầu và giữ bản cũ ([Chính sách lưu giữ](/docs/chinh-sach-luu-giu/))." Lý do: mâu thuẫn với xu-ly-su-co-bo-nho (MEM-002) và chinh-sach-luu-giu; BUG-003 nằm trong sources chính trang này.
- `app giữ thêm phần dự phòng (ít nhất 10 GB, không dưới 20 GB tổng)` → "app cộng thêm ít nhất 10 GB dự phòng, và luôn cần trống ít nhất 20 GB." Lý do: "không dưới 20 GB tổng" khó hiểu.
- `App chưa tự đính kèm chúng vào thư mục model đã lưu; tính năng này đang được làm.` → kiểm lại với export: app đã có lỗi `EXP-001` "Thiếu văn bản giấy phép để xuất model", nên có thể đã đính kèm. Nếu đã có: "Khi lưu model, app tự đính kèm giấy phép và dòng ghi công." Lý do: có thể đã cũ.
- Hint "Ví dụ thật": `val loss đạt 0,195 … bị cắt ở 86% hoá đơn` → cần nguồn trong `sources:` (file perf/eval của lần train lỗi). Lý do: quy tắc 4.
- `### Tôi dùng model đã train vào việc thương mại được không?` lặp câu "Mọi model trong catalog…" → trả lời 2 câu + link sang catalog-model và trang Giấy phép. Lý do: trùng lặp.
- `### Vì sao template hoá đơn tắt "phạt lặp từ"?` có `repetition_penalty` 1,1 trong thân bài → hợp lệ vì trang này dành cho kỹ sư; chỉ cần ghi rõ ở description: "dành cho người đã quen train model". Lý do: quy tắc 2 cho phép thuật ngữ khi trang rõ ràng dành cho kỹ sư.

---

## Trang của cleanup

### yeu-cau-he-thong — sửa nhỏ
- `Mỗi image container của dự án | khoảng 29,6 GB` trong khi cai-dat ghi `Image gốc của NVIDIA nặng khoảng 29,4 GB` → ghi rõ hai dòng "Image gốc NVIDIA: khoảng 29,4 GB" và "Image train của app: khoảng 29,6 GB". Lý do: người đọc tưởng lệch số.
- `(chưa tính model từ 27B trở lên)` và `Mỗi model cỡ 27–35B cần thêm khoảng 55–72 GB.` → xoá. Lý do: catalog không có model này; câu gây hiểu là app hỗ trợ.
- `Driver 580 (đã thử bản 580.142)` → tester kiểm "580.142" có trong sources. Lý do: quy tắc 4.
- `App cảnh báo khi dữ liệu dự án vượt **500 GB** và báo mức nguy hiểm từ **800 GB**.` trùng bảng ở chinh-sach-luu-giu → để ở quan-ly-dung-luong (xem phần Trùng lặp), trang này link. Lý do: trùng lặp.
- Mục `## Trên máy dùng chung` (một tác vụ nặng, không sudo) trùng kiem-tra và chinh-sach → rút còn 1 câu + link. Lý do: trùng lặp.
- Hint "Vì sao không dùng thẳng image của NVIDIA?": "bản vá lạ làm build thất bại" → giữ. Không cần sửa.

### cai-dat — sửa lớn
- `Địa chỉ kho mã: {{TODO_USER: địa chỉ kho mã công khai}}` → điền URL thật (`ntc-training`) hoặc ẩn trang tới khi có. Lý do: placeholder lọt ra bản phát hành.
- Hint "Cài trên máy dùng chung" chứa `Dùng scripts/dev/app.sh thay cho uv run vlm-app` → đưa ra thân bài thành bước 4b hoặc `:::note`. Lý do: thông tin bắt buộc, không phải phụ (quy tắc 3).
- `uv run vlm-app --worker container` cùng bảng có `--worker fake | Mặc định` → thêm câu "Nếu quên `--worker container`, app chỉ giả lập và không train thật." Lý do: lỗi dễ gặp nhất của người mới.
- Mục `## 5. Tự kiểm tra trước tác vụ nặng`: `--est-peak-gb 40 --ckpt-gb 9` không giải thích → thêm "40 = bộ nhớ bước đó cần (GB), 9 = dung lượng một checkpoint (GB)", hoặc bỏ mục, link sang xu-ly-su-co-bo-nho. Lý do: trùng nguyên lệnh với xu-ly; số không có nghĩa với người đọc.
- Mã kết thúc `2 | Thiếu bộ nhớ hoặc swap` → "Thiếu bộ nhớ" kèm Hint "swap" nếu cần. Lý do: thuật ngữ.
- Tên image `vlm-automodel:26.08-5` → giữ (quy ước container `vlm-`), nhưng cần ghi một lần "tên kỹ thuật bên trong vẫn dùng tiền tố `vlm-`". Lý do: người đọc thấy "ntc-training" ở bước 1 rồi gặp `vlm-` ngay sau đó.

### quan-ly-dung-luong — sửa lớn
- Trang dài nhất (104 dòng) và lặp lại chinh-sach-luu-giu ở: chu kỳ 6 giờ, "không bao giờ dọn model đã train", "lịch sử luôn được giữ", luật bộ nhớ đệm → trang này chỉ trả lời "Làm sao giải phóng ổ đĩa?" (panel Dung lượng, xoá mục bảo vệ, ghim); luật giữ/xoá để ở chinh-sach. Lý do: quy tắc 8.
- Hint "Trên máy dùng chung": `Mặc định app chỉ **chạy thử**: liệt kê những gì sẽ dọn mà không xoá` → đưa ra thân bài. Lý do: thông tin quyết định "vì sao ổ đĩa không giảm".
- Mục `## Giữ lại một lần chạy` (tạo file rỗng `KEEP`) → "Bấm **Ghim** trên mục trong panel Dung lượng." Chuyển cách tạo file `KEEP` vào mục dòng lệnh. Lý do: người không code không tạo file trong thư mục `outputs/`. Nếu panel chưa có nút Ghim, ghi rõ "chưa có trên giao diện".
- Mục `## Dọn bằng dòng lệnh` → chuyển sang trang dành cho kỹ sư, hoặc để trong `<details>` ở cuối. Lý do: thân bài cho người không code.
- `## Bốn nhóm dữ liệu`: `Model đã tải | Không bị dọn. Xoá thì phải tải lại.` và mục "Không bao giờ bị dọn tự động" (5 bullet) nói cùng một ý → gộp thành bảng 4 nhóm. Lý do: lặp ngay trong trang.
- `DISK-003 "Bộ nhớ đệm đầy, cần bạn chọn mục xoá"`: ở đây "bộ nhớ đệm" nghĩa là kết quả lưu trên ổ đĩa, ở trang khác "bộ nhớ" là RAM → thêm Hint "Bộ nhớ đệm ở đây nằm trên ổ đĩa, không phải bộ nhớ RAM". Báo frontend/i18n cân nhắc đổi tiêu đề thành "Kết quả tạm đầy…". Lý do: dễ nhầm với MEM-*.

### chinh-sach-luu-giu — sửa nhỏ
- `:::note[Hiện trạng]` lặp ý "6 giờ / máy dùng chung / chạy thử" của quan-ly-dung-luong → giữ ở đây (trang sở hữu luật), xoá đoạn tương ứng ở quan-ly. Lý do: một nguồn duy nhất.
- `Tốt nhất (sai số kiểm định thấp nhất)` → "Tốt nhất (điểm trên tập kiểm tra tốt nhất)", hoặc giữ "val loss thấp nhất" trong Hint. Lý do: "sai số kiểm định" là dịch máy, khó hiểu.
- Đoạn `Khi một model chưa hỗ trợ train tiếp mà bạn vẫn yêu cầu chạy tiếp…` (4 câu dài) → đưa vào Hint "Model chưa hỗ trợ train tiếp thì sao?". Lý do: trường hợp hiếm, chen giữa luồng chính.
- Mục `## Image container` (nhãn `project=vlm-training`, quy trình hai bước) → chuyển sang trang dành cho quản trị/kỹ sư hoặc Hint. Lý do: người đọc chính không quản lý image.
- Mục `## Mức cảnh báo dung lượng` (500/800 GB) trùng yeu-cau-he-thong → để ở quan-ly-dung-luong (nơi panel hiển thị mức), trang này link. Lý do: trùng lặp.
- Hint "Trạng thái optimizer là gì?" (2,7 GB trong 8,1 GB) → tốt, giữ.

### xu-ly-su-co-bo-nho — sửa nhỏ
- Anchor: heading `## MEM-001: Không đủ bộ nhớ cho bước này` sinh slug dài, có dấu. App link `/docs/ma-loi#<CODE>` (sitemap mục 6) nhưng chưa có trang `ma-loi` → hoặc đổi trang này thành đích (`id="MEM-001"` cố định), hoặc tạo `ma-loi` rồi redirect. Lý do: link "Đọc thêm" trong thẻ lỗi sẽ gãy.
- Không có câu trấn an → thêm dưới đoạn mở: "Các thông báo này thường do máy đang bận với ứng dụng khác, không phải lỗi của bạn." Lý do: quy tắc 7.
- MEM-001: `Dùng **LoRA** hoặc **QLoRA**` → "Dùng **Fine-tune nhẹ (LoRA)**" (đúng tên node) + Hint cho QLoRA, hoặc bỏ QLoRA nếu catalog chưa có. Lý do: QLoRA không xuất hiện ở catalog hay node-va-cong.
- MEM-002/003: `Bật gradient checkpointing` → "Bật **Tiết kiệm bộ nhớ**" (nhãn vi.yaml `enable_grad_checkpointing`). Lý do: dùng chữ người dùng thấy trên app.
- Thứ tự `DISK-003` đứng trước `DISK-002` → đổi lại 001, 002, 003. Lý do: tra cứu theo mã.
- Mục `## Tự kiểm tra bằng dòng lệnh` trùng cai-dat §5 → giữ ở một trang (đề xuất trang này, vì đây là trang xử lý sự cố), cai-dat link sang. Hint DISK-001 lặp công thức của yeu-cau-he-thong → giữ công thức ở yeu-cau, ở đây ghi ví dụ "giữ 2 checkpoint 8,1 GB cần khoảng 74 GB". Lý do: trùng lặp.

---

## Trùng lặp giữa các trang

| Chủ đề | Trang sở hữu (viết đầy đủ) | Các trang khác chỉ còn 1 câu + link |
|---|---|---|
| Thông số máy (GB10, aarch64, 128 GB, driver) | yeu-cau-he-thong | phan-cung-va-bo-nho, gioi-han-da-biet |
| Vì sao bộ nhớ trống ít hơn 128 GB, số đo thực tế | phan-cung-va-bo-nho | kiem-tra, faq ("App báo thiếu bộ nhớ…") |
| Ngưỡng bộ nhớ trước khi chạy và khi đang chạy (10/20 GB, 14/8/4 GB) | kiem-tra-truoc-khi-chay | faq, xu-ly (MEM-002 chỉ nói "dưới mức an toàn") |
| Mỗi mã lỗi: nghĩa và cách xử lý (MEM/DISK/SYS) | xu-ly-su-co-bo-nho (nên thành đích `/docs/ma-loi`) | node-va-cong (chỉ giữ CFG), kiem-tra, faq |
| Mã lỗi sơ đồ CFG-* | node-va-cong | kiem-tra |
| "Một tác vụ nặng tại một thời điểm" | kiem-tra-truoc-khi-chay | node-va-cong (Hint), phan-cung, gioi-han, yeu-cau, xu-ly |
| Lệnh preflight dòng lệnh | xu-ly-su-co-bo-nho | cai-dat |
| Công thức dung lượng ổ đĩa trước khi train | yeu-cau-he-thong | xu-ly (Hint DISK-001) |
| Luật giữ/xoá checkpoint, chu kỳ 6 giờ, những gì không bao giờ bị xoá | chinh-sach-luu-giu | quan-ly-dung-luong |
| Panel Dung lượng, xoá mục bảo vệ, ghim, mức 500/800 GB | quan-ly-dung-luong | chinh-sach, yeu-cau |
| Giấy phép thương mại của model | catalog-model (+ trang Giấy phép) | faq |
| So với số hãng công bố | doc-diem-benchmark | faq, gioi-han (chỉ giữ trạng thái "chưa đối chiếu") |
| Lỗi bf16 / bản sao fp32 | faq-ky-thuat (một mục gộp) | phan-cung (xoá), gioi-han (bỏ chữ "bf16") |
| Train tiếp sau khi dừng | chinh-sach-luu-giu | faq, xu-ly (MEM-002) |

Về hai cặp được hỏi riêng:
- **phan-cung-va-bo-nho ↔ xu-ly-su-co-bo-nho**: ít trùng nội dung, nhưng phân vai chưa rõ. phan-cung trả lời "vì sao" (máy dùng chung, số đo), xu-ly trả lời "làm gì" (theo mã lỗi). Mỗi mục MEM-* ở xu-ly nên có 1 link "Vì sao máy hay thiếu bộ nhớ?" sang phan-cung, thay vì lặp lại giải thích. Phần trùng thật sự là với kiem-tra (ngưỡng) và yeu-cau (thông số).
- **faq-ky-thuat ↔ gioi-han-da-biet**: gioi-han chỉ liệt kê "chưa làm được + trạng thái" dạng bảng, không giải thích. faq trả lời "vì sao/làm sao" và link sang gioi-han khi câu trả lời là "chưa có". Hiện hai trang chồng nhau ở: đối chiếu số công bố, giấy phép/điều khoản, thời gian train ("trang template sẽ ghi số khi có"). Mục "Train bao lâu thì đủ?" nên chuyển sang gioi-han thành dòng "Chưa có số epoch mặc định".
