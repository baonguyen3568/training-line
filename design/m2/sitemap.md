# M2-2 — Sitemap & IA (bản 1)
bao-design · 2026-10-13. Giữ nguyên các slug đã có trong `site/`. Thêm/đổi chỗ nào thì báo chủ trang.
Mode: **P** = Persuade (quảng bá, có animation), **R** = Read (đọc hiểu), **O** = Operate (app).
Nguồn sự thật: ADR, P001/P002, `benchmarks/`, `configs/catalog/models.yaml`, i18n. Số nào chưa có thì ghi "chưa có số đo", không ghi số.

## Thanh điều hướng chính (5 mục + 1 nút)
Tính năng · Template · Tài liệu · Giới thiệu · Mã nguồn ‖ nút chính **Tải về & cài đặt**
Footer chia 4 cột:
- **Sản phẩm:** Tính năng, Template, Lộ trình, Thay đổi
- **Tài liệu:** Bắt đầu, Cài đặt, FAQ
- **Dự án:** Giới thiệu, Miễn phí & mã nguồn mở, Liên hệ
- **Pháp lý:** Quyền riêng tư, Điều khoản, Cookie, Giấy phép & ghi công, Bảo mật

## Trang (28)
| # | Slug | Trang | Mode | Owner nội dung | Trạng thái |
|---|---|---|---|---|---|
| 1 | `/` | Trang chủ: hook, demo cơ chế (theo hướng được chọn), 3 lợi ích, template, CTA | P | design | mock M2-1 |
| 2 | `/tinh-nang` | Tính năng: canvas kéo thả, chặn lỗi trước khi chạy, ước lượng RAM/thời gian, lỗi dễ hiểu, chạy lại thông minh (cache), Simple/Advanced, node bên thứ 3 an toàn | P | design | có `tinh-nang.mdx` |
| 3 | `/template` | Thư viện template, đổi tên từ `mau` → `template` theo nav, giữ redirect | P | design + research | có `mau.mdx` |
| 4 | `/template/doc-hoa-don` | Template "Đọc hoá đơn → JSON" (Nemotron-Parse v1.1, CORD-v2) | P/R | research | chờ P-3b |
| 5 | `/template/anh-video` | Template "Hiểu ảnh & video" (Cosmos-Reason2-2B, gated) + dòng **"Built on NVIDIA Cosmos"** (ADR-012) | P/R | research | chờ |
| 6 | `/template/nen-va-do` | Template "Nén FP8 + đo điểm" | P/R | research | chờ |
| 7 | `/truong-hop-su-dung` | Use case: kế toán/hoá đơn, biểu đồ & báo cáo, ảnh tài liệu. **Mô tả việc làm được, không bịa khách hàng** | P | design | mới |
| 8 | `/gioi-thieu` | Giới thiệu: vì sao có dự án, nguyên tắc (5 nguyên tắc PRODUCT.md), đội ngũ ở mức vai trò | R | design | mới |
| 9 | `/ma-nguon-mo` | Miễn phí & mã nguồn mở, thay cho trang "Bảng giá": Apache-2.0, không gói trả phí, đóng góp | P/R | design | có |
| 10 | `/lo-trinh` | Lộ trình (M1 → M2 → …), chỉ ghi việc đã có trên BOARD | R | PM + design | mới |
| 11 | `/thay-doi` | Changelog, sinh từ git tag/commit (không viết tay) | R | cleancode | mới |
| 12 | `/lien-he` | Liên hệ: `mailto` + issue tracker, **không có form** (security §1) | R | design | có |
| 13 | `/docs/bat-dau` | Bắt đầu: quickstart 10 phút | R | research | có, chờ số |
| 14 | `/docs/yeu-cau-he-thong` | Yêu cầu hệ thống | R | cleanup | có |
| 15 | `/docs/cai-dat` | Cài đặt | R | cleanup | có |
| 16 | `/docs/node-va-cong` | Node, cổng & dây | R | research | có |
| 17 | `/docs/kiem-tra-truoc-khi-chay` | Kiểm tra trước khi chạy | R | research | có |
| 18 | `/docs/catalog-model` | Catalog model | R | research | có |
| 19 | `/docs/doc-diem-benchmark` | Đọc điểm benchmark | R | research | có |
| 20 | `/docs/phan-cung-va-bo-nho` | Phần cứng & bộ nhớ (DGX Spark, unified memory) | R | research | có |
| 21 | `/docs/quan-ly-dung-luong` | Quản lý dung lượng | R | cleanup | có |
| 22 | `/docs/chinh-sach-luu-giu` | Chính sách lưu giữ | R | cleanup | có |
| 23 | `/docs/xu-ly-su-co-bo-nho` | Xử lý sự cố bộ nhớ | R | cleanup | có |
| 24 | `/docs/ma-loi` | **Tra mã lỗi**: sinh tự động từ `error_catalog.yaml` + i18n (title/what/cách sửa), mỗi mã có anchor `#MEM-001` để thẻ lỗi trong app link tới. Chỉ hiện title/what/actions, không hiện `match`/slug nội bộ (fixcode). Generator `scripts/docs/gen_error_codes.py` | R | design + fixcode | mới |
| 25 | `/docs/node-ben-thu-ba` | Node bên thứ 3: trust level, quyền, cách cài an toàn (P003 D16) | R | security + design | mới |
| 26 | `/docs/gioi-han-da-biet` + `/docs/faq-ky-thuat` | Giới hạn đã biết · FAQ kỹ thuật | R | research | có |
| 27 | `/chinh-sach/*` | 5 trang: quyền riêng tư, điều khoản, cookie, giấy phép & ghi công, bảo mật & báo lỗ hổng; cộng thêm `ben-thu-ba` | R | security | có (chờ U7–U10) |
| 28 | `/404` | 404: một câu, kèm link Trang chủ + Tài liệu + ô tìm (Pagefind) | R | design | có |

**FAQ cho người dùng phổ thông** (không phải FAQ kỹ thuật): đặt ở cuối Trang chủ + `/tinh-nang`, dạng accordion. Không làm trang riêng, để khỏi trùng `faq-ky-thuat`.

## Quy tắc nội dung (áp cho mọi trang)
1. **Tiếng Việt có dấu, câu ngắn.** Thuật ngữ (LoRA, FP8, unified memory) không đặt ở thân bài mà vào icon **"?" + popover**: 1–3 câu, có link "Đọc thêm" tới docs.
2. **Một trang = một câu hỏi của người đọc.** Tiêu đề trang trả lời luôn câu hỏi đó. Không dùng eyebrow/kicker trên heading.
3. **Số liệu luôn có nguồn.**
   - Mỗi con số có `sources:` trong frontmatter; tester kiểm bằng `scripts/qa/check_site_numbers.py`.
   - Chưa đo thì ghi "chưa có số đo".
   - Không bịa testimonial, logo khách hàng, số sao hay số người dùng.
4. **NVIDIA chỉ xuất hiện ở dạng mô tả sự thật** ("Chạy trên NVIDIA NeMo AutoModel"). Không logo, không #76B900, không "Partner/Powered by".
   - "Built on NVIDIA Cosmos" bắt buộc ở trang #5 và trang Giấy phép.
5. **Animation chỉ có ở trang P:** một khoảnh khắc chính mỗi trang, `prefers-reduced-motion` tắt hết. Trang R không có animation trang trí.
6. **App ↔ docs:** thẻ lỗi trong app có link "Đọc thêm" → `/docs/ma-loi#<CODE>`, node có "?" → `/docs/node-va-cong#<type_id>`. Trang #24 cần anchor ổn định.

7. **Không dùng ảnh hay câu hỏi mẫu của các bộ benchmark** (OCRBench, ChartQA, DocVQA, MMMU…) trên site, docs hay screenshot (security 2026-10-13). Benchmark chỉ hiện điểm tổng hợp kèm trích dẫn. Ảnh minh hoạ "chạy thử" dùng CORD-v2 (CC BY 4.0, có ghi công) hoặc ảnh tự tạo.

## Việc tiếp theo
- **Chốt slug `mau` → `template`:** frontend/cleancode đổi nếu đồng ý.
- **Trang #24:** fixcode và mình làm generator (một trang, đọc YAML lúc build).
- **Copy trang #1, #2, #7, #8, #9:** mình viết sau khi user chọn hướng (M2-1), vì giọng văn phụ thuộc hướng thiết kế.

## Release 1.0 — bản chốt (2026-10-13, 30 trang)
Thay cho bảng trên. Các trang đã có giữ nguyên slug.

**Sản phẩm (9):** `/`, `/tinh-nang`, `/template`, `/truong-hop-su-dung`, `/gioi-thieu`, `/ma-nguon-mo`, `/lo-trinh` (mới), `/thay-doi` (mới), `/lien-he`
**Tài liệu (14):** bat-dau, cai-dat, chuan-bi-du-lieu, node-va-cong, kiem-tra-truoc-khi-chay, catalog-model, doc-diem-benchmark,
phan-cung-va-bo-nho, quan-ly-dung-luong, chinh-sach-luu-giu, xu-ly-su-co-bo-nho, ma-loi, gioi-han-da-biet, faq-ky-thuat
**Pháp lý (6):** 5 trang `/chinh-sach/*` + `ben-thu-ba`
**Khác (1):** `/404`

Thay đổi so với bản 1:
1. **Bỏ 3 trang con `/template/*` (#4–6).** `/template` trở thành thư viện pipeline tham chiếu (≥8 theo luồng C), mỗi pipeline là một khối có anchor
   (`/template#doc-hoa-don`…). Dữ liệu sinh lúc build từ `configs/templates/*.json` + i18n, giống cách làm `/docs/ma-loi`, để site không lệch app.
   Mỗi khối gồm: việc làm được, model, dữ liệu mẫu, giấy phép, số đo (thời gian, RAM, điểm) có nguồn hoặc "chưa có số đo", và nguồn công bố gốc.
   "Built on NVIDIA Cosmos" chỉ hiện ở khối Cosmos. Owner: research (dữ liệu) + frontend (generator) + design (copy khung).
2. **Gộp `/docs/yeu-cau-he-thong` vào `/docs/cai-dat`** thành mục "Trước khi cài" (anchor `#yeu-cau-he-thong`). Hai trang cùng trả lời
   câu "máy tôi cài được không". Trang cũ giữ lại dạng chuyển hướng giống `/mau`. Owner: cleanup.
3. **Hoãn `/docs/node-ben-thu-ba` (#25)** tới khi tính năng cài node bên thứ ba chạy được (quy tắc: không mô tả tính năng chưa có).
4. **`/lo-trinh`:** design viết từ BOARD (Release 1.0, M3, M4). Chỉ ghi "đã xong" hoặc "đang làm", không hứa ngày.
5. **`/thay-doi`:** cleancode làm generator đọc git tag. Khi chưa có tag 1.0 thì trang ghi "Chưa có bản phát hành", không viết tay.
6. **Bung bước lớn (subgraph) và chế độ Nâng cao:** thêm mục vào `/docs/node-va-cong` khi P016 ship (research). **Trang Cài đặt:** thêm mục vào
   `/docs/cai-dat`, giải thích mục bị khoá bởi quản trị (cleanup/backend).
7. FAQ phổ thông giữ dạng accordion ở Trang chủ + `/tinh-nang`, không có trang riêng.
