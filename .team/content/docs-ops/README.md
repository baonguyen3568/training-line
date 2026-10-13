# Nội dung docs vận hành (M2-2) — tác giả bao-cleanup

Bản nháp nội dung tiếng Việt cho các trang docs vận hành của site M2. Người dựng trang (M2-5) chuyển sang MDX/Starlight;
bao-design chỉnh giọng văn. Mỗi file có frontmatter `title`, `description`, `sources` (nguồn sự thật để QA M2-7 kiểm
"không bịa số"). Dòng bắt đầu bằng `> [!note]` / `> [!warning]` là callout; đoạn đánh dấu `<!-- popover -->` nên đưa vào
icon "?" thay vì để trong thân bài.

| File | Trang | Nguồn chính |
|------|-------|-------------|
| `yeu-cau-he-thong.md` | Yêu cầu hệ thống | CLAUDE.md (phần cứng), docker/IMAGES.lock, ADR-005 |
| `cai-dat.md` | Cài đặt & chạy app | scripts/dev/app.sh, app/server.py, frontend/README.md, docker/automodel-2.Dockerfile |
| `quan-ly-dung-luong.md` | Quản lý dung lượng & panel Dung lượng | app/storage.py, artifacts/cache_retention.py, i18n vi.yaml `storage.*` |
| `chinh-sach-luu-giu.md` | Chính sách lưu giữ dữ liệu | ADR-005, P005 D3/D7/§7, artifacts/retention.py |
| `xu-ly-su-co-bo-nho.md` | Xử lý sự cố bộ nhớ, ổ đĩa, khoá GPU | ADR-005 D8, contracts/S-4-preflight.md, error_catalog.yaml |

Khi code đổi (ngưỡng, mã lỗi, lệnh) → cập nhật trang tương ứng; bao-cleanup giữ các trang này khớp code.
