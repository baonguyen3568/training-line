# Docs vận hành (M2-2): đã chuyển vào site

Bản chính nằm ở `site/src/content/docs/docs/` (Starlight, tác giả bao-cleanup):
`yeu-cau-he-thong.mdx`, `cai-dat.mdx`, `quan-ly-dung-luong.mdx`, `chinh-sach-luu-giu.mdx`, `xu-ly-su-co-bo-nho.mdx`.
Bản nháp markdown cũ: commit 54fbfdd. Khi đổi ngưỡng, mã lỗi hay lệnh trong code thì sửa thẳng trang MDX, rồi chạy
`uv run python scripts/qa/check_site_numbers.py --strict` và `cd site && npm run check`.
