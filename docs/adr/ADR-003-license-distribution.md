# ADR-003 — License & phân phối

- Trạng thái: **ACCEPTED** — 2026-10-12 (user chốt)
- Nguồn: `.team/proposals/003-security-baseline.md` D6-final, D10–D15 (bao-security); R-1 (bao-research)

## Bối cảnh
Mục đích (user): nghiên cứu/phát triển + **cho người khác chạy/dùng miễn phí, không bán**. Nghĩa là phi thương mại nhưng CÓ phân phối.
GPL kích hoạt khi phân phối, bất kể có bán hay không.

## Quyết định
1. Code của dự án phát hành theo **Apache-2.0**.
2. **Không fork / không vendor ComfyUI** (GPL-3.0). Chỉ học ý tưởng và giao thức, viết lại từ đầu. Chỉ dùng thư viện MIT/Apache/BSD trong lõi.
3. App chỉ phân phối **code**. Model/dataset do end-user tự tải từ nguồn chính chủ, không rehost. Image NGC được pull từ nvcr.io trên máy end-user.
4. Asset NC (vd CC BY-NC): team được dùng cho R&D. Trong app, user tự chọn thì hiện badge ⛔ + bước xác nhận. **Template mặc định chỉ dùng asset được phép dùng thương mại.** Không bundle asset NC, không publish weights train trên data NC làm mặc định.
5. Release: ký số, SBOM, THIRD_PARTY_LICENSES, SECURITY.md; không telemetry mặc định.
6. Nếu sau này muốn tương thích ComfyUI: làm gói custom node GPL ở **repo riêng**, giao tiếp qua HTTP.
