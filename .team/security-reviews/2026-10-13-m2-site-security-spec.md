# M2-8 — Spec bảo mật cho site SaaS tĩnh + dashboard mới

Owner: bao-security · 2026-10-13 · Áp dụng cho: `site/` (M2-3/4/5), dashboard redesign (M2-6), QA (M2-7).
Mức độ: các mục **[BẮT BUỘC]** là điều kiện merge (security review theo ADR-004); **[NÊN]** là khuyến nghị.

## 0. Tóm tắt 1 dòng cho từng vai trò
- **frontend/cleancode (dựng site):** mọi tài nguyên (JS, CSS, font, icon, ảnh) nằm trong `site/` và được build vào `dist/`. Không `<script src="https://…">`, không Google Fonts, không analytics. Chạy `uv run python scripts/security/site_check.py site/dist` trước khi xin review.
- **design (copy + hình):** ảnh/illustration phải tự vẽ hoặc có license ghi rõ (CC0/CC-BY/Unsplash License…) và được ghi vào `site/ASSETS.md`. Không dùng logo NVIDIA. Không bịa số liệu, khách hàng hay testimonial.
- **tester (M2-7):** chạy `site_check.py` trong CI. Kiểm CSP bằng Playwright: console không có lỗi CSP, và tab Network chỉ có request tới origin của site.
- **PM/user:** cần chốt các mục ở §7 (U7–U10) trước khi publish trang chính sách.

## 1. Không tracker / không tài nguyên bên thứ ba [BẮT BUỘC]
- Không analytics (GA, Plausible, Umami, Vercel Analytics, Cloudflare Web Analytics, Hotjar…), không pixel, không chat widget, không A/B testing SaaS. Vì vậy cũng không có cookie banner: site không đặt cookie nào (§5).
- Không CDN bên thứ ba cho script, style, font hay ảnh. Không nhúng `<iframe>` YouTube/Twitter/Figma. Video demo thì self-host `.mp4`/`.webm` (ưu tiên `<video>` có poster, không autoplay kèm tiếng).
- **Được phép:** link điều hướng `<a href="https://…">` ra ngoài (GitHub, Hugging Face, docs NVIDIA). Link phải có `rel="noopener noreferrer"`; link ra ngoài mà không mở tab mới thì vẫn có `rel="noreferrer"`. Referrer-Policy cũng đã chặn (§2).
- Form "Liên hệ": **không** có form gửi dữ liệu (không Formspree, không Google Forms). Chỉ dùng `mailto:` hoặc link tới issue tracker. Lý do: site tĩnh không có backend; form qua bên thứ ba = gửi dữ liệu cá nhân cho bên thứ ba và phải viết lại chính sách quyền riêng tư.
- Search docs: dùng **Pagefind** (index tĩnh, chạy trong trình duyệt, MIT) hoặc search của Starlight (vốn là Pagefind). **Không** dùng Algolia DocSearch (gửi query ra ngoài).
- **npm deps của site:** áp `.npmrc` y như `frontend/` (`ignore-scripts=true`, `min-release-age=7`, `save-exact=true`). Lockfile phải được commit. Mọi package mới vào `configs/security/licenses.yaml` (scope `site`) **trước** khi commit (quy tắc ordering). Package nào cần postinstall (vd `sharp`) thì báo security trước, không tự bật script.

## 2. CSP và header [BẮT BUỘC]

Policy chuẩn cho site tĩnh (production):

```
default-src 'none';
script-src 'self' {hash-của-script-inline-nếu-có} {'wasm-unsafe-eval' CHỈ khi dùng Pagefind};
style-src 'self' {hash-của-style-inline-nếu-có};
img-src 'self' data:;
font-src 'self';
connect-src 'self';
media-src 'self';
manifest-src 'self';
worker-src 'self';
base-uri 'none';
form-action 'none';
frame-ancestors 'none';
object-src 'none';
upgrade-insecure-requests
```

Quy tắc:
- **Cấm** `'unsafe-inline'` trong `script-src` và **cấm** `'unsafe-eval'`. Script inline (vd script chống nháy theme của Starlight) phải được cho phép bằng hash `sha256-…` do build sinh ra. Astro có tính năng CSP sinh hash tự động (`security.csp`); frontend xác nhận nó có ở bản Astro đã pin và hoạt động với Starlight. Nếu không có thì đưa script inline ra file `.js`.
- `style-src`: ưu tiên `build.inlineStylesheets: 'never'` để mọi CSS là file. Nếu framework bắt buộc có thuộc tính `style="…"` (animation, Starlight), chỉ nới `style-src-attr 'unsafe-inline'`, **không** nới `style-src-elem`. Ghi rõ lý do trong PR.
- **Header vs meta:** nếu nơi host cho đặt header (Cloudflare Pages `_headers`, Netlify `_headers`, nginx) thì gửi CSP bằng header. `<meta http-equiv="Content-Security-Policy">` chỉ là dự phòng, vì meta **không** hỗ trợ `frame-ancestors`, `report-uri` hay `sandbox`. Chưa chốt host (U9) thì làm cả hai: meta trong layout, và file `site/public/_headers` mẫu.
- Header kèm theo:
  ```
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=(), usb=(), payment=(), browsing-topics=(), interest-cohort=()
  Cross-Origin-Opener-Policy: same-origin
  X-Frame-Options: DENY
  Strict-Transport-Security: max-age=31536000; includeSubDomains   (chỉ khi đã chạy HTTPS ổn định; không bật preload khi chưa hỏi user)
  ```
- **Animation trên landing:** không dùng thư viện cần `eval` (`new Function`, một số engine Lottie cũ). Lottie thì dùng `lottie-web` bản `light` hoặc chuyển sang CSS/SVG/WAAPI. Tôn trọng `prefers-reduced-motion`.
- **Dashboard (M2-6):** app 7870 hiện có `nosniff`, `X-Frame-Options DENY`, `Referrer-Policy no-referrer`, COOP, CORP và `no-store` (`src/vlm_pipeline/app/security.py:42`), **chưa có CSP**. Khi dashboard mới chạy ở route riêng, security sẽ thêm CSP (`default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'` tạm thời cho React Flow; `connect-src 'self'` để WebSocket same-origin chạy; `img-src 'self' data: blob:`; `frame-ancestors 'none'`; `object-src 'none'`; `base-uri 'none'`). Frontend báo mình khi có build ở route mới để test CSP trên đó. **Không** đổi dist 7870 khi user đang thử.

## 3. Font, icon, ảnh: chỉ self-host và license rõ ràng [BẮT BUỘC]

| Loại | Cho phép | Ghi chú |
|---|---|---|
| Icon | **Lucide** (ISC; một phần có nguồn từ Feather, MIT, nên giữ nguyên file LICENSE gốc). Phosphor (MIT), Tabler (MIT) cũng được | Bundle SVG vào build, không dùng icon font từ CDN. Lucide đã bỏ brand icon: logo GitHub/HF lấy từ press kit chính chủ, hoặc chỉ ghi chữ |
| Font | Font **SIL OFL-1.1** (Inter, Be Vietnam Pro, JetBrains Mono, IBM Plex…) hoặc Apache-2.0 | Lấy từ repo chính chủ hoặc `@fontsource/*` (npm, có lockfile). Kèm file `OFL.txt` trong `site/public/fonts/<family>/`. Tự subset thì **không** được giữ Reserved Font Name: đổi tên family khi subset nếu font có RFN (OFL §3) |
| Ảnh minh hoạ | Tự vẽ/tự chụp (Apache-2.0 theo repo); CC0; CC-BY-4.0 (phải ghi công); Unsplash License / Pexels License (không bắt buộc ghi công, mình vẫn ghi) | **Cấm:** ảnh không rõ nguồn, ảnh "free" lấy từ Google Images, ảnh có người/logo nhận diện được mà không có model/property release |
| Ảnh do AI sinh | Được, nếu công cụ cho phép dùng thương mại; ghi "minh hoạ do AI tạo" trong `ASSETS.md` | Không sinh ảnh giống logo/sản phẩm NVIDIA hay người thật |
| Screenshot app | Được (là sản phẩm của mình) | Dữ liệu trong ảnh phải là dữ liệu mẫu có license (vd CORD-v2 CC-BY-4.0, phải ghi công). Không chụp đường dẫn `/home/<user>`, token hay tên máy |

`site/ASSETS.md` (bắt buộc), mỗi asset 1 dòng: `đường dẫn | nguồn (URL) | tác giả | license | cần ghi công? | ngày lấy | sha256`. Security review theo file này, và trang "Giấy phép & ghi công" sinh từ nó.

## 4. Tên và thương hiệu NVIDIA [BẮT BUỘC]

Căn cứ:
- NVIDIA Open Model License (bản 24/10/2025) §5: license **không** cấp quyền dùng tên, nhãn hiệu hay logo NVIDIA, "except as required for reasonable and customary use in describing the origin of the Model" và để tái tạo file Notice.
- §3.2 yêu cầu ghi "Built on NVIDIA Cosmos" (trên website, UI, blog, trang giới thiệu hoặc tài liệu sản phẩm) khi phân phối hoặc cung cấp sản phẩm dùng Cosmos model, hoặc dùng Cosmos/output của nó để train hay fine-tune model.
- Không tìm thấy hướng dẫn công khai nào cho phép bên thứ ba dùng logo NVIDIA nếu không có thoả thuận đối tác (NPN/Inception). → **Không dùng logo.**

Quy tắc:
1. **Không** logo NVIDIA, logo "eye", màu xanh NVIDIA `#76B900` làm màu thương hiệu chính, hay badge kiểu "NVIDIA Partner" / "NVIDIA Certified" / "Powered by NVIDIA". Mình không có thoả thuận nào cho phép.
2. Tên sản phẩm của mình **không** chứa "NVIDIA", "NeMo", "Nemotron", "Cosmos", "DGX", "CUDA", "TensorRT" (vd không đặt "NeMo Studio"). Đây là quy tắc cho M2-1 khi đề xuất 3 tên.
3. **Được viết** dạng mô tả sự thật (nominative use): "Chạy trên NVIDIA NeMo AutoModel", "Tối ưu cho NVIDIA DGX Spark", "Dùng NVIDIA TensorRT Model Optimizer để quantize". Nhắc lần đầu thì viết đúng tên đầy đủ, không biến tấu ("Nvidia" → "NVIDIA").
4. **"Built on NVIDIA Cosmos"**: **chỉ** hiện khi sản phẩm thực sự cung cấp template dùng Cosmos (hiện Cosmos-Reason2-2B mới là ứng viên, gated, chưa phải template mặc định). Khi có thì đặt ở trang Giới thiệu, trang Template tương ứng và trang Giấy phép. Câu "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" thuộc file Notice khi **phân phối model**; app không phân phối weights (ADR-003, nhánh F) nên câu này chỉ ghi ở trang Giấy phép như thông tin.
5. **Không** viết câu ngụ ý NVIDIA bảo trợ, chứng nhận hay hợp tác ("được NVIDIA tin dùng", "đối tác NVIDIA", "giải pháp chính thức"). Footer mọi trang ghi: *"NVIDIA, NeMo, Nemotron, Cosmos, DGX, CUDA và TensorRT là nhãn hiệu hoặc nhãn hiệu đã đăng ký của NVIDIA Corporation tại Hoa Kỳ và các quốc gia khác. Dự án này độc lập, không liên kết với và không được NVIDIA bảo trợ."* Áp tương tự cho Hugging Face, Qwen/Alibaba và Google/Gemma nếu được nhắc tới.
6. Ảnh GPU hay phần cứng: không dùng ảnh sản phẩm của NVIDIA (ảnh press kit có điều khoản riêng). Vẽ sơ đồ trừu tượng.
7. Benchmark: chỉ số lấy từ `benchmarks/` hoặc ADR, có ghi điều kiện đo (model@SHA, image, phần cứng, ngày). Không so sánh với sản phẩm khác nếu không đo cùng điều kiện (M2-7 kiểm).

## 5. Sự thật để viết trang chính sách (đã đối chiếu code 2026-10-13)

| Mệnh đề | Bằng chứng |
|---|---|
| App chỉ nghe trên `127.0.0.1` (máy local), không mở ra mạng | `src/vlm_pipeline/app/server.py:1,22` |
| App đặt **đúng 1 cookie**: `vlm_session` (HttpOnly, SameSite=Strict, Path=/, không Max-Age nên bị xoá khi đóng trình duyệt). Mục đích duy nhất là xác thực phiên local, chặn web khác điều khiển app (CSRF/DNS-rebinding) | `security.py:198` |
| CSRF token chỉ nằm trong bộ nhớ tab, không lưu | `frontend/src/api/client.ts:4` |
| localStorage có 2 key: `vlm.ui.mode` (chế độ Đơn giản/Nâng cao) và `vlm.draft.v1` (bản nháp workflow chưa lưu). Không bao giờ rời trình duyệt | `frontend/src/store/ui.ts:4`, `frontend/src/lib/draft.ts:6` |
| Job train/eval chạy trong container `--network none`; `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `WANDB_MODE=offline`, Hugging Face ở chế độ offline | `src/vlm_pipeline/common/jobrun.py:69-71,99` |
| Token (HF_TOKEN, NGC_*, WANDB_API_KEY…) bị chặn, không bao giờ truyền vào job | `jobrun.py:73` và regex `_SECRET_NAME` |
| Truy cập Internet chỉ xảy ra khi **cài đặt/chuẩn bị**: pull image từ NVIDIA NGC (`nvcr.io`), tải model/dataset từ Hugging Face Hub, tải package PyPI/npm khi build. Khi đó máy người dùng kết nối trực tiếp tới các dịch vụ này và chính sách của họ được áp dụng | IMAGES.lock, configs/models/pinned |
| Access log của app tắt; log nội bộ redact query string | `security.py` (logger `vlm_pipeline.access`) |
| Dữ liệu, checkpoint, log, `outputs/app.db` nằm trên máy người dùng; xoá thủ công qua panel Lưu trữ hoặc xoá thư mục | ADR-005, `app/storage.py` |
| Không telemetry mặc định | ADR-003 §5 |
| **Site tĩnh:** không cookie, không localStorage (trừ theme sáng/tối nếu Starlight dùng → khai báo trong trang Cookie), không analytics. Nhà cung cấp hosting có thể ghi log truy cập kỹ thuật (IP, user-agent, URL, thời gian) | chờ U9 |

Nội dung trang chính sách đặt tại `docs/legal/*.md` (nguồn sự thật, tiếng Việt). `SECURITY.md` ở gốc repo (tiếng Anh). Site import từ đó, **không chép tay** để tránh lệch. Mỗi khi code thay đổi một mệnh đề ở bảng trên (thêm cookie, thêm kết nối mạng, thêm telemetry), PR đó **phải** sửa trang chính sách. Mục này nằm trong checklist security review.

## 6. Kiểm tự động: `uv run python scripts/security/site_check.py <dist> [--publish]` [BẮT BUỘC trong CI của M2-7]
Exit ≠ 0 nếu gặp:
- tài nguyên tải từ origin khác (`script/link/img/source/video/audio/iframe/embed/object src|href|srcset|poster`, `@import`/`url()` trong CSS, `@font-face` từ xa);
- domain tracker đã biết (google-analytics, googletagmanager, plausible, umami, segment, hotjar, mixpanel, posthog, clarity, facebook, doubleclick, cloudflareinsights, vercel-insights…);
- `<script>` inline mà CSP meta không có hash khớp, hoặc CSP có `'unsafe-inline'` ở script-src / `'unsafe-eval'`;
- trang HTML không có CSP meta (trừ khi truyền `--csp-from-headers <file _headers>`);
- `target="_blank"` không có `noopener`;
- placeholder `{{TODO_USER` còn sót (trang chính sách chưa được user chốt);
- file font (`.woff2/.woff/.ttf/.otf`) mà thư mục font không có file license (`OFL*.txt|LICENSE*`);
- file có tên/đường dẫn chứa `nvidia` + `logo`.

## 7. Cần user/PM quyết định (đưa vào danh sách U)
- **U7** Đơn vị/người vận hành và email liên hệ ghi trên trang Điều khoản/Chính sách bảo mật, cùng email nhận báo lỗ hổng (`security@…`). Nếu chưa có thì dùng GitHub Security Advisories (private vulnerability reporting) của repo public, nhưng hiện chưa có repo public (cấm push).
- **U8** Luật áp dụng / quốc gia (mặc định đề xuất: Việt Nam; Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân, Luật Bảo vệ dữ liệu cá nhân 2025 có hiệu lực 01/01/2026). Site không thu dữ liệu cá nhân nên nghĩa vụ chủ yếu là minh bạch.
- **U9** Nơi host site (GitHub Pages / Cloudflare Pages / Netlify / tự host). Ảnh hưởng: có đặt header được không, và log của host (phải nêu tên trong Chính sách bảo mật).
- **U10** Ngày hiệu lực cho các trang chính sách (đặt khi publish).

## 8. Checklist security review PR site/dashboard
- [ ] `site_check.py site/dist --publish` pass (trước khi publish) (dán output vào PR)
- [ ] Lockfile và `.npmrc` (ignore-scripts, min-release-age) giữ nguyên; package mới có trong `licenses.yaml`
- [ ] `ASSETS.md` có đủ asset mới (font, icon, ảnh, video)
- [ ] Không logo NVIDIA hay ngụ ý bảo trợ; footer có dòng nhãn hiệu
- [ ] Số liệu có nguồn trong `benchmarks/` hoặc ADR
- [ ] Mệnh đề chính sách (§5) vẫn đúng với code của PR
- [ ] Playwright: 0 lỗi CSP trong console; mọi request cùng origin
