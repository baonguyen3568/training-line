# Task Board
Trạng thái: TODO | DOING | REVIEW | DONE | BLOCKED. Nhận task = ghi tên vào Owner.

## Phase 0 — Onboarding (mọi người, song song)
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P0-1 | Mỗi role: tổng hợp kiến thức chuyên ngành → `.team/knowledge/<role>.md` | all | TODO design✓ fixcode✓ tester✓ frontend✓ backend✓ security✓ research✓ |
| P0-2 | Mỗi role: tạo skill riêng trong `.claude/skills/<role>-*/SKILL.md` | all | TODO design✓ fixcode✓ tester✓ frontend✓ backend✓ security✓ research✓ |
| P0-3 | Tự giới thiệu với đồng đội liên quan qua SendMessage | all | TODO design✓ fixcode✓ tester✓ frontend✓ backend✓ security✓ research✓ |

## ⚠️ RESET 2026-10-12 — Tầm nhìn mới (xem CLAUDE.md "Tầm nhìn sản phẩm")
Sản phẩm = app no-code node-graph kéo thả (kiểu ComfyUI/Unreal Blueprint) cho toàn bộ vòng đời VLM, user không biết code.
Giai đoạn hiện tại: CHỈ reset + setup dự án. KHÔNG chạy train/spike GPU, KHÔNG tải thêm image/model lớn cho tới khi PM mở lại.
| ID | Task | Owner | Status |
|----|------|-------|--------|
| R-1 | Research: khảo sát app node-graph/no-code ML có sẵn (ComfyUI, LLaMA-Factory/LlamaBoard, NVIDIA NeMo Customizer/AI Workbench, Flowise/Langflow, React Flow, LiteGraph…) → nên fork/extend hay tự xây | bao-research | REVIEW (.team/proposals/R1-node-graph-survey.md) |
| R-2 | Proposal 009: Kiến trúc node-graph app (node type system, executor/scheduler, graph→job, template workflow) | bao-backend + bao-frontend | DOING (ADR-009 accepted; chi tiết) |
| R-3 | UX cho no-tech user: canvas, node library, template gallery, validate trước khi chạy, thông báo lỗi dễ hiểu | bao-design | REVIEW (P010) |
| R-4 | Mỗi người rà proposal của mình theo tầm nhìn mới, ghi mục "Ảnh hưởng của RESET" | all | TODO cleanup✓ tester✓ fixcode✓ frontend✓ security✓ (P003 §3b, D6-final) |

## Phase 1 — Kiến trúc (research dẫn dắt)
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P1-1 | Proposal 001: stack NVIDIA + base model VLM + phiên bản container arm64 | bao-research | REVIEW (001-stack-model-catalog.md) |
| P1-2 | Proposal 002: eval/benchmark protocol (bộ benchmark, metric, harness) | bao-research + bao-tester | REVIEW (Phần A research xong) |
| P1-3 | Proposal 003: security baseline (deps pinning, secrets, license) | bao-security | REVIEW |
| P1-4 | Proposal 004: code standards + repo layout + CI local | bao-cleancode | DONE → ADR-004 |
| P1-5 | Proposal 005: storage/retention policy (checkpoints, HF cache, datasets) | bao-cleanup | DONE → ADR-005 |
| P1-6 | Proposal 006: dashboard scope + design system | bao-frontend + bao-design | DOING |
| P1-8 | Proposal 007: reliability baseline (preflight, UMA memory watchdog, env fingerprint, converter round-trip test, bug workflow) | bao-fixcode | REVIEW |
| P1-9 | Spike NVIDIA containers trên GB10 (AutoModel 26.08 VLM, ModelOpt, vLLM) + Proposal 008: backend runtime (host orchestrator ↔ container worker, job runner, GPU lock/memguard, RunManifest) | bao-backend | PAUSED (RESET: spike hoãn; P008 giữ hướng, chờ ghép vào P009) |
| P1-7 | PM tổng hợp → ADR-001..00N, chốt với user | bao-pm | TODO |

## 🎨 MILESTONE M2 — SaaS site + Dashboard kiểu Unreal (user yêu cầu 2026-10-13)
Dùng skill `impeccable` (~/.claude/skills/impeccable/SKILL.md). Session nào chưa thấy skill thì Read trực tiếp SKILL.md + reference/ và chạy
`~/.claude/skills/impeccable/scripts/impeccable context`. Mode: landing/PR = Persuade; docs/giới thiệu/chính sách = Read; dashboard = Operate.
YÊU CẦU CỦA USER:
- Site SaaS chuẩn doanh nghiệp, 20–30 trang: landing page (PR, quảng bá dịch vụ, animation bắt mắt), giới thiệu, tính năng, use case, template,
  docs (nhiều trang), chính sách (bảo mật, điều khoản, cookie, giấy phép/ghi công), liên hệ, changelog, roadmap, FAQ, 404…
- Tiếng Việt CÓ DẤU, rõ ràng, đầy đủ. Không nhồi chữ rác ngoài nội dung chính: phần giải thích phụ → icon "?" + popover.
- Trang ngoài (landing/docs/giới thiệu) được dài; landing có animation để PR.
- Dashboard (editor) đẹp, clean như Unreal Engine: thấy TOÀN BỘ thanh công cụ trong 1 màn hình, icon-first + tooltip/popover,
  KHÔNG chữ nếu không thật cần, không ghi chú nhỏ thừa. Chuẩn doanh nghiệp.
RÀNG BUỘC CỨNG (PM):
- KHÔNG bịa: không testimonial/logo khách hàng/số liệu giả; mọi con số lấy từ benchmark/ADR thật. Không giả danh NVIDIA: chỉ "Built on NVIDIA …"
  theo license; không dùng logo NVIDIA trái hướng dẫn thương hiệu (security kiểm).
- Sản phẩm miễn phí, mã nguồn Apache-2.0 (ADR-003) → trang "Bảng giá" = "Miễn phí & mã nguồn mở", không bịa gói trả phí.
- Không tracker/analytics/CDN bên thứ 3; font self-host (security).
- Không đổi dist 7870 khi user đang thử app; build dashboard mới ở nhánh/route riêng tới khi user duyệt.
| ID | Task | Owner | Status |
|----|------|-------|--------|
| G-1 | GAP: Panel Dung lượng UI (node-cache + set-aside + checkpoint retention, dry-run/apply, PERMANENT_LOSS confirm) | bao-frontend + bao-cleanup (API) | TODO |
| G-2 | GAP: retention chạy theo lịch trong app (hiện chỉ CLI) | bao-cleanup + bao-backend | TODO |
| G-3 | GAP: nút "Sửa nhanh" theo error_catalog actions trên UI | bao-frontend + bao-fixcode | TODO |
| M2-1 | Research phong cách (Linear, Vercel, Modal, Replicate, Hugging Face, Unreal Editor, Figma, ComfyUI…) + `impeccable init`/`shape` → PRODUCT.md, DESIGN.md, 2–3 hướng moodboard + đề xuất 3 tên sản phẩm cho user chọn | bao-design | TODO |
| M2-2 | Sitemap 20–30 trang + IA + copy tiếng Việt (nguồn sự thật: ADR, P001/P002, benchmarks) | bao-design (copy) + bao-research (nội dung kỹ thuật docs) + bao-cleanup (docs vận hành/lưu trữ) | TODO |
| M2-3 | Chọn stack site tĩnh (đề xuất: Astro + MDX, docs Starlight; MIT) trong `site/`, chung design tokens với app | bao-frontend + bao-cleancode | IN PROGRESS (stack chốt 10-13: Astro 7.3 + Starlight 0.42, MDX; frontend dựng khung; cleancode: gate check.sh + check `sources`) |
| M2-4 | Build landing + trang marketing + animation (prefers-reduced-motion) | bao-frontend (+ bao-fixcode hỗ trợ dựng trang) | TODO |
| M2-5 | Build docs + chính sách + trang phụ | bao-cleancode (dựng) + bao-design (review) | IN PROGRESS: 5 trang chính sách import từ docs/legal (chờ security review); tiếp: docs + trang phụ |
| M2-6 | Redesign dashboard kiểu Unreal: dock panels, toolbar icon-first, tooltip/popover, 1 màn hình, dark-first | bao-design (direction) + bao-frontend | TODO |
| M2-7 | QA: link checker, a11y (WCAG AA), visual regression 3 viewport, Lighthouse, kiểm "không bịa số" | bao-tester | IN PROGRESS (P011 draft; G1 check_site_numbers.py xong report-only) |
| M2-8 | Security: CSP, không tracker, license ảnh/font/icon, trademark NVIDIA, trang chính sách đúng thực tế | bao-security | REVIEW (spec + gate site_check + docs/legal + SECURITY.md xong; chờ U7–U10 của user, fix CSP Starlight của frontend) |

## 🎯 MILESTONE M1 — "Kéo thả → nối → Chạy thật" (ƯU TIÊN SỐ 1, user phản hồi 2026-10-12: app còn thô, chưa tạo/nối pipeline được)
Definition of Done: user mở app → tạo workflow TRỐNG → kéo node từ thư viện vào canvas → nối dây → bấm Chạy →
backend thật thực thi graph (FakeWorker cho node nhanh, job container thật cho 1 node train nhỏ) → thấy tiến độ/kết quả
từng node trên canvas → lưu/mở lại file workflow. Template chỉ là workflow dựng sẵn bằng chính cơ chế này.
Quy tắc tốc độ cho M1: code UI/core chỉ cần 1 reviewer (cleancode HOẶC người liên quan). Security chỉ review bắt buộc phần
Docker/deps/network/secrets/xoá file/plugin. Không viết proposal mới cho M1, chỉ làm theo P009/P010 đã có.
| ID | Task | Owner | Status |
|----|------|-------|--------|
| M1-1 | `nodes/`: NodeSpec + registry cho ~8 node v1: Dataset (CORD-v2 / thư mục ảnh), Model (từ catalog), Fine-tune, Evaluate, Chạy thử (inference), Quantize FP8, Benchmark, Export | bao-cleancode + bao-research (params từ recipe) | DONE (12 node + catalog: f1eb045, 4b12dd7, 7f0e0ae, 4739bd6, afdc074; research soát OK, i18n gate xanh) |
| M1-2 | `graph/`: validate (type/cycle/thiếu input) + executor (topo, cache fingerprint, chạy node → job, event stream, cancel/resume cơ bản) | bao-fixcode (dùng spec xfail của tester) | DONE (e6da8a1 + 2dd806a): validate + cache + executor; 60 spec + 9 test; review cleancode |
| M1-3 | `app/`: FastAPI GET /api/nodes, POST /api/graphs/validate, POST /api/runs, GET runs/metrics, WS events, auth token (D13); FakeWorker | bao-backend | REVIEW (ec726ac: API + D13 security live; cancel + container worker chờ executor M1-2) |
| M1-4 | Frontend: thư viện node (sidebar, tìm kiếm, kéo thả vào canvas), workflow trống, nối dây kéo-thả, xoá/nhân bản node, lưu/mở file workflow (.json), trạng thái chạy từng node trên canvas, nối API thật | bao-frontend | DONE (design APPROVE m1-review-3; còn: mock từ REGISTRY snapshot) |
| M1-5 | UX thư viện node + trạng thái node khi chạy (spec nhanh, review UI) | bao-design | DONE (review r3 APPROVE) |
| M1-6 | E2E Playwright: tạo trống → kéo 3 node → nối → chạy FakeWorker → thấy kết quả; chuyển spec xfail graph/executor sang pass | bao-tester | DONE (8009706: E2E PASS cả mock và app thật + FakeWorker; spec graph/executor chạy thật) |
| M1-6b | E2E kịch bản train thật nhỏ (5 step, @gpu, GPU lock + memguard + worktree sạch) khi có container Worker + 26.08-3 | bao-tester | TODO (chờ backend) |
| M1-7 | Review security tối thiểu cho app/ (auth, CORS/Host, WS) | bao-security | DONE (app/security.py ec726ac + fix d87595b; D13 spec 29/29) |
| M1-8 | Node cache retention + dọn worktree; hỗ trợ backend FakeWorker fixtures | bao-cleanup | REVIEW→DONE phần code: worktrees 42dc8fc, testing+runs_index aba6332, index+cache_retention d162093, lease acadb6f, release_all_except e40d180 (fixcode/security/backend approve). Còn: backend nối app (startup release, panel Dung lượng, DISK-003) |
S-9 (patch 26.08-2 → P-1/P-2) chạy song song, không chặn M1.

## Phase 2 — MỞ 2026-10-12 (user): scaffold + GPU smoke nhỏ. Tải >20GB phải hỏi PM. Mọi job GPU qua GPU_LOCK + preflight.
| ID | Task | Owner | Status |
|----|------|-------|--------|
| S-1 | Repo scaffold: pyproject/uv, ruff/mypy/import-linter, pre-commit(prek), gitleaks, scripts/check.sh, LICENSE Apache-2.0 | bao-cleancode | DONE (5ee7729, e85281a) |
| S-2 | Runtime: docker/ (image dẫn xuất vlm-automodel từ 26.08@digest, non-root), IMAGES.lock, common/gpu_lock.py, job runner skeleton | bao-backend | DONE (gpu_lock, i18n, runinfo, jobrun, schemas, container check; 26.08-2 recipe chờ patch) |
| S-3 | GPU smoke: LoRA 20 step Qwen3.5-0.8B trên 26.08 + đo RAM (tester song song) → hiệu chỉnh catalog P001 | bao-backend + bao-tester + bao-research | DOING (tester: đo RAM xong → benchmarks/perf/s3-*.json) |
| S-4 | common/memguard.py + preflight (disk/lock) | bao-fixcode + bao-cleanup | DONE (8029ab6, 236252e, 69f5373) |
| S-5 | frontend/ scaffold theo checklist 12 bước (Vite+React+@xyflow/react, tokens, i18n VI/EN) — vertical slice gallery→validate overlay | bao-frontend + bao-design | DONE (64be07f; cleancode+security approve; design review UI còn mở) |
| S-6 | P001 cập nhật default image 26.08 → ADR-001 | bao-research | DONE (backend+security APPROVED) → PM chốt ADR-001 |
| S-7 | Security review mọi PR S-1..S-5; xác nhận CORD-v2; review image dẫn xuất | bao-security | DOING (CORD-v2 ✓, Dockerfile vlm-automodel ✓, S-1 ✓, S-4 cleanup+fixcode ✓, retention ✓, S-5 frontend ✓ + CSRF/WS D13, licenses.yaml npm ✓, P001 ✓, NVIDIA models: Qwen3-VL-2B ✓, Parse-v1.1 ✓ có điều kiện, Cosmos chờ gate) |
| S-9 | Tự vá AutoModel (normalize + tied emb) → image 26.08-2 + regression test; so Parse v1.1 vs 2.0 | bao-fixcode + bao-backend + bao-tester + bao-research | DOING (patch 0001-0004 + 0003 v2 3a768ca → image 26.08-5; golden gate (b) PASS, BUG-001 CLOSED): critical path cho template mặc định (patch → 26.08-2 → parity → P-1 → P-2) |
| S-10 | FastAPI app/: POST /api/runs + FakeWorker (cho T1 usability + frontend bỏ mock) — sau critical path S-9 | bao-backend | TODO |
| TD-2 | Provenance dataset cho data.local_folder / data.split khi nối vào RealWorker: hash hiện không có @<sha40> → gap `dataset_revision` (9457abf) sẽ chặn gate. Chọn: split giữ `repo@rev#split=…`, local folder dùng content digest `sha256:<64hex>` được reproducibility_gaps chấp nhận | bao-backend | TODO (cleancode nêu khi review 9457abf) |
| TD-1 | Tech debt: KV cache for the Parse decoder (0003 tắt cache → generate O(n²)); trước tiên backend ghi thời gian generate đo ở golden gate (b) 26.08-4 vào VENDORED §0003, rồi quyết định có cần làm/đẩy upstream không | bao-fixcode | DEFERRED (đo: 1024 tok = 24 s, RAM phẳng 2.21 GiB → giữ no-cache cho eval v1; mở lại khi trang >~2k token hoặc eval vượt ngân sách) |
| S-8 | Gom test xfail hiện có vào tests/ theo layout ADR-004 | bao-tester | DONE (cleancode APPROVE sau fix; 103 passed, 217 xfailed) |

## Phase 2 (cũ) — Build pipeline
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P2-1 | pyproject + uv + Dockerfile arm64 (NGC base) | | TODO |
| P2-2 | Data module: download/curate/format (LLaVA/HF chat JSONL, Energon) | | TODO |
| P2-3 | Train module: SFT + LoRA recipe (AutoModel / Megatron-Bridge) | | TODO |
| P2-4 | RL module: GRPO/DPO (NeMo-RL) | | TODO |
| P2-5 | Quantize module: ModelOpt FP8 / NVFP4 / AWQ + export | | TODO |
| P2-6 | Eval module: accuracy benchmarks + perf (latency/throughput/memory) | | TODO |
| P2-7 | Serve/export: TensorRT-LLM / vLLM | | TODO |
| P2-8 | Smoke test end-to-end trên model nhỏ + dataset mini | | TODO |
| P2-9 | Dashboard v1 | | TODO |
