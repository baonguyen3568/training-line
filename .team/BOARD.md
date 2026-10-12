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
| R-1 | Research: khảo sát app node-graph/no-code ML có sẵn (ComfyUI, LLaMA-Factory/LlamaBoard, NVIDIA NeMo Customizer/AI Workbench, Flowise/Langflow, React Flow, LiteGraph…) → nên fork/extend hay tự xây | bao-research | DOING |
| R-2 | Proposal 009: Kiến trúc node-graph app (node type system, executor/scheduler, graph→job, template workflow) | bao-backend + bao-frontend | DOING |
| R-3 | UX cho no-tech user: canvas, node library, template gallery, validate trước khi chạy, thông báo lỗi dễ hiểu | bao-design | DOING |
| R-4 | Mỗi người rà proposal của mình theo tầm nhìn mới, ghi mục "Ảnh hưởng của RESET" | all | TODO |

## Phase 1 — Kiến trúc (research dẫn dắt)
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P1-1 | Proposal 001: stack NVIDIA + base model VLM + phiên bản container arm64 | bao-research | DOING |
| P1-2 | Proposal 002: eval/benchmark protocol (bộ benchmark, metric, harness) | bao-research + bao-tester | DOING |
| P1-3 | Proposal 003: security baseline (deps pinning, secrets, license) | bao-security | REVIEW |
| P1-4 | Proposal 004: code standards + repo layout + CI local | bao-cleancode | DONE → ADR-004 |
| P1-5 | Proposal 005: storage/retention policy (checkpoints, HF cache, datasets) | bao-cleanup | DONE → ADR-005 |
| P1-6 | Proposal 006: dashboard scope + design system | bao-frontend + bao-design | DOING |
| P1-8 | Proposal 007: reliability baseline (preflight, UMA memory watchdog, env fingerprint, converter round-trip test, bug workflow) | bao-fixcode | DOING |
| P1-9 | Spike NVIDIA containers trên GB10 (AutoModel 26.08 VLM, ModelOpt, vLLM) + Proposal 008: backend runtime (host orchestrator ↔ container worker, job runner, GPU lock/memguard, RunManifest) | bao-backend | PAUSED (RESET: spike hoãn; P008 giữ hướng, chờ ghép vào P009) |
| P1-7 | PM tổng hợp → ADR-001..00N, chốt với user | bao-pm | TODO |

## Phase 2 — Build pipeline (sau khi ADR-001 chốt)
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
