# Task Board
Trạng thái: TODO | DOING | REVIEW | DONE | BLOCKED. Nhận task = ghi tên vào Owner.

## Phase 0 — Onboarding (mọi người, song song)
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P0-1 | Mỗi role: tổng hợp kiến thức chuyên ngành → `.team/knowledge/<role>.md` | all | TODO |
| P0-2 | Mỗi role: tạo skill riêng trong `.claude/skills/<role>-*/SKILL.md` | all | TODO |
| P0-3 | Tự giới thiệu với đồng đội liên quan qua SendMessage | all | TODO |

## Phase 1 — Kiến trúc (research dẫn dắt)
| ID | Task | Owner | Status |
|----|------|-------|--------|
| P1-1 | Proposal 001: stack NVIDIA + base model VLM + phiên bản container arm64 | bao-research | TODO |
| P1-2 | Proposal 002: eval/benchmark protocol (bộ benchmark, metric, harness) | bao-research + bao-tester | TODO |
| P1-3 | Proposal 003: security baseline (deps pinning, secrets, license) | bao-security | TODO |
| P1-4 | Proposal 004: code standards + repo layout + CI local | bao-cleancode | TODO |
| P1-5 | Proposal 005: storage/retention policy (checkpoints, HF cache, datasets) | bao-cleanup | TODO |
| P1-6 | Proposal 006: dashboard scope + design system | bao-frontend + bao-design | TODO |
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
