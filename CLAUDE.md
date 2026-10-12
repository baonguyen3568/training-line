# Trainning-model — VLM Training Platform (NVIDIA stack)

## Mục tiêu
Một source hoàn chỉnh để **fine-tune / train Vision-Language Model (VLM) mới nhất** bằng stack NVIDIA mới nhất,
pipeline end-to-end:

```
data (thu thập → curate → format) → train (SFT / LoRA / RL) → validate
  → quantize (FP8 / NVFP4 / INT4-AWQ) → benchmark chuẩn (accuracy + latency/throughput) → serve/export
```

Ưu tiên tận dụng tối đa tooling chính chủ NVIDIA (chỉ chốt sau khi bao-research xác minh phiên bản hiện tại):
- Train: NeMo AutoModel (HF-native), Megatron-Bridge (Nemotron VL), NeMo-RL (GRPO/DPO)
- Data: NeMo Curator (multimodal), Energon/WebDataset
- Quantize: TensorRT Model Optimizer (`nvidia-modelopt`) — PTQ/QAT, FP8, NVFP4
- Eval: NeMo Evaluator, VLMEvalKit / lmms-eval (MMMU, MMBench, DocVQA, ChartQA, OCRBench, MathVista, Video-MME…)
- Serve: TensorRT-LLM / vLLM / NIM
- DGX Spark playbooks: https://build.nvidia.com/spark/vlm-finetuning

## Phần cứng (QUAN TRỌNG)
- 1x **NVIDIA GB10 (DGX Spark)**, **aarch64**, 128GB **unified memory** (CPU+GPU dùng chung), CUDA 13, driver 580.
- Máy **dùng chung** với service khác (docker: kafka, elastic, banking-rag, comfy-ui…). RAM đang dùng sẵn ~70GB+.
- Mọi image/wheel phải hỗ trợ **arm64**. Ưu tiên container NGC (`nvcr.io/nvidia/...`) bản arm64.
- **GPU lock**: chỉ 1 job nặng (train/quantize/benchmark) tại 1 thời điểm. Trước khi chạy: kiểm tra/tạo
  `.team/GPU_LOCK` (ghi: ai, job gì, bắt đầu lúc nào, PID). Xong thì xoá. Không tự ý giết job của người khác.

## Đội ngũ (xem `.team/ROSTER.md`)
PM = `bao-trainning-1` (điều phối, giải quyết mâu thuẫn, chốt quyết định cùng user).
9 thành viên chạy trong tmux `bao-trainning-2..10`, mỗi người một chuyên môn.

## Giao thức trao đổi
1. **Nhắn trực tiếp**: dùng tool `SendMessage` với tên session (vd `bao-backend`). PM = `trainning-model-6a`. Dùng `ListAgents` để xem ai online.
2. **Ghi lại bền vững**: mọi trao đổi quan trọng → append 1 dòng vào `.team/log/YYYY-MM-DD.md`
   format: `HH:MM [from → to] nội dung ngắn`.
3. **Task board**: `.team/BOARD.md` — nhận task bằng cách ghi tên mình vào cột owner. Không làm trùng task.
4. **Đề xuất kiến trúc / thay đổi lớn**: viết file `.team/proposals/NNN-tieu-de.md` → nhắn các bên liên quan
   + `bao-pm` để review. Quyết định cuối ghi vào `docs/adr/` (PM chốt, có thể hỏi user).
5. **Mâu thuẫn**: 2 bên trao đổi trước; không thống nhất được → ghi `.team/CONFLICTS.md` (lập luận 2 phía) và nhắn `bao-pm`.
6. Tôn trọng chuyên môn: review chéo là bắt buộc cho code merge (ít nhất 1 người khác ngoài tác giả).

## Quy tắc an toàn (bắt buộc với mọi agent)
- Chỉ ghi/xoá file **bên trong** `/home/ntcai/Trainning-model` (trừ cache pip/uv/HF/docker image pull).
- **KHÔNG**: `docker system prune`, xoá/stop container/image không thuộc dự án, `git push`, sửa `~/.claude/settings.json`,
  `sudo` thay đổi hệ thống, `rm -rf` ngoài project. Cần những việc đó → hỏi PM, PM hỏi user.
- Không commit secrets (HF token, NGC key, W&B key) — dùng `.env` (đã gitignore).
- Download model/dataset lớn (>20GB) → báo PM trước (disk còn ~2TB nhưng dùng chung).
- Container dự án đặt tên prefix `vlm-` để dễ nhận diện/dọn dẹp.

## Cấu trúc thư mục
```
src/vlm_pipeline/{data,train,eval,quantize,serve,common}   # code Python chính
configs/{data,train,eval,quantize}                          # YAML recipes
scripts/                                                    # CLI/entrypoints, scripts/team = điều phối team
docker/                                                     # Dockerfile/compose (arm64)
tests/                                                      # pytest
benchmarks/                                                 # kết quả benchmark (json/md)
frontend/  design/                                          # dashboard theo dõi + design system
docs/adr/                                                   # Architecture Decision Records
.team/                                                      # phối hợp team (board, log, knowledge, proposals)
.claude/skills/                                             # skill dự án (mỗi role tự tạo skill của mình)
data/ outputs/                                              # gitignored
```

## Quy ước code
- Python 3.12, quản lý bằng `uv` (pyproject.toml). Ruff + mypy + pytest.
- Config bằng YAML + pydantic; mọi bước pipeline chạy được bằng 1 lệnh CLI và reproducible (seed, version pin).
- Tiếng Việt khi nói với user / trong .team; code, comment, commit message bằng tiếng Anh.
