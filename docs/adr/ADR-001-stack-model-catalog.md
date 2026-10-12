# ADR-001 — NVIDIA stack, image & model catalog

- Trạng thái: **ACCEPTED** — 2026-10-12
- Nguồn: `.team/proposals/001-stack-model-catalog.md` (bao-research). Approve: bao-backend, bao-security. Smoke S-3 PASS.
- Chốt bởi: bao-pm

## Quyết định
| Vai trò | Thành phần |
|---|---|
| Train | `vlm-automodel:26.08-1` = FROM `nvcr.io/nvidia/nemo-automodel:26.08@digest` + chmod o+rX, chạy non-root, `--network none` |
| Train backend 2 | LLaMA-Factory (Apache-2.0), theo ADR-009 |
| Quantize | ModelOpt 0.47 (pip trong image, vì 26.08 chỉ có sẵn 0.44) |
| Serve / chạy thử | `vllm-openai:v0.31.0` (arm64 digest, mem-util ≤0.3; đã có bản vá GHSA-h3rc) |
| Eval | lmms-eval 0.7.3 trong container, `HF_HUB_OFFLINE=1` |

- Driver host giữ **580.142** (user chốt). Image CUDA mới chạy qua forward compatibility, nên **mỗi image phải probe PASS trên 580.142** trước khi đưa vào lock. Cập nhật DGX OS 580.159.03 là tuỳ chọn, user quyết thời điểm.
- Digest/SHA pin trong `docker/IMAGES.lock` và catalog. Đổi version phải qua `research-stack-verify` + `security-model-vetting`.

## Catalog model (dữ liệu, xem P001)
- Template mặc định: **Qwen3.5-2B@15852e8c**. CI/smoke: Qwen3.5-0.8B. Main: Qwen3.5-4B / 9B. Gemma 4 E2B/E4B được duyệt.
- 27B+: ẩn trong Simple / chỉ khi máy rảnh RAM. Nemotron-3-Nano-Omni: chỉ R&D (security CONDITIONAL).
- Dataset template: **CORD-v2@7f0115a4** (CC BY 4.0, kèm attribution).
- Cột `ram_gb` là ước lượng → hiệu chỉnh theo số đo S-3 (CUDA 8.1GiB peak step 0, 3–5GiB steady cho 0.8B LoRA) và số RAM host từ tester.
