# Proposal 003 — Security baseline (supply chain, secrets, license, host safety)

- Tác giả: bao-security · Ngày: 2026-10-12 · Trạng thái: **DRAFT — chờ review**
- Reviewer cần: bao-backend (Docker/deps), bao-cleancode (P004 hooks), bao-research (model/license), bao-tester (tests), bao-cleanup (retention), bao-pm
- Nền tảng & nguồn: `.team/knowledge/security.md`. Skill thực thi: `security-model-vetting`, `security-pr-review`, `security-license-check`.

## 1. Bối cảnh & rủi ro chính
Máy dùng chung (DGX Spark GB10, aarch64, 128GB unified memory) đang chạy service của người khác, trong đó có banking-rag. Nhóm agent có quyền tương đương root qua group `docker`. Có 3 rủi ro cao nhất:
1. **RCE khi load model/data từ Hub** (pickle, `trust_remote_code`, `auto_map`, decoder ảnh/video).
2. **Container escape qua NVIDIA Container Toolkit**: host đang chạy **1.19.0, dính CVE-2026-24260** (bulletin NVIDIA 5850, fix 1.19.1; apt hiện có 1.20.1). Đã báo PM, user sẽ quyết lịch vá.
3. **Vi phạm license** dataset/model khi chưa chốt mục đích sử dụng.

## 2. Quyết định đề xuất (D1–D9)

### D1. Container images
- Chỉ lấy image từ `nvcr.io/nvidia/*`. Image khác phải được security duyệt và ghi lý do.
- Mọi reference trỏ tới digest **linux/arm64** `@sha256:`, tag ghi trong comment. File lock là `docker/IMAGES.lock` (TSV: `name  tag  arm64_digest  index_digest  date  reviewer  source_url`).
- Image đầu tiên được duyệt: `nvcr.io/nvidia/nemo-automodel:26.08`, arm64 `sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a`.
- Chưa vá CVE-2026-24260 thì **cấm chạy image GPU bên thứ 3** với `--gpus`.
- Trivy scan khi thêm image mới hoặc đổi digest, chỉ để báo cáo (image NGC luôn có CVE OS). Chặn khi có CRITICAL đã có bản fix ở tầng mình kiểm soát (pip layer).

### D2. Runtime hardening container `vlm-*` (bắt buộc)
`--user 1000:1000`, `--security-opt no-new-privileges`, `--cap-drop ALL` (thêm lại cap cụ thể khi có log lỗi làm bằng chứng), port `127.0.0.1:` (dải dashboard 7870-7879), `--memory` (mặc định 48g, chỉ chặn RSS CPU, nên vẫn cần memguard CUDA trong code), `--shm-size` thay cho `--ipc=host`, `--env-file .env`.
**Cấm**: `--privileged`, `--network host`, `--pid host`, mount `/`, `$HOME`, `~/.ssh`, `/var/run/docker.sock` (ngoại lệ duy nhất là container trivy mount `:ro` để scan), file `~/.cache/huggingface/token`, `LD_PRELOAD`. Name prefix `vlm-` + label `project=vlm-training`.

### D3. Model weights
- Chỉ lấy từ org chính chủ. Revision là **commit SHA 40-hex**, ghi trong config (không dùng `main`).
- **Safetensors-only**: `use_safetensors=True`, tải bằng `allow_patterns` loại `*.bin/*.pt/*.pth/*.ckpt/*.pkl`.
- `trust_remote_code=False` mặc định toàn cục. Chỉ bật `True` khi repo+SHA nằm trong `configs/security/remote_code_allowlist.yaml` (security đã đọc toàn bộ `.py` và hash từng file). Nếu `transformers` có class native thì **bắt buộc** dùng native.
- Lần load đầu tiên của model mới chạy với `--network none` + `HF_HUB_OFFLINE=1`.
- `torch.load`/`pickle.load` bị chặn bằng ruff banned-api (P004). Dùng `vlm_pipeline.common.safe_load` (`weights_only=True`, assert torch ≥ 2.6). Chỉ dùng cho checkpoint nội bộ.
- Serving: vLLM/TRT-LLM phải ở bản ≥ bản vá các CVE đã biết (vLLM ≥ 0.14.1 tối thiểu; bao-research xác minh bản mới nhất), có `--api-key`, bind localhost, tắt fetch media URL từ xa nếu không cần.

### D4. Python dependencies (đã đồng bộ với P004)
`uv.lock` được commit. Cài bằng `uv sync --locked`. `[tool.uv] exclude-newer = "7 days"` (cooldown). Muốn vá CVE gấp thì override từng package và ghi lý do trong commit. Index NVIDIA khai báo `[[tool.uv.index]] explicit = true` + `[tool.uv.sources]` (chống dependency confusion). Cấm `git+…@<branch>`. Wheel ngoài PyPI/NGC (fork, nightly cho GB10) phải gửi security URL + sha256 + commit + lý do **trước khi cài**. Package có sẵn trong container NGC thì không cài đè. `pip-audit` chạy trong `scripts/check.sh` ở chế độ non-blocking.

### D5. Secrets
- Biến chuẩn: `HF_TOKEN` (fine-grained, read-only), `NGC_API_KEY`, `WANDB_API_KEY`, `WANDB_MODE=offline` mặc định. Commit `.env.example` với value rỗng.
- Pre-commit: gitleaks v8.30.1 (freeze), `detect-private-key`, `check-added-large-files --maxkb=1000`, `check-merge-conflict`. Chạy `gitleaks dir .` **trước commit đầu tiên**.
- Không `ENV/ARG` chứa token trong Dockerfile, dùng `RUN --mount=type=secret`. Env fingerprint (P007 R4) dùng **allowlist tên biến** và redact value khớp `://.*@|hf_|nvapi-|sk-`.
- Khi lộ secret: **revoke trước**, sau đó mới dọn git, rồi báo PM. File bug `[SEC]` không chứa giá trị secret.

### D6. License — 2 nhánh, chờ user chọn
| | **Nhánh A: research nội bộ** | **Nhánh B: thương mại / phân phối** |
|---|---|---|
| Model base | Apache-2.0, MIT, NVIDIA Nemotron Open Model License, Gemma ToU, Llama Community; NC research license cũng được | Chỉ Apache-2.0 / MIT / Nemotron Open Model License (12/2025). Gemma/Llama phải được user duyệt riêng các nghĩa vụ (AUP, attribution, MAU). **Không** dùng NC |
| Dataset train | NC được phép (LLaVA-Instruct CC BY-NC, ShareGPT4V…) nếu không phân phối weights | Chỉ license cho phép thương mại. **Loại** mọi dữ liệu sinh từ GPT-4/4V/Gemini có ToS cấm train model cạnh tranh, NC, và re-upload không rõ nguồn |
| Phân phối weights fine-tune | Không | Được, kèm LICENSE/NOTICE của upstream |
| Benchmark | eval-only (cả 2 nhánh) | eval-only |
Dù chọn nhánh nào, mọi asset phải có entry trong `configs/security/licenses.yaml` **trước khi** xuất hiện trong config. Asset sẽ gặp vấn đề nếu chọn B: LLaVA-Instruct-150K, ShareGPT4V/ALLaVA, MedPix-VQA (re-upload, nguồn gốc NLM MedPix chưa rõ quyền), checkpoint NVIDIA dùng license cũ.

### D7. Dataset hygiene
Dataset chỉ nhận parquet/arrow/webdataset, **không chạy loading script**. Đặt `Image.MAX_IMAGE_PIXELS`, decode có timeout/giới hạn bộ nhớ. Ghi sha256 manifest cho từng shard sau curate. Có test chặn benchmark lọt vào train mix. Không đưa ảnh y tế/PII lên dashboard/W&B.

### D8. Host chung & mạng
Không sửa daemon Docker, UFW, sysctl (thuộc user). Port 0.0.0.0 của service khác chỉ báo, không đụng. Dashboard và serve bind localhost, truy cập qua SSH tunnel. Dashboard hiện tại không cần auth; mở ra LAN thì phải có auth. Gradio `/demo`: CSP riêng nới tối thiểu, `share=False`, `analytics_enabled=False`, không dùng CDN.

### D9. Quy trình
- PR đụng Docker/deps/network/model loading/xoá file phải được bao-security review (skill `security-pr-review`). Mỗi lần thêm model/dataset/image thì chạy `security-model-vetting` + `security-license-check`.
- Retention (P005): **không xoá** `configs/security/**`, `docker/IMAGES.lock`, manifest sha256, SBOM, `provenance/` của bản release.

## 3. Schema file (cho bao-tester & bao-design)

`configs/security/licenses.yaml`
```yaml
schema_version: 1
assets:
  - repo_id: Qwen/Qwen3.5-0.8B        # stable key (HF repo id, or nvcr.io/... for images)
    kind: model                       # model | dataset | benchmark | image | code
    revision: 2fc06364715b967f1860aea9cf38778875588b17   # 40-hex; images: sha256:...
    license: Apache-2.0               # SPDX id or exact name+version
    license_url: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/LICENSE
    upstream: [Qwen/Qwen3.5-0.8B-Base]
    commercial: yes                   # yes | no | conditional
    derivatives: yes                  # yes | no | conditional
    obligations: "keep LICENSE"
    eval_only: false
    scope: [spike]                    # where allowed: spike | train | eval | serve
    reviewer: bao-security
    date: 2026-10-12
```

`configs/security/remote_code_allowlist.yaml`
```yaml
schema_version: 1
entries:
  - repo_id: <org/repo>
    revision: <40-hex>
    files: {<file.py>: "sha256:<hex>"}
    reviewer: bao-security
    date: <YYYY-MM-DD>
    notes: "<what was checked>"
```

## 4. Test tự động (bao-tester đã nhận)
(a) Quét secret pattern trong `outputs/**` + file fingerprint; (b) mọi revision trong configs là 40-hex/`sha256:`; (c) `trust_remote_code: true` chỉ đi kèm repo+sha có trong allowlist; (d) không asset `eval_only: true` nào xuất hiện trong train mix; (e, đề xuất) mọi `repo_id` dùng trong configs phải có trong `licenses.yaml` với `scope` phù hợp.

## 5. Việc cần người khác / user
| # | Việc | Ai |
|---|------|----|
| U1 | Vá NVIDIA Container Toolkit lên ≥1.19.1 (sudo + restart docker ảnh hưởng mọi service) | user (qua PM) |
| U2 | Chọn nhánh license A/B | user (qua PM) |
| U3 | Tạo HF token fine-grained read-only cho dự án (không dùng token cá nhân full quyền đang có trong `~/.cache/huggingface/token`, chưa rõ scope) | user |
| T1 | Implement D2 trong compose/skill backend-gpu-job | bao-backend (đã áp) |
| T2 | Pre-commit + ruff S + banned-api (P004) | bao-cleancode (đã gộp) |
| T3 | `safe_load` wrapper + guard `trust_remote_code` | bao-backend / bao-fixcode |
| T4 | Test (a)–(e) | bao-tester |

## 6. Phương án đã cân nhắc và loại
- **Dựa vào scanner pickle (picklescan/modelscan) thay vì safetensors-only**: loại, vì các scanner này đã có nhiều bypass CVSS 9.3 trong 2025.
- **Rootless Docker / userns-remap**: tốt nhưng phải sửa daemon dùng chung, ảnh hưởng service khác. Để ngoài phạm vi, đề xuất user cân nhắc sau.
- **Verify chữ ký cosign cho image NGC**: NGC có nhãn "Signed" nhưng tôi chưa tìm được tài liệu verify chính chủ cho người dùng cuối. Tạm thay bằng pin digest, sẽ xem lại khi NVIDIA công bố.
