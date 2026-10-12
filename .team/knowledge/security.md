# Security knowledge base — VLM training platform (DGX Spark GB10, aarch64)

Tác giả: bao-security · Cập nhật: 2026-10-12 · Nguồn ở cuối mỗi mục. Ghi chú độ tin cậy: [chính chủ] / [thứ cấp] / [chưa xác minh].

---

## 0. Phát hiện thực tế trên host (audit 2026-10-12, read-only)

| # | Phát hiện | Mức | Hành động |
|---|-----------|-----|-----------|
| H1 | **NVIDIA Container Toolkit 1.19.0** cài trên host → dính **CVE-2026-24260** (TOCTOU, CVSS 8.5, code exec / privesc / data tampering). Fix ở **1.19.1** (NVIDIA bulletin 5850, 06/2026). Bulletin không có workaround. | **HIGH** | Cần `sudo apt upgrade nvidia-container-toolkit` → PM hỏi user (agent không có sudo, và đây là thay đổi hệ thống ảnh hưởng service khác). Tới khi vá: **chỉ chạy image GPU đã pin digest từ nguồn chính chủ**, không chạy image lạ với `--gpus`. |
| H2 | User `ntcai` thuộc group `docker` → mọi agent có quyền tương đương root qua docker socket. | MEDIUM (thiết kế) | Quy tắc: không `-v /:/...`, không `--privileged`, không mount `/var/run/docker.sock` vào container dự án. |
| H3 | Nhiều service khác (không thuộc dự án) publish `0.0.0.0` (7860, 18002, 18003, 4317, 6006, 50051, 8080…). Docker publish **bypass UFW** (DNAT trước chain INPUT). | INFO (không phải của mình) | Container `vlm-*` **bắt buộc bind `127.0.0.1:`**. Không đụng service người khác; báo PM như thông tin. |
| H4 | HF token cache ở `~/.cache/huggingface/token` (perm 600 — OK). | LOW | Không copy token vào repo/image; mount read-only khi cần hoặc truyền qua env từ `.env`. |
| H5 | Chưa có gitleaks / trivy / pre-commit trên host. `.gitignore` đã có `.env`, `*.safetensors`, `*.bin`, `*.pt`. | MEDIUM | Proposal 003: thêm pre-commit + gitleaks (cài qua uv/pre-commit, không cần sudo). |
| H6 | Docker 29.2.1, `daemon.json` chỉ khai báo runtime nvidia (không userns-remap, không `no-new-privileges` mặc định). | INFO | Đặt hardening ở mức container (compose) thay vì sửa daemon (đụng service khác). |

Nguồn H1: [NVIDIA product-security 2026/5850](https://github.com/NVIDIA/product-security/tree/main/2026/5850) [chính chủ]; [Wiz CVE-2026-24260](https://www.wiz.io/vulnerability-database/cve/cve-2026-24260). H3: [Baeldung — Docker ports ignoring UFW](https://www.baeldung.com/linux/docker-container-published-port-ignoring-ufw-rules).

---

## 1. Threat model rút gọn (dự án này)

Tài sản: (a) host dùng chung + dữ liệu service khác (banking-rag!), (b) secrets HF/NGC/W&B, (c) checkpoint & dataset, (d) tính toàn vẹn kết quả benchmark.

Bề mặt tấn công chính, theo xác suất thực tế:
1. **Model weights / code từ Hub** — pickle (`.bin/.pt/.pth/.ckpt`), `trust_remote_code=True`, `auto_map` → **RCE khi load**. Rủi ro số 1 với VLM vì nhiều VLM (InternVL, một số Nemotron VL, Phi-vision…) cần custom code.
2. **Python deps** — typosquat, package bị chiếm, sdist chạy code lúc build; ecosystem ML kéo hàng trăm deps.
3. **Container image** — tag trôi (`:latest`, `:25.xx` bị re-push), image cộng đồng không rõ nguồn; GPU container chạy qua NVIDIA Container Toolkit (đã có lịch sử container-escape).
4. **Dataset** — license không cho thương mại, PII, ảnh độc hại (decoder bug trong PIL/OpenCV/FFmpeg), poisoning.
5. **Serving** — vLLM/TRT-LLM endpoint expose ra LAN; SSRF qua media URL; RCE qua multimodal decode.
6. **Secrets** — commit nhầm `.env`, token trong log W&B/TensorBoard, token bake vào layer Docker.

---

## 2. Model weights & `trust_remote_code`

**Sự thật cần nhớ**
- Pickle = thực thi code tùy ý khi load. HF Hub có quét pickle nhưng chỉ **đánh dấu "unsafe"**, không chặn tải. [HF docs security-pickle](https://huggingface.co/docs/hub/security-pickle) [chính chủ]; [NSFocus](https://nsfocusglobal.com/?p=28440)
- **Scanner không đáng tin làm lớp phòng thủ duy nhất**: picklescan có 3 bypass CVSS 9.3 (CVE-2025-10155/10156/10157: đổi đuôi file, lỗi CRC zip, subclass né blocklist) + CVE-2025-71325, 71366; chiến dịch nullifAI (2025) dùng pickle nén 7z lọt scanner. [THN 12/2025](https://thehackernews.com/2025/12/picklescan-bugs-allow-malicious-pytorch.html), [Checkmarx](https://checkmarx.com/blog/free-hugs-what-to-be-wary-of-in-hugging-face-part-4/), [JFrog](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models/), [Cisco](https://blogs.cisco.com/ai/hardening-pickle-file-scanners)
- `torch.load(weights_only=True)` **không an toàn trên PyTorch < 2.6.0** (CVE-2025-32434, CVSS 9.3). [OSV](https://osv.dev/vulnerability/CVE-2025-32434) — Container NGC 2025+ đã ≥ 2.6, nhưng vẫn phải kiểm.
- vLLM có nhiều CVE liên quan load model/multimodal: CVE-2026-22807 (RCE qua `auto_map`), CVE-2026-27893 (một số model multimodal **hardcode `trust_remote_code=True`** bất chấp opt-out), CVE-2025-66448 (config loading), CVE-2026-22778 (RCE qua video URL, fix 0.14.1), CVE-2026-24779 (SSRF MediaConnector). [Orca](https://orca.security/resources/blog/cve-2026-22778-vllm-rce-vulnerability/), [aggregator](https://cve.akaoma.com/vendor/vllm) [thứ cấp — xác minh lại trên GitHub advisories của vllm-project trước khi chốt version].

**Quy tắc đề xuất (vào Proposal 003)**
1. **Chỉ safetensors**: `use_safetensors=True`; tải bằng `allow_patterns=["*.safetensors","*.json","*.txt","*.model","*.py"(chỉ khi đã duyệt)]`. Repo không có safetensors → không dùng, hoặc convert trong sandbox (container không mạng, không mount gì ngoài thư mục đó).
2. **Pin revision = commit SHA 40 ký tự** (không dùng `main`). Ghi SHA vào config YAML + manifest. Tải 1 lần bằng `snapshot_download(revision=SHA)` để mọi file cùng revision. [HF download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
3. **Nguồn chính chủ**: org `nvidia/`, `Qwen/`, `OpenGVLab/`, `google/`, `meta-llama/`, `microsoft/`, `HuggingFaceM4/`… Re-upload (user cá nhân, "-GGUF", "-fixed") → cấm trừ khi PM duyệt.
4. **`trust_remote_code`**: mặc định `False` toàn cục. Cho phép `True` **chỉ khi**: (a) repo chính chủ, (b) pin SHA, (c) security đã đọc diff các file `*.py` ở SHA đó và ghi vào allowlist `configs/security/remote_code_allowlist.yaml` (repo, sha, file hashes, reviewer, ngày), (d) nếu có bản native trong `transformers` thì dùng bản native thay vì remote code. Đổi SHA = review lại.
5. Load model **lần đầu** trong container có `--network none` sau khi đã tải (chặn exfiltration nếu có payload).
6. Không bao giờ `torch.load` file từ nguồn ngoài; checkpoint nội bộ của mình: lưu safetensors + optimizer state chỉ load lại từ chính `outputs/` của mình.

---

## 3. Container image (NGC, arm64)

- NGC catalog có nhãn **"Signed"** cho image; NVIDIA chưa công bố rõ quy trình verify bằng cosign cho người dùng cuối (tôi chưa tìm được tài liệu chính chủ — [NGC catalog](https://catalog.ngc.nvidia.com/)). → Lớp phòng thủ thực tế: **pin theo digest** `nvcr.io/nvidia/pytorch@sha256:...`, ghi digest vào `docker/IMAGES.lock` (tag gốc + digest arm64 + ngày + người duyệt). Lấy digest: `docker buildx imagetools inspect nvcr.io/nvidia/pytorch:<tag>` (chọn manifest `linux/arm64`).
- Không dùng `:latest`, không dùng image cộng đồng cho bước GPU nếu có bản NGC.
- **NVIDIA Container Toolkit** có lịch sử escape: CVE-2025-23266 "NVIDIAScape" (CVSS 9.0, ≤1.17.7, khai thác bằng `LD_PRELOAD` trong Dockerfile qua hook createContainer), CVE-2025-23267, bypass symlink 02/2025, và **CVE-2026-24260 (≤1.19.0 — host đang dính)**. Hệ quả: image độc hại + `--gpus` = nguy cơ chiếm host. [THN 07/2025](https://thehackernews.com/2025/07/critical-nvidia-container-toolkit-flaw.html), [THN 02/2025](https://thehackernews.com/2025/02/researchers-find-new-exploit-bypassing.html)
- Scan image: `trivy image` / `grype` (chạy bằng container `aquasec/trivy` pin digest, không cần cài host) — báo cáo, không chặn cứng ở Phase 2 (image NGC luôn có CVE OS mức thấp; chặn theo CRITICAL có fix).

**Hardening container `vlm-*` (compose)**
```yaml
services:
  vlm-train:
    image: nvcr.io/nvidia/pytorch@sha256:<digest>   # from docker/IMAGES.lock
    user: "1000:1000"                 # non-root; NGC images run fine as uid 1000 with HOME set
    security_opt: ["no-new-privileges:true"]
    cap_drop: ["ALL"]
    read_only: false                  # training needs writable /tmp; keep rootfs writable but narrow mounts
    ipc: private                      # avoid ipc=host on shared box; set shm_size instead
    shm_size: "16g"
    ulimits: { memlock: -1, stack: 67108864 }
    ports: ["127.0.0.1:8000:8000"]    # NEVER 0.0.0.0
    volumes:
      - ./data:/workspace/data:ro
      - ./outputs:/workspace/outputs
      - ${HF_HOME:-~/.cache/huggingface}/hub:/hf/hub   # cache only, NOT the token file
    env_file: .env
    deploy: { resources: { reservations: { devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }] } } }
```
Cấm: `--privileged`, `--network host` (trừ khi lý do rõ + PM duyệt), `--pid host`, mount `/`, `$HOME`, `/var/run/docker.sock`, `LD_PRELOAD` trong Dockerfile/env.
Lưu ý GB10 unified memory: không có ranh giới VRAM — container ăn RAM GPU = ăn RAM host (đang ~70GB dùng sẵn). Đặt `mem_limit` cho container dự án để tránh OOM-kill service khác (an toàn = availability). Phối hợp với backend để chọn số (gợi ý khởi điểm ≤ 48GB, đo thực tế).

---

## 4. Python dependencies

- `uv lock` ghi hash mặc định trong `uv.lock`; CI/Docker dùng `uv sync --locked` (hoặc `--frozen`) để không resolve lại. `uv lock` **không** có `--require-hashes` (đó là flag pip). [pydevtools](https://pydevtools.com/handbook/how-to/how-to-pin-dependencies-with-hashes-in-uv/), [bernat.tech](https://bernat.tech/posts/securing-python-supply-chain/)
- Hash không chặn được bản độc hại phát hành hợp lệ → thêm **cooldown**: `exclude-newer` trong `[tool.uv]` (vd 7 ngày trước ngày lock).
- Ưu tiên wheel (`--no-build` / `no-build-package` cho package lớn); sdist chạy code lúc build.
- Audit: `uvx pip-audit` (OSV + PyPA advisory) chạy trong CI local/pre-push; `uv` có tính năng audit/malware-check mới (preview, cần flag) [chưa xác minh trên bản uv hiện tại — kiểm `uv --help`].
- Package NVIDIA (`nvidia-modelopt`, `tensorrt_llm`, `nemo_*`) phần lớn **đã có sẵn trong container NGC** → không pip install đè lên bản trong container trừ khi cần; nếu cài thêm, chỉ từ PyPI hoặc `pypi.nvidia.com`, ghi rõ index. Cẩn thận **dependency confusion** khi dùng `--extra-index-url` (pip trộn index) → với uv dùng `index-strategy = "first-index"` (mặc định) + `[[tool.uv.index]]` explicit cho package NVIDIA.
- Không `curl | bash`. Không `pip install git+https://...@main` (pin commit nếu buộc phải dùng).

---

## 5. Secrets

- `HF_TOKEN`, `NGC_API_KEY`, `WANDB_API_KEY` chỉ trong `.env` (đã gitignore) + `.env.example` có key rỗng được commit.
- Token HF: dùng **fine-grained, read-only**, chỉ scope repo cần (gated models). NGC key: personal key, scope tối thiểu. W&B: cân nhắc `WANDB_MODE=offline` mặc định (máy dùng chung, tránh đẩy metadata ra ngoài ngoài ý muốn).
- Không `ENV HF_TOKEN=...` / `ARG` token trong Dockerfile (lộ trong layer/history). Dùng BuildKit `--secret id=hf,env=HF_TOKEN` nếu build cần tải.
- Không log env (`print(os.environ)`), W&B `config` không chứa token; tắt `wandb` code/env capture (`WANDB_DISABLE_CODE` không cần, nhưng kiểm `settings`).
- Pre-commit: **gitleaks** (hook id `gitleaks`, pin `rev` tag v8.x mới nhất — kiểm releases; README mẫu v8.24.2) + `detect-private-key`, `check-added-large-files` (chặn commit weights). Config v8.21+: dùng `[[rules.allowlists]]`. [gitleaks README](https://github.com/gitleaks/gitleaks)
- Repo **chưa có commit nào** → cơ hội tốt: chạy `gitleaks dir .` trước commit đầu tiên.
- Lộ secret → **revoke ngay** (không chỉ xoá commit), báo PM.

---

## 6. License (model & dataset)

| Đối tượng | License | Thương mại / fine-tune | Ghi chú |
|---|---|---|---|
| Qwen3-VL (2B–235B) | Apache-2.0 (theo model card — cần xác minh từng size) | Có | Qwen-VL đời đầu dùng Tongyi Qianwen License (custom) — đừng nhầm. [thứ cấp](https://playground.roboflow.com/models/qwen/qwen3-vl-8b-instruct) |
| NVIDIA Nemotron (mới, ≥12/2025) | NVIDIA Nemotron Open Model License | Có; được phát hành derivative; NVIDIA không claim output | Model NVIDIA cũ có thể dùng license khác (AI Foundation Models Community License, Internal Scientific R&D — **không thương mại**). Kiểm từng card. [NVIDIA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license) [chính chủ] |
| Gemma 3 | Gemma Terms of Use (custom, có Prohibited Use Policy) | Có điều kiện; derivative phải mang theo terms | Đọc ToU trực tiếp trước khi chốt |
| InternVL3/3.5 | Code MIT; weights **thừa hưởng license LLM backbone** (Qwen/InternLM…) | Tùy size | Kiểm từng checkpoint; thường cần `trust_remote_code` |
| Llama 3.x Vision | Llama Community License | Có (giới hạn 700M MAU, AUP, attribution "Built with Llama") | Hạn chế địa lý EU với model đa phương thức 3.2 — kiểm |
| LLaVA-Instruct-150K / LLaVA data | **CC BY-NC 4.0** + điều khoản GPT-4 | **Không thương mại** | Dữ liệu sinh bởi GPT-4. [LLaVA repo / Fireworks](https://www.fireworks.ai/blog/firellava-the-first-commercially-permissive-oss-llava-model) |
| ShareGPT4V, ALLaVA, các bộ "GPT-4V caption" | Thường NC hoặc dính ToS OpenAI | Rủi ro | Phải kiểm từng card |
| COCO images | Ảnh Flickr, license từng ảnh (nhiều CC) — annotation CC BY 4.0 | Cẩn trọng | |
| Benchmark eval (MMMU, DocVQA, ChartQA…) | Đa số chỉ cho research/eval | Chỉ dùng để **eval**, **không train** | Train trên test set = contamination + vi phạm |

Nguyên tắc: mục tiêu dùng (research nội bộ vs thương mại) phải được **user chốt** — quyết định này đổi toàn bộ danh sách dataset hợp lệ. Mỗi model/dataset dùng phải có dòng trong `configs/security/licenses.yaml` (id, revision SHA, license SPDX/tên, link, commercial: yes/no/conditional, reviewer).

---

## 7. Serving / mạng

- vLLM/TRT-LLM/NIM: bind `127.0.0.1`, bật API key (`--api-key` với vLLM) kể cả localhost; không bật endpoint fetch URL ảnh/video từ xa nếu không cần (SSRF — CVE-2026-24779); dùng vLLM ≥ bản vá mới nhất (≥0.14.1 tối thiểu; kiểm advisory mới hơn).
- Dashboard (frontend): bind localhost; truy cập từ xa qua SSH tunnel, không mở port.
- Không dùng `PyNcclPipe`/kênh distributed lắng nghe 0.0.0.0 (CVE-2025-47277) — single-node nên không cần.

---

## 8. Dataset hygiene

- Decode ảnh/video là bề mặt RCE (PIL/OpenCV/FFmpeg). Giữ libs cập nhật (theo container), đặt `Image.MAX_IMAGE_PIXELS` (chặn decompression bomb), timeout khi decode, xử lý trong worker có giới hạn bộ nhớ.
- Không train trên dữ liệu chứa PII/secrets; NeMo Curator có filter PII cho text — áp dụng cho caption/OCR text.
- Ghi checksum (sha256) cho shard dataset sau curate → manifest; verify trước train (chống tampering, cũng giúp reproducible).

---

## 9. Checklist nhanh (dùng khi review)

- [ ] Image pin `@sha256:` arm64, có trong `docker/IMAGES.lock`
- [ ] Không `--privileged` / `network host` / `pid host` / docker.sock / mount `$HOME` hoặc `/`
- [ ] Port bind `127.0.0.1`
- [ ] `user:` non-root, `no-new-privileges`, `cap_drop: ALL`, `mem_limit`
- [ ] Container tên `vlm-*`
- [ ] Model: org chính chủ, revision SHA, safetensors-only, `trust_remote_code` tắt hoặc có trong allowlist
- [ ] Deps: `uv.lock` commit, `uv sync --locked`, không index lạ, pip-audit sạch CRITICAL
- [ ] Không secret trong code/Dockerfile/log; gitleaks pass
- [ ] License model + dataset có trong `licenses.yaml`, khớp mục đích sử dụng
- [ ] Benchmark data không lẫn vào train set
- [ ] Không ghi/xoá ngoài repo; không đụng container/image không phải `vlm-*`

## 10. Công cụ (không cần sudo)
| Mục đích | Công cụ | Cách chạy |
|---|---|---|
| Secret scan | gitleaks | pre-commit hook hoặc `docker run --rm -v $PWD:/repo:ro zricethezav/gitleaks@sha256:... dir /repo` |
| Secret scan (baseline) | detect-secrets | `uvx detect-secrets scan` |
| Python CVE | pip-audit | `uvx pip-audit` (trong venv) |
| Image CVE/SBOM | trivy / grype / syft | chạy bằng container pin digest |
| Pickle scan (lớp phụ) | modelscan / fickling / picklescan | `uvx modelscan -p <path>` — **không** thay cho chính sách safetensors-only |
| Docker lint | hadolint | `docker run --rm -i hadolint/hadolint < Dockerfile` |

---

## 11. Bổ sung sau RESET (app no-code node-graph) — 2026-10-12

**ComfyUI ecosystem (bài học trực tiếp)**
- Custom node độc hại: ComfyUI_LLMVISION (06/2024, wheel giả openai/anthropic, đánh cắp credential); ultralytics bị chiếm (12/2024) lan qua node. Comfy Registry sau đó cấm `eval/exec`, cấm pip install lúc runtime, cấm obfuscation. [Comfy blog 01/2025](https://blog.comfy.org/p/comfyui-2025-jan-security-update), [Registry standards](https://docs.comfy.org/registry/standards)
- ComfyUI-Manager: CVE-2025-67303 (fix ≥3.38) bị botnet đào coin khai thác năm 2026 để tự cài node; CVE-2026-22777 (bypass security level → RCE, fix 3.39.2 / 4.0.5). [Censys](https://censys.com/blog/comfyui-servers-cryptomining-proxy-botnet/), [CSA](https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-workload-exposure-cryptomining-20260408/)
- License: ComfyUI + ComfyUI_frontend **GPL-3.0**; litegraph.js, xyflow, Langflow MIT; LLaMA-Factory Apache-2.0; Flowise Apache-2.0 trừ `packages/server/src/enterprise`. (Xác minh từ file LICENSE ngày 2026-10-12.)

**App web local**
- DNS rebinding: Ollama CVE-2024-28224 (fix: kiểm Host header); NemoClaw 2026 (bind 0.0.0.0 bỏ qua kiểm Host). [NCC](https://www.nccgroup.com/research/technical-advisory-ollama-dns-rebinding-attack-cve-2024-28224/), [Cyera](https://www.cyera.com/research/nemoclaw-one-website-visit-to-hijack-your-ai-agent)
- Phòng thủ: bind 127.0.0.1, allowlist Host, kiểm Origin (cả WS), header CSRF, không bật CORS, auth bằng token khởi động → cookie HttpOnly SameSite=Strict, redact token trong access log.

**vLLM mới**
- GHSA-h3rc-6mm3-gc2m (06/10/2026, High): RCE qua `code_revision` trong `mm_processor_kwargs` khi trust_remote_code=True + processor dùng auto_map. `/tokenize` không auth kể cả khi set `--api-key`. Ảnh hưởng 0.7.3 ≤ v < 0.31.0 (máy đang có 0.27.1). [advisory](https://github.com/vllm-project/vllm/security/advisories/GHSA-h3rc-6mm3-gc2m)
- Nhiều advisory DoS trên đường xử lý multimodal/video trong 09-10/2026 (max_frames, processor cache, cache_salt). Serve cho end-user phải pin bản mới nhất và giới hạn input.

**License model (xác minh 2026-10-12)**
- Qwen3.5 / 3.6 / 3.8: Apache-2.0, kiến trúc native, không remote code.
- Nemotron-3-Nano-Omni: remote code **bắc cầu** — `vision_config.auto_map` trỏ `nvidia/C-RADIOv4-H` (không pin revision). Bài học chung: khi review remote code phải kiểm `auto_map` trong **mọi sub-config**, không chỉ config gốc. Hồ sơ: `.team/security-reviews/2026-10-12-nemotron-3-nano-omni.md`.
- Gemma 4: **Apache-2.0, commercial=yes**. Gemma Terms of Use (sửa 2026-04-01) chỉ áp dụng cho các model trong Appendix (Gemma 1→3n…), ghi rõ "For Gemma 4 terms, see the Gemma 4 license" (/gemma/apache_2). Prohibited Use Policy gắn với ToU nên không ràng buộc Gemma 4 về license. [ToU](https://ai.google.dev/gemma/terms)
- NVIDIA Open Model Agreement (v. 2026-03-09): được dùng thương mại và làm derivative. Khi phân phối phải kèm license + câu "Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement". License chấm dứt nếu kiện NVIDIA về IP. Nemotron-3-Nano-Omni cần remote code.
