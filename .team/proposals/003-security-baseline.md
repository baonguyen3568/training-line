# Proposal 003 — Security baseline (supply chain, secrets, license, host safety)

- Tác giả: bao-security · Ngày: 2026-10-12 · Trạng thái: **DRAFT v2 — đã thêm Ảnh hưởng RESET (§3b), schema v2 — chờ review**
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

### D6-final. Nhánh đã chốt theo câu trả lời của user (2026-10-12): **"F": phát triển cho bản thân + phân phối app MIỄN PHÍ cho người khác, không bán**
Nhánh F không trùng A (vì có phân phối) và không trùng B (vì không bán). Bảng A/B ở trên chỉ còn để tham khảo. Áp dụng như sau:

**Nguyên tắc nền: app phân phối CODE, không phân phối weights/dataset.** Node download tải model/dataset **trực tiếp từ nguồn chính chủ về máy end-user**. End-user trở thành bên nhận license trực tiếp từ HF/NVIDIA/Google, còn mình không phải bên phân phối lại. Đây là cách giảm rủi ro license lớn nhất, và ComfyUI/LLaMA-Factory cũng làm như vậy.

**(1) Model/dataset NC (CC BY-NC…), khi phân phối miễn phí:**
| Tình huống | Được không | Điều kiện |
|---|---|---|
| Team tự dùng để nghiên cứu/phát triển | Được | |
| App cho end-user **tự chọn tải** asset NC | Được | Badge ⛔ "Phi thương mại — chỉ nghiên cứu/cá nhân". End-user phải bấm xác nhận đã đọc license trước khi tải. Asset gated thì end-user tự accept trên HF bằng token của họ |
| **Template mặc định** dùng asset NC | **Không** | Template mặc định chỉ dùng asset commercial=yes. "Miễn phí" không có nghĩa end-user cũng phi thương mại: người tải app có thể là công ty, và template không được dẫn họ vào bẫy license |
| **Đóng gói kèm** (bundle) dataset/weights NC trong bản cài | Tránh | CC BY-NC cho phép phân phối lại phi thương mại có attribution, nhưng biến thể -SA kéo theo share-alike, -ND cấm sửa, còn data sinh từ GPT-4 dính thêm ToS của OpenAI. Chi phí kiểm từng bộ không đáng |
| Publish **weights fine-tune** (HF) mà train trên data NC | Hạn chế | Phải gắn NC + ghi upstream. Không làm mặc định. Mỗi lần publish cần security vet + PM duyệt |
| Asset "research only / internal R&D" (vd một số license NVIDIA cũ) | Team dùng nội bộ được, **không** đưa vào catalog | |

**(2) GPL-3.0:** GPL kích hoạt khi **phân phối**, dù bán hay cho miễn phí. Có 2 lựa chọn:
- (i) Fork ComfyUI: **bắt buộc** mở toàn bộ source app theo GPL-3.0, mãi mãi. Mọi dependency phải tương thích GPL. Kéo theo chuỗi CVE của ComfyUI/Manager.
- (ii) **Khuyến nghị**: xây trên MIT (xyflow/litegraph + backend tự viết), chỉ học ý tưởng từ ComfyUI. Sau đó user **tự chọn** license cho code của mình. Vì đã phân phối miễn phí thì mở source là hợp lý, nên tôi đề xuất phát hành **Apache-2.0** (có điều khoản patent, tương thích với hầu hết dependencies, không ép người dùng hạ nguồn). Cách này vẫn giữ được khả năng đổi hướng sau này, còn fork GPL thì khoá cứng.

**(3) Hệ quả phụ khi phân phối:**
- **Không rehost image NGC** (NVIDIA Deep Learning Container license hạn chế phân phối lại). App pull image từ `nvcr.io` trên máy end-user, và end-user dùng NGC key của họ nếu cần.
- Cần **D15: an toàn bản phát hành** vì app chạy trên máy người khác:
  - Release có checksum và chữ ký (Sigstore/cosign keyless qua CI, hoặc GPG). Có SBOM (syft) và `THIRD_PARTY_LICENSES`/NOTICE tổng hợp từ licenses.yaml.
  - Không có telemetry mặc định.
  - Auto-update (nếu có) phải verify chữ ký.
  - Mặc định an toàn: bind localhost, có auth token (D13).
  - Có file SECURITY.md hướng dẫn báo lỗi.

### D6b. License code bên thứ 3 cho app node-graph (GPL-3.0 của ComfyUI) — trình user cùng U2

License xác minh từ file LICENSE trên nhánh chính ngày 2026-10-12:

| Dự án | License | Ghi chú |
|---|---|---|
| comfyanonymous/ComfyUI (backend) | **GPL-3.0** | |
| Comfy-Org/ComfyUI_frontend | **GPL-3.0** | |
| litegraph.js (jagenjo, bản Comfy-Org) | MIT | thư viện canvas node gốc |
| xyflow (React Flow) | MIT | |
| Langflow | MIT | |
| LLaMA-Factory / LlamaBoard | Apache-2.0 | giữ NOTICE, có điều khoản patent |
| Flowise | Apache-2.0, **trừ** `packages/server/src/enterprise` và các file có ghi chú riêng (Commercial License) | chỉ lấy phần Apache |

Cách tích hợp ComfyUI → hệ quả theo nhánh. GPL-3.0 **không phải AGPL**: nghĩa vụ chỉ phát sinh khi **phân phối** (giao bản cài đặt cho người/tổ chức khác). Chạy nội bộ, hay cho người khác dùng qua web (SaaS), không kích hoạt nghĩa vụ.

| Cách dùng | A — nội bộ, không phân phối | B — thương mại, đóng nguồn | C — thương mại, mở nguồn |
|---|---|---|---|
| **Fork / vendor code ComfyUI hoặc ComfyUI_frontend** | OK, không phát sinh nghĩa vụ | **Không được** (cả app thành GPL-3.0 khi phân phối) | OK. **Toàn bộ app phải phát hành GPL-3.0**, kèm source, giữ copyright notice. Mọi dependency phải tương thích GPL-3 (MIT/BSD/Apache-2.0 thì được) |
| **Viết node/plugin chạy bên trong ComfyUI** (Python import cùng process) | OK | **Không được**: FSF coi plugin cùng process là tác phẩm kết hợp → GPL | OK, node phát hành GPL-3.0 |
| **Gọi ComfyUI không sửa đổi qua HTTP API, process riêng** | OK | Vùng xám nhưng thực tế phổ biến (FSF: giao tiếp qua socket/CLI ở mức "arm's length" thường được coi là chương trình riêng). Nếu **đóng gói kèm** ComfyUI trong bản cài thì phải kèm source + GPL text cho phần ComfyUI. Cần luật sư xác nhận | OK |
| **Chỉ học ý tưởng/UX, tự viết code trên xyflow/litegraph (MIT)** | OK | **OK, khuyến nghị** | OK |

**Bổ sung cho nhánh F (phân phối miễn phí), phương án "ComfyUI process riêng" (P009 §9):** về pháp lý là **khả thi** nếu giữ ranh giới cứng:
(a) ComfyUI không sửa, chạy process/container riêng. Nếu đóng gói kèm thì phải kèm source/offer GPL-3 cho phần đó.
(b) Custom node (import `comfy.*`) nằm trong **package/repo riêng, license GPL-3.0**, và chỉ là lớp mỏng gọi orchestrator qua HTTP/JSON.
(c) Orchestrator/engine của mình (Apache-2.0) **không import** code ComfyUI, và ngược lại.
Khi đó chi phí chính nằm ở security (pin ComfyUI, không có ComfyUI-Manager, tự theo dõi CVE của upstream) và UX, không còn ở license.

Khuyến nghị của security: nếu user chưa chắc sẽ chọn A hay B về lâu dài, **xây trên xyflow hoặc litegraph.js (MIT) + backend tự viết**, chỉ tham khảo thiết kế của ComfyUI. Cách này giữ mọi lựa chọn mở, đồng thời tránh kế thừa chuỗi CVE của ComfyUI/Manager (§3b). Fork ComfyUI là quyết định **khó đảo ngược**: một khi đã trộn code GPL thì không gỡ ra được để chuyển sang B. Tên/logo "ComfyUI" là thương hiệu, không dùng trong tên sản phẩm. (Tôi không phải luật sư. Nhánh B/C cần tư vấn pháp lý trước khi phát hành.)
Nguồn: [GPL-3.0 text](https://www.gnu.org/licenses/gpl-3.0.html), [GNU GPL FAQ — plugins, aggregate, separate programs](https://www.gnu.org/licenses/gpl-faq.html#GPLPlugins).

### D7. Dataset hygiene
Dataset chỉ nhận parquet/arrow/webdataset, **không chạy loading script**. Đặt `Image.MAX_IMAGE_PIXELS`, decode có timeout/giới hạn bộ nhớ. Ghi sha256 manifest cho từng shard sau curate. Có test chặn benchmark lọt vào train mix. Không đưa ảnh y tế/PII lên dashboard/W&B.

### D8. Host chung & mạng
Không sửa daemon Docker, UFW, sysctl (thuộc user). Port 0.0.0.0 của service khác chỉ báo, không đụng. Dashboard và serve bind localhost, truy cập qua SSH tunnel. Dashboard hiện tại không cần auth; mở ra LAN thì phải có auth. Gradio `/demo`: CSP riêng nới tối thiểu, `share=False`, `analytics_enabled=False`, không dùng CDN.

### D9. Quy trình
- PR đụng Docker/deps/network/model loading/xoá file phải được bao-security review (skill `security-pr-review`). Mỗi lần thêm model/dataset/image thì chạy `security-model-vetting` + `security-license-check`.
- Retention (P005): **không xoá** `configs/security/**`, `docker/IMAGES.lock`, manifest sha256, SBOM, `provenance/` của bản release.

## 3. Schema file (cho bao-tester & bao-design) — v2 (sau review của tester)

Config tham chiếu asset **chỉ** qua kiểu chung `AssetRef` (pydantic, `vlm_pipeline.common.config`):
```python
class AssetRef(BaseModel, extra="forbid"):
    repo_id: str                 # "Qwen/Qwen3.5-0.8B" | "lmms-lab/DocVQA" | "nvcr.io/nvidia/nemo-automodel"
    revision: str                # ^[0-9a-f]{40}$  or  ^sha256:[0-9a-f]{64}$
    split: str | None = None     # datasets only
```
Scope suy ra từ stage (configs/train → train, configs/eval → eval, configs/serve → serve).

`configs/security/licenses.yaml`
```yaml
schema_version: 2
assets:
  - repo_id: Qwen/Qwen3.5-0.8B        # stable key
    kind: model                       # model | dataset | benchmark | image | code
    revision: 2fc06364715b967f1860aea9cf38778875588b17
    license: Apache-2.0
    license_url: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/LICENSE
    upstream: [Qwen/Qwen3.5-0.8B-Base]
    commercial: yes                   # yes | no | conditional
    derivatives: yes
    obligations: "keep LICENSE"
    scope: [spike]                    # spike | train | eval | serve  (model/image level)
    reviewer: bao-security
    date: 2026-10-12
  - repo_id: lmms-lab/DocVQA          # example dataset/benchmark: rules are PER SPLIT
    kind: benchmark
    revision: <40-hex>
    license: <...>
    commercial: conditional
    splits:
      train:      {scope: [train]}               # allowed in SFT mix
      validation: {scope: [eval], eval_only: true}
      test:       {scope: [eval], eval_only: true}
```
Test (d) kiểm theo cặp `(repo_id, split)`. Rò rỉ ở mức nội dung (cùng ảnh xuất hiện ở dataset khác) do dedup G5 của P002 phụ trách.

`configs/security/remote_code_allowlist.yaml` — giữ như v1 (`entries[]: repo_id, revision, files{name: sha256}, reviewer, date, notes`).

## 4. Test tự động (bao-tester)
- (a) Quét secret pattern (`hf_`, `nvapi-`, `sk-`, W&B key 40-hex sau `WANDB`, URL có `user:pass@`) trên `outputs/**`, `benchmarks/**`, `.team/**`, mọi file dump resolved-config, file env fingerprint.
- (b) Mọi `AssetRef.revision` khớp regex pin.
- (c) Unit: `trust_remote_code: true` ⇒ (repo_id, revision) có trong allowlist. Integration: sha256 các `.py` trong HF cache khớp allowlist.
- (d) Không `(repo_id, split)` có `eval_only: true` nào nằm trong train mix.
- (e) Mọi AssetRef trong configs có trong `licenses.yaml` với scope khớp stage.
- (f) Harness eval (lmms-eval/VLMEvalKit) bắt buộc dùng revision đã khai báo: pre-download theo AssetRef, chạy với `HF_HUB_OFFLINE=1`. Integration test đối chiếu revision trong cache sau khi chạy.

## 3b. Ảnh hưởng của RESET (2026-10-12): app no-code node-graph

Sản phẩm đổi từ "pipeline cho team" thành "app cho người không biết code tự chọn model, dataset, node". Hệ quả: các quy tắc D1–D9 trước đây do người (security) thực hiện, nay **phải thành tính năng của app**, vì end-user không biết đọc `.py` hay kiểm license. D1–D9 vẫn giữ nguyên, áp cho đội dev và cho image/node first-party. Thêm D10–D14.

Bằng chứng từ hệ sinh thái tương tự (ComfyUI):
- ComfyUI_LLMVISION (06/2024): custom node cài wheel giả mạo openai/anthropic, đánh cắp mật khẩu trình duyệt. Sự cố ultralytics bị chiếm (12/2024) lan qua custom node. Sau đó Comfy Registry **cấm `eval/exec`, cấm pip install lúc runtime, cấm code obfuscate**. [Comfy blog 01/2025](https://blog.comfy.org/p/comfyui-2025-jan-security-update), [Registry standards](https://docs.comfy.org/registry/standards), [Gigazine](https://gigazine.net/gsc_news/en/20240611-comfyui-llmvision-malware)
- 2026: botnet đào coin quét các instance ComfyUI lộ ra mạng, tự cài node độc hại qua ComfyUI-Manager (CVE-2025-67303, fix ≥3.38). CVE-2026-22777 trên Manager (bypass security level → RCE, fix 3.39.2 / 4.0.5). [Censys](https://censys.com/blog/comfyui-servers-cryptomining-proxy-botnet/), [CSA note](https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-workload-exposure-cryptomining-20260408/), [BytePlus CVE-2026-22777](https://docs.byteplus.com/vi/docs/byteplus-security-bulletins/CVE-2026-22777)
- App AI local bị DNS rebinding: Ollama CVE-2024-28224 (đọc file qua API local), NemoClaw 2026 (bind 0.0.0.0 làm mất kiểm tra Host header). [NCC Group](https://www.nccgroup.com/research/technical-advisory-ollama-dns-rebinding-attack-cve-2024-28224/), [Cyera](https://www.cyera.com/research/nemoclaw-one-website-visit-to-hijack-your-ai-agent)

### D10. Custom node = code tuỳ ý → mô hình tin cậy 3 tầng
1. **v1: chỉ node first-party** (code trong repo, qua review + ruff S + security review). **Không** có "cài node từ git URL", **không** có node Python/script tự do, **không** có node "custom code" cho end-user.
2. **Sau v1, nếu mở plugin**: registry nội bộ dạng allowlist. Mỗi node pack được pin theo commit + sha256 wheel. Có static check khi nhận (cấm `eval/exec`, `subprocess` cài pip, network lúc import, obfuscation, `pickle`/`torch.load`, theo Comfy Registry standards). Security review từng bản. Phải có manifest khai báo quyền (network hosts, GPU, đường dẫn đọc/ghi).
3. **Cách ly lúc chạy, cho mọi node kể cả first-party**: node chạy trong **worker container** `vlm-*` (non-root, cap_drop ALL, chỉ mount thư mục workspace của run, **không** mount secret store, không docker.sock). Network mặc định `none`. Riêng node tải về (download) có egress tới allowlist host (huggingface.co, cdn-lfs*.hf.co, nvcr.io, pypi.nvidia.com). Orchestrator/UI **không bao giờ** import code của node.
- Nếu fork/nhúng ComfyUI hay LiteGraph: **không** kèm ComfyUI-Manager. Pin commit, theo dõi advisories ComfyUI (đã có nhiều CVE RCE 2025–2026). bao-research đánh giá fork-vs-build (R-1) cần tính chi phí này.

### D11. Workflow file = dữ liệu thuần, không chứa code hay secret
- Định dạng JSON có `schema_version`. Validate bằng pydantic `extra="forbid"`. Kích thước tối đa (vd 2MB). **Không** có field kiểu code/expression/script/template Jinja tự do. Tham số node được validate theo kiểu (enum, range, regex).
- `node_type` phải khớp node đã cài trong allowlist. Workflow tham chiếu node lạ thì từ chối và báo lỗi dễ hiểu, **không** tự cài.
- Model/dataset trong workflow là `AssetRef` pin revision. Import workflow có asset chưa vet hoặc cần remote code thì UI chặn và hiện cảnh báo (cờ `</>`, `⌖`, license).
- Đường dẫn chỉ là tương đối trong workspace (realpath check, chặn `..`, symlink, đường dẫn tuyệt đối). URL chỉ cho host trong allowlist.
- Secret chỉ tham chiếu theo tên (`"secret://hf"`), **không bao giờ** chứa value. Khi export: strip secret, đường dẫn tuyệt đối, username, hostname. Test (a) quét luôn file workflow export.
- Workflow nhúng trong PNG (kiểu ComfyUI) coi là untrusted, đi qua cùng validator. Template ship kèm app được hash-pin trong repo.

### D12. Token do user dán qua UI
- Field token chỉ cho ghi (write-only): API **không bao giờ** trả lại value, UI chỉ hiện `hf_…abcd` (4 ký tự cuối) + scope.
- Khi nhận token: gọi HF `whoami` để kiểm tra, hiện scope. Cảnh báo nếu token có quyền write. Khuyến nghị fine-grained read-only (UI có hướng dẫn từng bước).
- Lưu trữ: OS keyring (Secret Service) nếu có. Nếu không (DGX headless thường không có) thì lưu file trong app data dir **ngoài repo/workspace**, perm 0600, chủ là user chạy app. Nói rõ: mã hoá bằng key nằm trên cùng đĩa không tăng bảo mật thật, nên không quảng cáo là "encrypted".
- Token chỉ được truyền (env, trong lúc chạy) cho đúng node cần nó (download). Không vào log, workflow, run manifest, W&B, hay error message hiển thị cho user (redact trước khi dịch lỗi sang ngôn ngữ dễ hiểu).
- Có nút "Xoá token" và "Kiểm tra lại".

### D13. App web local: CSRF / DNS rebinding / máy dùng chung
- Bind `127.0.0.1` (không bind 0.0.0.0). **Host header allowlist** (`127.0.0.1:<port>`, `localhost:<port>`), sai thì trả 403. Biện pháp này chặn DNS rebinding.
- **Origin check** trên mọi request đổi trạng thái và mọi WebSocket handshake. Không bật CORS. API nhận JSON kèm custom header (vd `X-VLM-CSRF` mang token phiên) nên form cross-site không gửi được.
- **Cần auth kể cả trên localhost** (tôi sửa lại khuyến nghị trước đây cho P006): máy dùng chung có nhiều user/service, container nào chạy `--network host` cũng gọi được 127.0.0.1. Cách làm: token ngẫu nhiên sinh lúc khởi động (kiểu Jupyter), mở URL `?token=` một lần rồi đổi sang cookie `HttpOnly; SameSite=Strict`. Token lưu file 0600.
- Endpoint upload: whitelist loại file (ảnh/video/parquet/jsonl/safetensors), **không** deserialize pickle, giới hạn kích thước và số pixel, lưu tên file ngẫu nhiên.
- CSP và header như P006 §8.

### D14. Supply chain do end-user chọn → policy chạy trong app
- Node download thực thi D3: chỉ safetensors (`allow_patterns`), resolve và lưu revision SHA vào workflow/run manifest, mặc định `trust_remote_code=False`. Model cần remote code thì chỉ dùng được khi nằm trong `remote_code_allowlist.yaml` ship kèm app. Ngoài allowlist thì UI chặn, v1 không có nút override.
- Catalog model/dataset được hỗ trợ (P001) = `licenses.yaml` ship kèm app, hiển thị badge. Asset ngoài catalog (user dán repo id bất kỳ) thì hiện license đọc từ HF metadata với nhãn "chưa xác minh", kèm cảnh báo.
- Ước lượng disk/RAM trước khi tải (đồng bộ với memguard/preflight P007). Asset lớn hơn 20GB cần xác nhận.

## 5. Việc cần người khác / user
| # | Việc | Ai |
|---|------|----|
| U1 | Vá NVIDIA Container Toolkit lên ≥1.19.1 (sudo + restart docker ảnh hưởng mọi service) | user (qua PM) |
| U2 | ~~Chọn nhánh A/B~~ → user chốt **F** (miễn phí, phân phối, không bán). Còn chờ: user đồng ý (ii) xây trên MIT + phát hành Apache-2.0, hay chấp nhận fork GPL-3.0 | user (qua PM) |
| U3 | HF token: token đã đưa bị 401 và đã lộ trong transcript chat → **revoke**, tạo token fine-grained read-only mới, user tự dán vào `.env` (không qua chat) | user |
| U4 | Chấp thuận: v1 không có custom node bên thứ 3 / node code tự do (D10) | user (qua PM) |
| T1 | Implement D2 trong compose/skill backend-gpu-job | bao-backend (đã áp) |
| T2 | Pre-commit + ruff S + banned-api (P004) | bao-cleancode (đã gộp) |
| T3 | `safe_load` wrapper + guard `trust_remote_code` | bao-backend / bao-fixcode |
| T4 | Test (a)–(e) | bao-tester |

## 6. Phương án đã cân nhắc và loại
- **Dựa vào scanner pickle (picklescan/modelscan) thay vì safetensors-only**: loại, vì các scanner này đã có nhiều bypass CVSS 9.3 trong 2025.
- **Rootless Docker / userns-remap**: tốt nhưng phải sửa daemon dùng chung, ảnh hưởng service khác. Để ngoài phạm vi, đề xuất user cân nhắc sau.
- **Verify chữ ký cosign cho image NGC**: NGC có nhãn "Signed" nhưng tôi chưa tìm được tài liệu verify chính chủ cho người dùng cuối. Tạm thay bằng pin digest, sẽ xem lại khi NVIDIA công bố.
