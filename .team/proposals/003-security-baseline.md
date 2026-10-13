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
**Bổ sung 2026-10-12 (review patch 0004):** text do user cung cấp (caption/target) có thể chứa nguyên văn chuỗi special token (`</s>`, `<s>`, token task như `<predict_bbox>`). Tokenizer HF mặc định (`split_special_tokens=False`) sẽ hiểu chúng là token đặc biệt thật, nên có thể chèn EOS sớm hoặc giả lệnh task (data poisoning). Bước validate dataset của app phải phát hiện và escape/loại các chuỗi này, hoặc tokenize phần target với `split_special_tokens=True`.
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
    kind: model                       # model | dataset | benchmark | image | code | plugin
    revision: 2fc06364715b967f1860aea9cf38778875588b17
    license: Apache-2.0
    license_url: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/LICENSE
    upstream: [Qwen/Qwen3.5-0.8B-Base]
    commercial: "yes"                 # "yes" | "no" | "conditional" — quoted strings, not YAML bools
    derivatives: "yes"
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
- Lưu trữ: OS keyring (Secret Service) nếu có. Nếu không (DGX headless thường không có) thì lưu file trong app data dir **ngoài repo/workspace**, perm 0600, chủ là user chạy app. Nói rõ: mã hoá bằng key nằm trên cùng đĩa không tăng bảo mật thật, nên không quảng cáo là "encrypted". (Cập nhật 2026-10-13: D-7 lưu trong DB riêng, AES-GCM, khoá ở `~/.config/ntc-training/secrets.key` 0600, xem P015 §4.5–4.6. Mục đích là để DB/backup không lộ plaintext; vẫn không quảng cáo là "encrypted" với user.)
- Token chỉ được truyền (env, trong lúc chạy) cho đúng node cần nó (download). Không vào log, workflow, run manifest, W&B, hay error message hiển thị cho user (redact trước khi dịch lỗi sang ngôn ngữ dễ hiểu).
- Có nút "Xoá token" và "Kiểm tra lại".
- **Loại token (chốt cùng bao-design, P010 §16):** chế độ Simple hướng dẫn tạo token **Read** cổ điển, vì fine-grained dễ tick sai quyền và người dùng sẽ cấp rộng hơn để "sửa lỗi". Fine-grained để ở Advanced. Server đọc `whoami.auth.accessToken.role`: `write` → **cảnh báo một lần lúc lưu** (icon ⚠ + popover, không banner thường trực), audit `warning:write_scope`, vẫn chấp nhận. Đổi từ "từ chối" ngày 2026-10-13 (PM): user chọn giữ token write (U3 đóng, chấp nhận rủi ro); bù bằng 5 bảo vệ P015 §6.1; `fineGrained` → kiểm có quyền đọc repo gated. UI ghi rõ rủi ro còn lại: "Token Read đọc được mọi repo của bạn, kể cả riêng tư". Gợi ý đặt tên `vlm-app-read` + thời hạn hết hạn.

### D13. App web local: CSRF / DNS rebinding / máy dùng chung
- Bind `127.0.0.1` (không bind 0.0.0.0). **Host header allowlist** (`127.0.0.1:<port>`, `localhost:<port>`), sai thì trả 403. Biện pháp này chặn DNS rebinding.
- **Origin check** trên mọi request đổi trạng thái và mọi WebSocket handshake. Không bật CORS. API nhận JSON kèm custom header (vd `X-VLM-CSRF` mang token phiên) nên form cross-site không gửi được.
- **Cần auth kể cả trên localhost** (tôi sửa lại khuyến nghị trước đây cho P006): máy dùng chung có nhiều user/service, container nào chạy `--network host` cũng gọi được 127.0.0.1. Cách làm: token ngẫu nhiên sinh lúc khởi động (kiểu Jupyter), mở URL `?token=` một lần rồi đổi sang cookie `HttpOnly; SameSite=Strict`. Token lưu file 0600.
- **CSRF + WebSocket (chốt sau khi hiện thực S-5, frontend + backend):**
  - CSRF token = HMAC(server_secret, session_id), cấp qua `GET /api/session` (JSON body). Client giữ token trong bộ nhớ, gửi header `X-VLM-CSRF`. **Không** dùng cookie đọc được: cookie không tách theo port và SameSite coi mọi port của 127.0.0.1 là cùng site, nên app khác trên máy có thể tossing cookie.
  - WebSocket: lúc handshake kiểm Origin khớp chính xác scheme+host+port và cookie `vlm_session`. Message đầu tiên bắt buộc là `{"type":"auth","csrf_token":…}` (so bằng compare_digest). Server không gửi gì trước khi auth xong. Sai hoặc quá 5 giây → close 4401. **Không** truyền token qua query string hay subprotocol (lọt vào log).
- **Hiện thực:** `src/vlm_pipeline/app/security.py` (bao-security, M1-7). ASGI middleware + `install_security()` + `ws_authenticate()`. uvicorn bắt buộc chạy `access_log=False`.
- **E2E Playwright trên host dev:** Chromium chạy `--no-sandbox`, vì host có `kernel.apparmor_restrict_unprivileged_userns=1`, không đổi được khi không có sudo. Điều kiện bù: chặn mọi request không phải 127.0.0.1, env đã lọc (không có token thật), profile tạm, fixture tĩnh. Khi E2E render nội dung từ internet thì chuyển sang container Playwright chính chủ.
- Endpoint upload: whitelist loại file (ảnh/video/parquet/jsonl/safetensors), **không** deserialize pickle, giới hạn kích thước và số pixel, lưu tên file ngẫu nhiên.
- CSP và header như P006 §8.

### D14. Supply chain do end-user chọn → policy chạy trong app
- Node download thực thi D3: chỉ safetensors (`allow_patterns`), resolve và lưu revision SHA vào workflow/run manifest, mặc định `trust_remote_code=False`. Model cần remote code thì chỉ dùng được khi nằm trong `remote_code_allowlist.yaml` ship kèm app. Ngoài allowlist thì UI chặn, v1 không có nút override.
- Catalog model/dataset được hỗ trợ (P001) = `licenses.yaml` ship kèm app, hiển thị badge. Asset ngoài catalog (user dán repo id bất kỳ) thì hiện license đọc từ HF metadata với nhãn "chưa xác minh", kèm cảnh báo.
- Ước lượng disk/RAM trước khi tải (đồng bộ với memguard/preflight P007). Asset lớn hơn 20GB cần xác nhận.

### D16. Custom node sandbox v1 (user chốt cho phép node bên thứ 3 từ v1, theo ADR-009). Đồng tác giả: bao-security + bao-backend (DRAFT)
D16 thay thế tầng 1–2 của D10. Tầng 3 (cách ly lúc chạy) vẫn giữ, áp cho **mọi** node.

**Threat model.** Mục tiêu là giảm thiệt hại trong trường hợp xấu nhất, vì tác giả node có thể độc hại hoặc bị chiếm (tiền lệ LLMVISION, ultralytics).
| # | Kẻ tấn công muốn | Đường đi | Chặn bằng |
|---|---|---|---|
| T1 | Lấy token HF/NGC, cookie, SSH key | Đọc env, mount, `$HOME` | Container node **không bao giờ** nhận secret hay env của host. Không mount gì ngoài `/in` (ro) và `/out`. Muốn tải từ HF thì dùng node download first-party |
| T2 | Exfiltrate dataset/model của user | Mạng | **v1: luôn `--network none`** cho node bên thứ 3. Từ v1.1, nếu mở thì qua egress proxy allowlist (container `vlm-egress`, network docker `internal`) |
| T3 | Đào coin / chiếm GPU (đúng payload của botnet ComfyUI 2026) | GPU, CPU | GPU chỉ cấp khi manifest khai báo **và** user duyệt. Có giới hạn thời gian chạy, `--cpus`, `--memory`, `--pids-limit`. Lấy GPU_LOCK như job khác. Hiển thị usage |
| T4 | **Thoát container ra host** (host đang chạy banking-rag, kafka…) | Lỗi kernel, lỗi NVIDIA Container Toolkit (CVE-2025-23266, CVE-2026-24260), driver ioctl khi có `--gpus` | Toolkit ≥1.19.1 là **hard gate**. Non-root, `cap_drop ALL`, `no-new-privileges`, rootfs read-only, seccomp mặc định. **Rủi ro còn lại: kernel + driver GPU dùng chung, không có sandbox mạnh hơn** (gVisor nvproxy chưa có bằng chứng hỗ trợ arm64/driver 580) |
| T5 | Leo thang qua orchestrator | Manifest đòi mount/flag; output độc hại (pickle) được node first-party load | Manifest **không** được chọn mount hay docker flag; orchestrator tự sinh từ bảng quyền cố định. Output của node bên thứ 3 chỉ được nhận khi **parse được đúng định dạng thuộc allowlist** (json/jsonl, safetensors header hợp lệ, parquet `PAR1`, ảnh/video decode được), có giới hạn size. Không dùng denylist magic pickle, vì pickle protocol 0/1 là dạng text. **Cấm pickle/.pt/.bin** đi qua cạnh nối |
| T6 | Bám trụ, cài cắm lâu dài | Ghi vào code node, cache dùng chung | Container mới cho mỗi lần chạy. Code node mount ro. Không cache chung giữa các node pack. Không auto-update |
| T7 | Mã độc lúc **cài đặt** | `setup.py`/build hooks, pip deps | Build image riêng cho từng node pack trong sandbox build: chỉ có mạng tới PyPI/pypi.nvidia.com, không mount gì, deps từ lockfile có hash. Sau khi build thì image bất biến, digest được ghi lại |
| T9 | Chiếm phiên app trên trình duyệt (đề xuất của bao-frontend) | Node nạp JS/CSS vào SPA: chạy cùng origin nên gọi được mọi API với cookie phiên (HttpOnly không cứu được), đọc được CSRF token, phá toàn bộ D13 | Node bên thứ 3 **chỉ đóng góp dữ liệu khai báo** (NodeSpec, JSON Schema, x-ui, i18n). **Không nạp JS/CSS.** Text render dạng plain text. Widget chỉ lấy từ bộ widget có sẵn. UI riêng cho node (v1.1+) phải chạy trong iframe sandbox **khác origin**, giao tiếp qua postMessage có schema |
| T8 | Lừa UI / người dùng | Tên giả node first-party, mô tả gây hiểu lầm | Namespace bắt buộc `publisher/pack`; prefix `vlm/` dành riêng. Badge "Bên thứ 3 – chưa xác minh". Quyền hiển thị do orchestrator tự sinh, không lấy từ text tác giả viết |

**Mô hình quyền: manifest chốt (P009 §2.4, backend + design + frontend thống nhất). Danh sách đóng, orchestrator tự sinh docker args:**
```yaml
permissions:
  gpu: false                    # bool
  network: none                 # none | {hosts: ["huggingface.co", ...]}  — exact hostnames, no "*", no IP, no "any"
  max_runtime: 1h
  max_memory: 16GiB             # checked by preflight/memguard (P007)
# NO "mounts"/"extra_paths"/"env"/"secrets"/"devices"/"docker_args" keys — schema extra=forbid → manifest_invalid
origin: {publisher, package, version, source_repo, commit_sha (40-hex), source_sha256, image_digest, license}
```
Mount do orchestrator cố định: `/in` (ro, artifact được nối vào), `/out` (rw, run dir của node). **Không có quyền đọc đường dẫn tuỳ ý.**

**Mức cảnh báo, v1. Chốt theo đề xuất của backend: node bên thứ 3 CHỈ có `network: none`.** Lý do: một allowlist host vẫn là kênh exfiltrate. Ví dụ huggingface.co: plugin upload dữ liệu của user lên repo của kẻ tấn công bằng token của chính kẻ đó. Proxy còn là một service sống lâu phải vá. Field `network.hosts` giữ trong schema để mở lại ở v1.1 (cần proposal riêng). v1 khai báo khác `none` → `permission_unsupported`.
| Tổ hợp | Mức | Xác nhận |
|---|---|---|
| chỉ io | xanh | bấm Cài |
| `gpu: true`, nguồn **chưa xác minh** | ⛔ đỏ | **gõ tên node** (theo phản biện của bao-design: cài là việc làm 1 lần nên không gây "alarm fatigue"; GPU mở thêm đường thoát T4 và là payload đào coin T3) |
| `gpu: true`, nguồn **đã review** / first-party | ⚠ vàng | tick checkbox + bấm Cài |
| `max_runtime`/`max_memory` vượt mặc định CPU-only (1h / 8GB) | ⚠ vàng | bấm Cài |
| *(v1.1+)* network tới host phổ biến / host khác / GPU + network | vàng / ⛔ đỏ + gõ tên / ⛔ đỏ + gõ tên | |

**Image của plugin: build từ source, v1 không nhận image dựng sẵn.** `Dockerfile` của plugin phải `FROM` một base trong allowlist (image `vlm-*` của mình đã pin digest). Build chạy trong sandbox: không mount, chỉ có mạng tới PyPI, không secret. Sau đó trivy scan, kết quả hiển thị dạng thông tin. Image dựng sẵn của publisher là binary không review được, xem lại ở v1.1+.

**Lý do chặn cài (blocking):**
| Mã | Khi nào |
|---|---|
| `manifest_invalid` | schema sai, có key lạ (mounts/env/secrets…), `network` chứa `*`/IP/"any" |
| `hash_unpinned` | không có commit SHA 40-hex / source sha256, hoặc tải về không khớp |
| `forbidden_code` | quét tĩnh phát hiện: `exec`/`eval`/`compile` trên dữ liệu không phải hằng; pip/uv/apt install lúc runtime; payload encode (base64/zlib/marshal) lớn đem đi thực thi; tải và chạy binary; `ctypes`/`LD_PRELOAD`; `os.setuid`/ghi vào `/proc`, `/sys` |
| `license_blocked` | không có LICENSE, hoặc license cấm sử dụng |
| `base_image_not_allowed` | `FROM` ngoài allowlist |
| `build_failed` | build lỗi, hoặc build cần mạng ngoài PyPI |
| `permission_unsupported` | xin quyền chưa hỗ trợ ở v1 (`network` khác `none`) |
| `policy_block` | chính sách quản trị của host (vd host dev chặn plugin bên thứ 3 có GPU) |
| `image_digest_mismatch` (lúc chạy) | digest image hiện tại khác lúc cài, nghĩa là bị sửa |

**Không chặn, chỉ cảnh báo:** `subprocess`, network lúc import, `pickle`/`torch.load`, `trust_remote_code`, file nhị phân lạ. Kèm câu "Quét tự động không đảm bảo an toàn".

**UI khi cài (lời thường, tiếng Việt):**
1. Nguồn: URL repo + commit SHA, publisher, ngày commit cuối, có phải registry nội bộ hay không. Badge "Chưa xác minh".
2. **Quyền xin**, mỗi quyền 1 dòng bằng lời thường, vd "Dùng GPU: node có thể chạy tính toán nặng trên máy bạn" hay "Kết nối Internet tới: api.example.com: dữ liệu của bạn có thể được gửi đi".
3. License của node (đọc từ file LICENSE; GPL/NC hiện badge). Node không đi kèm app.
4. Kết quả quét tĩnh, hiển thị dạng cảnh báo, không phải "an toàn": có `eval/exec`, `subprocess`, network lúc import, code obfuscate/base64 lớn, `pickle/torch.load`, `trust_remote_code`, pip install lúc runtime. Mọi kết quả đều ghi kèm "quét không đảm bảo an toàn".
5. Danh sách pip deps mới + kích thước image.
6. Xác nhận: mức ⚠ vàng thì tick 1 checkbox cho mỗi quyền rồi bấm Cài. Mức ⛔ đỏ (v1: GPU + nguồn chưa xác minh; v1.1+: network) thì phải **gõ tên node**. Không có lựa chọn "luôn cho phép". Câu trấn an về cách ly chỉ hiện khi policy `sandbox_verified=true`.
7. Cập nhật: không auto-update. Màn update hiện **diff quyền** và diff code (số file/dòng đổi). Quyền tăng thì phải duyệt lại từ đầu. Có nút gỡ, gỡ là xoá luôn image.

**"Cài node còn thiếu" từ workflow chia sẻ (bao-fixcode phát hiện, CFG-004).** Đây là đường tấn công điển hình: workflow mồi + nút "Install missing nodes", vốn là đường ComfyUI-Manager bị lạm dụng.
- File workflow chỉ được mang `node_type` và gợi ý `publisher/package@version`. Gợi ý này **chỉ dùng làm khoá tra** registry/allowlist. `suggested_package` hiển thị là **kết quả server tra ra** (không ra thì `null`). Tên gói trong workflow không bao giờ được hiện nguyên văn hay đưa vào lệnh cài (chống typosquat/spoof). UI chỉ hiện `suggested_package` khi `trust_level` ∈ {first_party, reviewed}. Lỗ này do bao-frontend phát hiện, bao-fixcode chốt. **App bỏ qua mọi URL, commit hay nguồn cài nằm trong workflow**: nguồn cài chỉ resolve từ registry/allowlist của app, hoặc từ URL người dùng **tự gõ**.
- Không có cài hàng loạt ("cài tất cả node thiếu"). Mỗi node một màn cài, có đủ bước xác nhận D16.
- Node gợi ý không có trong nguồn tin cậy: trang thông tin ghi "Không có trong nguồn tin cậy — workflow này do người khác chia sẻ", không có nút cài 1 chạm. Người dùng muốn cài thì tự dán URL repo, và node rơi vào mức "chưa xác minh" (nếu có GPU thì phải gõ tên).
- Workflow có node chưa cài vẫn mở được (placeholder) nhưng không chạy được (CFG-004). Không bao giờ tự cài.
- **Định danh vs tên hiển thị** (bao-fixcode đề xuất, security chốt cách làm đơn giản hơn):
  - `node_type`, `publisher`, `package` là **ASCII-only** theo regex `^[a-z0-9][a-z0-9_.-]{0,63}$` (node_type = `publisher/package/node`). Định danh có ký tự ngoài regex → `manifest_invalid` / CFG-007. Cách này loại bỏ cả lớp homoglyph Cyrillic, bidi, zero-width ở tầng định danh.
  - `name_lookalike_of` trên định danh ASCII: so sau khi map confusables ASCII (`rn→m`, `vv→w`, `1/l/I`, `0/o`, `-`/`_`/`.` coi như nhau) cộng khoảng cách edit ≤ 2 với node first_party/reviewed.
  - Tên hiển thị/i18n được phép Unicode (tiếng Việt): normalize NFKC. Có ký tự bidi (U+202A–202E, U+2066–2069), điều khiển hoặc zero-width (U+200B–200F, U+FEFF) thì từ chối. Lookalike của tên hiển thị dùng skeleton Unicode TR39. UI escape ký tự lạ.
- UI không điền sẵn link cài lấy từ workflow, kể cả dạng gợi ý (bao-design). Tên node lấy từ workflow là text không tin cậy: hiển thị plain, cắt ngắn, kèm nhãn "tên do file workflow ghi". Server cảnh báo khi tên giống node first-party/reviewed (T8). Ô link chỉ nhận commit SHA 40-hex.
- `trust_level` ∈ `first_party | reviewed | unverified` **chỉ do server tính** từ registry/allowlist và hash thực tế. Mọi trường kiểu trust/verified nằm trong workflow hay manifest đều bị bỏ qua (workflow mồi có thể tự khai "verified"). Đề xuất của bao-fixcode.

**Nguồn tin cậy & pin (phối hợp P004 §8.5 của cleancode):**
- 3 mức tin cậy hiển thị trên UI: `first-party` (code nằm trong repo, phát hành kèm chữ ký release D15) / `đã review` (có trong `configs/security/plugin_allowlist.yaml`: repo, commit SHA, sha256 tarball nguồn, digest image build, reviewer, ngày) / `chưa xác minh`.
- Pin khi cài: **commit SHA 40-hex** (không nhận branch/tag) + sha256 tarball nguồn + digest image đã build. Lần chạy nào digest không khớp với lúc cài thì từ chối.
- v1 không làm hạ tầng ký cho bên thứ 3 (chưa có PKI hay registry). Hash pin + review là đủ cho v1. Ký bằng Sigstore xem lại khi có registry.
- **Chỉ chạy trong container. Cấm phương án "venv riêng / subprocess trên host"**: venv không cách ly filesystem, network, secret hay `$HOME`, nên mọi bảo vệ T1–T6 sẽ mất.
- Secret: **v1 không truyền secret vào plugin, kể cả khi user muốn cấp**. User no-tech sẽ bấm "Cho phép" theo phản xạ. Plugin cần dữ liệu từ HF thì nhận artifact từ node download first-party nối vào. Nếu sau này thật sự cần, thiết kế token phạm vi hẹp cho từng plugin (fine-grained, read, 1 repo), làm proposal riêng.
- RAM: `max_memory` được preflight/memguard (P007) kiểm như job khác, vì trên UMA GPU allocation không bị `--memory` chặn.
- License: entry `kind: plugin` trong licenses.yaml cho plugin first-party / đã review. Plugin chưa xác minh thì license đọc từ repo và hiển thị nhãn "chưa xác minh". App không phân phối kèm plugin bên thứ 3.

**Hiện thực phía executor (bao-backend, trả lời a/b/c):**
- **(a) Network.** Docker SDK (`docker` Python lib / CLI) làm được: `docker network create --internal vlm-sandbox` + container `vlm-egress` (proxy CONNECT allowlist hostname, vd tinyproxy/squid pin digest) nối cả `vlm-sandbox` lẫn bridge; node có quyền network nhận `--network vlm-sandbox -e HTTPS_PROXY=http://vlm-egress:3128`. Không cần quyền daemon đặc biệt.
  **Đề xuất của backend: v1 chỉ hỗ trợ `network = []` (none) cho node bên thứ 3**, quyền network để v1.1. Lý do: mọi nhu cầu tải model/data đã có node download first-party; proxy là thêm 1 service sống lâu phải vá/giám sát, và allowlist hostname vẫn exfiltrate được qua host được phép. Giữ field trong manifest (schema) để không phá tương thích khi bật.
- **(b) Giới hạn tài nguyên.** Không dùng một mặc định cứng 1h/16GB cho mọi node:
  - Node **CPU-only**: `--memory` mặc định 8GB, `--cpus 4`, `--pids-limit 512`, timeout 1h; manifest được xin cao hơn (cảnh báo Trung bình), trần 32GB / 12h.
  - Node **gpu=true**: `--memory` chỉ chặn RSS CPU, **không chặn CUDA trên UMA** → giới hạn thật là `max_memory` khai báo + preflight + memguard watchdog (P007, dừng exit 75 → MEM-002). `--memory` đặt = `max_memory` (vì load weights đi qua RAM CPU trước: 16GB không đủ cho model 9B bf16 ≈ 18GB). Timeout theo `max_runtime` khai báo (trần 24h), executor gửi SIGTERM → grace 120s để lưu checkpoint → SIGKILL (exit 137 do mình huỷ → `interrupted`, không phải lỗi — quy tắc fixcode).
  - Mọi node: `--read-only` + tmpfs `/tmp` (giới hạn size), `--shm-size` 2GB (GPU 8GB), `--ulimit nofile`.
- **(c) UID riêng.** Làm được không cần root: chạy node bên thứ 3 với `--user 61001:<gid của user app>` (uid cao để tránh trùng user thật trên host; 1001 có thể trùng). Orchestrator (uid app) tạo `/out` trước với mode `2770` (setgid group app) → file node ghi ra thuộc `61001:<gid app>`, orchestrator đọc được qua group và **xoá được** (xoá cần quyền ghi thư mục cha, do orchestrator sở hữu). Sau khi node xong, orchestrator validate output (whitelist định dạng, size) rồi **copy/hardlink sang vùng artifact** do uid app sở hữu; `/out` gốc bị xoá. `/in` mount ro nên uid khác không cần quyền ghi.
- Container: tên `vlm-node-<run_id>`, label `project=vlm-training`, `vlm.trust=third_party`, KHÔNG `--rm` (để đọc OOMKilled/exit), `docker rm` sau khi ghi run.json. Executor chỉ sinh docker args từ bảng quyền cố định; mọi field lạ trong manifest → từ chối cài (extra=forbid).

**Chính sách riêng cho máy dev dùng chung này (DGX Spark) — khuyến nghị gửi PM/user:**
- **Cho tới khi user xác nhận đã vá toolkit (≥1.19.1): CẤM chạy node bên thứ 3 có `gpu=true`** trên máy này. Lý do: CVE-2026-24260 cho phép từ container ra host, nghĩa là thoát ra host chung với banking-rag và các service khác.
- Sau khi vá: node bên thứ 3 trên máy này chỉ cài từ danh sách team đã review (security đọc code ở SHA đó). Không cài node tuỳ ý từ git URL. Lý do: kernel và driver GPU dùng chung, không có sandbox mạnh hơn container cho GPU trên arm64. Một lỗi kernel/driver 0-day ảnh hưởng **dữ liệu của người khác** (banking-rag), không chỉ dự án mình. Rủi ro này không cân xứng với lợi ích trên máy dev.
- Đây là chính sách **vận hành cho host này**, không giới hạn tính năng sản phẩm. Trên máy của end-user, app vẫn cho cài node bên thứ 3 theo ADR-009, kèm cảnh báo ở trên, và quyết định là của end-user cho máy của họ.

### D17. `deployment.profile`: chính sách theo loại máy (user chấp nhận chính sách host dev, ghi trong ADR-009 A1). DRAFT, bao-security + bao-backend
- **Một nguồn cấu hình duy nhất** cho các chính sách phụ thuộc loại máy, dùng chung với `storage.approval_policy` của P005:
  ```toml
  # ~/.config/vlm-app/policy.toml  — admin-owned, 0644; NOT inside the repo/workspace; NOT writable via any API/UI
  [deployment]
  profile = "shared_dev"          # "shared_dev" | "personal" (default when file absent: "personal")
  [storage]
  approval_policy = "threshold"   # derived default from profile; may be set explicitly
  threshold_gb = 20
  ```
- **Chống hạ cấp:** UI, API, workflow, manifest và env đều **không** đổi được profile. Server chỉ đọc file lúc khởi động. Nếu có override (env `VLM_PROFILE` cho test/CI) thì chỉ được phép **chặt hơn** (personal → shared_dev), không bao giờ nới ra.
- **Tự phát hiện trạng thái vá toolkit, không khai báo tay:** lúc khởi động server chạy `nvidia-ctk --version` (argv cố định), parse ra `toolkit_patched = version >= 1.19.1`. Không đọc được hoặc parse lỗi thì coi là `false` (fail-closed). Kết quả được ghi vào `/api/policy` và log khởi động.
- **Quy tắc plugin khi `profile = shared_dev`:**
  1. `trust_level = unverified` → `policy_block(reason=not_reviewed)`.
  2. `gpu = true` và `toolkit_patched = false` → `policy_block(reason=toolkit_unpatched)` (áp cho mọi plugin bên thứ 3, kể cả reviewed).
  3. first_party không bị policy chặn (vẫn qua GPU_LOCK + memguard).
  Ghi chú: quy tắc (1) **chặt hơn** chính sách user đã duyệt ở giai đoạn *trước khi vá* (lúc đó chỉ chặn plugin có GPU), vì nó chặn cả plugin không dùng GPU nhưng chưa được review. Lý do: plugin không GPU vẫn chạy code tuỳ ý trên kernel dùng chung. Đã nêu rõ cho PM xác nhận (U5).
- `personal`: không có policy_block mặc định. Mọi kiểm soát D16 vẫn áp dụng như bình thường.
- `/api/policy` trả `{profile, toolkit_patched, toolkit_version, sandbox_verified, storage:{approval_policy, threshold_gb}}`. UI hiện banner "Máy dùng chung — chính sách quản trị đang bật" khi `shared_dev`.
- Trên host này (bao-backend chỉ ra: agent không được ghi ra ngoài repo theo CLAUDE.md):
  (1) mọi launcher/script dev của dự án `export VLM_PROFILE=shared_dev`, vì env chỉ được làm chặt hơn nên không mở được lỗ;
  (2) PM nhờ user tạo `~/.config/vlm-app/policy.toml` (`profile = "shared_dev"`) một lần (U6);
  (3) khi profile hiệu lực là `personal` mà host đang chạy container không thuộc dự án (không phải `vlm-*`), server log WARNING kèm banner "Máy có vẻ dùng chung — cân nhắc profile shared_dev". Chỉ cảnh báo, không tự đổi profile.
  Loader `app/policy.py`: pydantic frozen extra=forbid, đọc 1 lần lúc khởi động. File có nhưng hỏng thì fail-closed về `shared_dev` và ghi log.
  File không nằm trong repo nên không lan sang máy end-user.

## 5. Việc cần người khác / user
| # | Việc | Ai |
|---|------|----|
| U1 | Vá NVIDIA Container Toolkit lên ≥1.19.1 (sudo + restart docker ảnh hưởng mọi service) | user (qua PM) |
| U2 | ~~Chọn nhánh A/B~~ → user chốt **F** (miễn phí, phân phối, không bán). Còn chờ: user đồng ý (ii) xây trên MIT + phát hành Apache-2.0, hay chấp nhận fork GPL-3.0 | user (qua PM) |
| U3 | HF token: token đã đưa bị 401 và đã lộ trong transcript chat → **revoke**, tạo token fine-grained read-only mới, user tự dán vào `.env` (không qua chat) | user — **ĐÓNG 2026-10-13: user giữ token hiện tại, chấp nhận rủi ro**; bù bằng 5 bảo vệ có test hồi quy (P015 §6.1) |
| U6 | Tạo `~/.config/vlm-app/policy.toml` với `profile = "shared_dev"` (agent không được ghi ngoài repo) | user (qua PM) |
| U5 | D17: trên host dev, xác nhận chặn cả plugin **không GPU** nhưng chưa review (chặt hơn bản đã duyệt ở giai đoạn trước vá) | user (qua PM) |
| U4 | ~~v1 không custom node bên thứ 3~~ → user CHO PHÉP (ADR-009) ⇒ D16. Chờ user duyệt **chính sách host dev**: cấm node bên thứ 3 có GPU tới khi vá toolkit; sau vá chỉ node đã được team review | user (qua PM) |
| T1 | Implement D2 trong compose/skill backend-gpu-job | bao-backend (đã áp) |
| T2 | Pre-commit + ruff S + banned-api (P004) | bao-cleancode (đã gộp) |
| T3 | `safe_load` wrapper + guard `trust_remote_code` | bao-backend / bao-fixcode |
| T4 | Test (a)–(e) | bao-tester |

## 6. Phương án đã cân nhắc và loại
- **Dựa vào scanner pickle (picklescan/modelscan) thay vì safetensors-only**: loại, vì các scanner này đã có nhiều bypass CVSS 9.3 trong 2025.
- **Rootless Docker / userns-remap**: tốt nhưng phải sửa daemon dùng chung, ảnh hưởng service khác. Để ngoài phạm vi, đề xuất user cân nhắc sau.
- **Verify chữ ký cosign cho image NGC**: NGC có nhãn "Signed" nhưng tôi chưa tìm được tài liệu verify chính chủ cho người dùng cuối. Tạm thay bằng pin digest, sẽ xem lại khi NVIDIA công bố.
