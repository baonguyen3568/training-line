# 012 — Export ships the licence + attribution of everything it is built on

- **Author:** bao-backend · **Requested by:** PM (2026-10-13, after bao-research found Export attaches nothing)
- **Reviewers:** bao-security (licence content, fail-closed rule), bao-research (licence obligations per model/dataset),
  bao-fixcode (lineage), bao-frontend (what the Export node shows)
- **Status:** DRAFT. No code until the open questions (§6) are answered.

## 1. Problem
`export.hf_dir` promises "the licence notice attached (ADR-003)" (`nodes/model_ops.py` ExportParams docstring), but
there is no export worker yet, and nothing could attach a notice today:
- **NVIDIA snapshots carry no LICENSE file.** Nemotron-Parse v1.1@fd1e3c6b has only README front matter
  (`license: other`, `license_name: nvidia-open-model-license`, `license_link: https://www.nvidia.com/...`).
  Qwen snapshots do ship `LICENSE` (Apache-2.0).
- **Catalog attribution texts are short labels.** For example Parse: `"NVIDIA Open Model License; tokenizer CC-BY-4.0"`;
  Apache models: `""`. CORD: `"... CC BY 4.0; note if modified"`. These are not the notice text a licence requires.
- **Lineage is implicit.** A trained Model artifact points into a run dir whose `run.json` has `base_model(+revision)`,
  `dataset(+revision)`, `parent_run_id`, but nothing walks that chain at export time.

Sharing an export without these is the user's legal exposure (OML attribution/notice, CC-BY attribution + "indicate
if changes were made"), and the user is non-technical: the app must do it, not document it.

## 2. Proposal
### 2.1 Vendored licence texts (no network at export, no paraphrase)
`configs/licenses/<licence-id>.txt` = the licence text **verbatim**, plus `configs/licenses/LICENSES.lock`:
`<licence-id> | source URL | sha256 | fetched_at | fetched_by | security verdict`.
- Who fetches: bao-research (or security) from the official URL. **Nobody writes or edits licence text by hand**, and
  backend does not author it.
- Ids = the catalog `license` values (`nvidia-open-model-license`, `apache-2.0`, `openmdw-1.1`, `cc-by-4.0`, …).
  A unit test asserts every catalog licence id (models + datasets) has a locked text with a matching sha256.
- If the snapshot ships its own `LICENSE`, it is copied **as well** (it is the licensor's own copy). The vendored text
  is still required, so that every export has the same layout.

### 2.2 Notice texts per licence (catalog, research-owned)
Add a `notice` field to models/datasets catalog entries. It holds the exact sentence(s) the licence asks redistributors
to include. Research supplies it from the licence and security approves it; backend never composes legal wording.
`attribution` stays the short UI label.

### 2.3 What Export writes next to the weights
```
<dest>/
  model.safetensors …            (weights; Parse: after ckpt_complete, BUG-002)
  LICENSE                        model licence, verbatim vendored text
  LICENSES/<id>.txt              every other licence involved (tokenizer, datasets), verbatim
  LICENSE.upstream               the snapshot's own LICENSE file, when there is one
  NOTICE                         human-readable: per component -> name, repo@revision, licence id + URL, notice
                                 text, and the modification statement (fine-tuned by <user tool> on <date>)
  README.md                      minimal model card (PM): base model, datasets, licences, notices, 3-line usage,
                                 known limitations from the catalog
  ATTRIBUTION.json               machine-readable, same content + sha256 of every licence file and weight file
```
The modification statement is generated from the run manifests (task lora/sft, dates, datasets with repo@revision),
because CC-BY needs "indicate if changes were made". `ATTRIBUTION.json` follows a pydantic model exported as a JSON
schema, so the frontend can render it, and gets `schema_version: 1`.

### 2.4 Lineage
`export.lineage(model_artifact) -> list[Component]`:
- Walk the artifact's run dir `run.json` → `parent_run_id` …, down to the catalog model `repo@revision`.
- Collect each run's `dataset@revision`, and resolve it in the datasets catalog. `local_folder` datasets are TD-2:
  they are listed as "user data (no licence recorded)" and the NOTICE asks the user to add its terms.
- Gaps (missing run.json, unknown repo) are listed in the NOTICE; they are never guessed.

### 2.5 Fail closed
Export refuses (new code `EXP-001 LICENSE_TEXT_MISSING`, owner fixcode for catalog/i18n) when any component's licence
has no locked text, or when its `notice` is empty for a licence that requires one. This runs in **preflight**,
before any copy. A model the catalog marks `commercial: "no"` gets the restriction stated at the top of NOTICE.

## 3. Where the code goes
- `src/vlm_pipeline/export/notices.py`: pure host code (stdlib + pydantic). `lineage()`, `build_attribution()`,
  `write_notices(dest, attribution)`. Unit-testable without GPU or models.
- `RealWorker._export`: copy the weights (hard copy, not symlink, so the folder is shareable), run
  `complete_from_base` for Parse, then `write_notices`. It is a host node, so no GPU lock.
- Graph preflight: the EXP-001 check uses the same lineage code on the planned graph (catalog-level only).

## 4. Tests
- Every catalog licence id has a vendored text whose sha256 matches LICENSES.lock.
- Lineage: hf_pick → sft → export gives model + CORD. lora → merge → export gives model + dataset. local_folder is
  reported as unrecorded. A broken chain is reported as a gap and the export still runs.
- The NOTICE contains every component's name, revision, licence URL and notice text. ATTRIBUTION.json validates
  against its schema, and its sha256 values match the files.
- Fail closed: removing a licence text gives EXP-001 in preflight, and nothing is written to `dest`.

## 5. Estimate
About 1 day of code after §6 is answered. The licence texts and notice fields are research/security work.

## 6. Open questions
1. **research/security:** exact notice wording required by the NVIDIA Open Model License for a derivative model, and
   whether a fine-tuned model counts as a "Derivative Model" that must carry it. Same question for OpenMDW-1.1.
   Is the guardrail clause something the NOTICE must restate, or only the licence text?
2. **research/security:** does CC-BY-4.0 data used for training require attribution in the *model* (debatable) or
   only good practice? PM asked to include it in either case. Proposal: include it, labelled "training data".
3. ~~PM: model card now or M2?~~ **Answered (PM 2026-10-13): now, minimal, generated per export.** README.md =
   base model name, datasets, licence(s), notice text, 3-line usage, known limitations from the catalog. Fuller card later.
4. **fixcode:** EXP-001 naming and i18n text.

## 7. Trả lời của research (2026-10-13). Đây là cách đọc licence của kỹ sư, KHÔNG phải tư vấn pháp lý; verdict cuối thuộc bao-security
### 7.1 Licence text đã vendor (§2.1)
- Có 4 id trong catalog: `nvidia-open-model-license`, `apache-2.0`, `cc-by-4.0`, `openmdw-1.1`. Text nằm ở `configs/licenses/<id>.txt`; nguồn, sha256 và cách trích ghi ở `LICENSES.lock`. Verdict security: PENDING.
- OML lấy từ **PDF chính thức** (bản phát hành 2025-10-24), không lấy từ trang HTML, vì HTML làm mất số mục (§2.1, §3.1…) mà licence tự tham chiếu. OpenMDW chỉ có HTML: lấy phần thân licence, bỏ nav/footer.
- Snapshot Parse 2.0 và Qwen có file `LICENSE` riêng. Parse v1.1 thì không. Cosmos: chưa đọc được vì gated.

### 7.2 Q1: fine-tune có là "Derivative Model" theo OML không?
- **Có.** OML §1.1 định nghĩa "Derivative Model" gồm "(a) modifications to the Model, (b) works based on the Model". Trọng số fine-tune thuộc cả (a) lẫn (b). §1.3 "Model" bao gồm "checkpoints, learnt weights … configuration files".
- **Nghĩa vụ khi phân phối** (OML §3.1, Redistribution): cấp kèm một bản copy Agreement, và kèm file "Notice" có đúng câu **"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"**. Câu chữ của điều này nói "If you distribute the Model". Áp dụng cho Derivative Model là cách đọc thận trọng, và ta chọn cách đó.
- **Cosmos** (OML §3.2): nếu phân phối model Cosmos hoặc model phái sinh từ Cosmos, **hoặc** dùng Cosmos để fine-tune, thì phải ghi **"Built on NVIDIA Cosmos"** trên website / UI / tài liệu liên quan. ⇒ NOTICE của export có câu này. Đồng thời **site và docs của chính ta** cũng phải có câu đó ở chỗ nói về template Cosmos (báo PM/design/security, M2-8).
- **Guardrail** (§2.1): đây là điều kiện ràng buộc người dùng, không có yêu cầu nhắc lại trong NOTICE. Bản copy Agreement đi kèm là đủ về nghĩa vụ. Đề xuất (thực hành tốt, không bắt buộc): NOTICE có 1 dòng trỏ tới §2.1 (Guardrail) và §2.3 (Trustworthy AI terms, https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).
- **OpenMDW-1.1** (Parse 2.0): phân phối thì phải giữ (1) bản copy agreement, (2) "all copyright notices and other notices of origin included in the Model Materials". Licence không có câu notice cố định. ⇒ Copy `LICENSE` của snapshot (đây là notice of origin của NVIDIA) cùng mọi dòng copyright có trong snapshot.
- **Apache-2.0** (Qwen, Gemma 4): §4 yêu cầu (a) kèm licence; (b) file đã sửa phải có thông báo nổi bật là đã sửa; (c) giữ các notice copyright/attribution; (d) nếu upstream có file NOTICE thì kèm phần attribution trong đó. ⇒ Copy `LICENSE` (và `NOTICE` nếu có) của snapshot, cộng với câu sửa đổi do app sinh.
- **Tokenizer của Parse (CC-BY-4.0)**: CC-BY §3(a) yêu cầu ghi tác giả, notice copyright, notice licence + URI, và **ghi rõ nếu có sửa**. Tokenizer không bị sửa khi fine-tune, nên ghi "not modified".

### 7.3 Q2: dữ liệu CC-BY dùng để train có buộc ghi công trong model không?
- **Không chắc.** Điều kiện của CC-BY chỉ áp khi model là "Adapted Material" hoặc khi ta "Share" chính dữ liệu. Trọng số model có phải Adapted Material hay không còn tranh cãi, và Creative Commons không đưa ra kết luận chung.
- Creative Commons **khuyến nghị** ghi công kể cả khi không bắt buộc. Chi phí ghi công gần như 0, nên **ghi công dưới nhãn "Training data"** như PM muốn. Không bao giờ viết rằng việc này là bắt buộc hay không bắt buộc. Câu sửa đổi ghi: "used to fine-tune this model; the dataset itself is not redistributed".

### 7.4 Giá trị `notice` đề xuất cho catalog (chờ security duyệt; cleancode mở rộng loader trước khi research ghi vào YAML)
| catalog | notice |
|---|---|
| Model `nvidia-open-model-license` (Parse v1.1) | `Licensed by NVIDIA Corporation under the NVIDIA Open Model License` |
| Model `nvidia-open-model-license` + Cosmos (cosmos-reason2-2b/8b) | `Licensed by NVIDIA Corporation under the NVIDIA Open Model License` + `Built on NVIDIA Cosmos` |
| Model `openmdw-1.1` (Parse 2.0) | `NVIDIA Nemotron Parse 2.0, provided by NVIDIA Corporation under the OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)`, kèm copy LICENSE của snapshot |
| Model `apache-2.0` (Qwen*, Gemma 4) | `Based on <repo>@<revision> by <vendor: Qwen Team, Alibaba Cloud / Google>, licensed under the Apache License 2.0. Modified: fine-tuned with <tool> on <date>.` (phần `<...>` do app điền) |
| Parse tokenizer (thành phần riêng, CC-BY-4.0) | `Tokenizer from nvidia/NVIDIA-Nemotron-Parse-v1.1 by NVIDIA Corporation, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Not modified.` |
| Dataset `cord-v2` | `Training data: CORD: A Consolidated Receipt Dataset for Post-OCR Parsing (Park et al., NAVER Clova AI), https://github.com/clovaai/cord, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Used to fine-tune this model; the dataset is not redistributed.` |
- **Cần security xác nhận:** chủ thể copyright của tokenizer Parse (README chỉ ghi licence, không ghi copyright holder), và tên vendor chính xác cho Qwen/Gemma (lấy từ LICENSE/NOTICE của snapshot).
- Đề xuất schema: thay vì 1 chuỗi `notice`, dùng `notices: list[str]` để Cosmos có 2 câu; tokenizer là component riêng có licence id riêng (`component_licenses` hoặc entry con).
- (research, sau phản hồi backend) Câu Guardrail đưa vào `notices` của model OML như dữ liệu: `Use of this model remains subject to the NVIDIA Open Model License, including Section 2.1 (Guardrail) and Section 2.3 (NVIDIA Trustworthy AI terms: https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).` Câu notice Apache theo vendor của từng entry: Qwen → "Qwen Team, Alibaba Cloud", Gemma 4 → "Google". Cả hai chờ security xác nhận theo LICENSE/NOTICE của snapshot.
- **Đính chính sau verdict security (2026-10-13):**
  - §7.1 viết "Snapshot Parse 2.0 và Qwen có file LICENSE riêng" là **khái quát sai**. Research chỉ kiểm Qwen3.5-2B@15852e8c (có LICENSE). Security xác nhận Qwen3-VL-2B-Instruct@89644892 và Gemma-4-E4B-it@ee0ef602 **không có** LICENSE/NOTICE. ⇒ Export luôn kèm bản vendor; chỉ copy LICENSE.upstream/NOTICE.upstream khi file **thật sự có** trong snapshot (loader kiểm tồn tại).
  - §7.4 Apache: **không** ghi chủ thể bản quyền mà snapshot không ghi. Dùng: `Based on {repo}@{revision}, published on Hugging Face by the Qwen organization, licensed under the Apache License 2.0. Modified: fine-tuned with {tool} on {date}.` Gemma dùng "…by Google…".
  - §7.4 tokenizer: `Tokenizer included in nvidia/NVIDIA-Nemotron-Parse-v1.1@{revision} (published by NVIDIA), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Not modified.` Parse 2.0 áp dụng tương tự.
  - Cosmos: "Built on NVIDIA Cosmos" chỉ hiện trên site **khi** template Cosmos được phát hành (spec M2-8 §4.4).
