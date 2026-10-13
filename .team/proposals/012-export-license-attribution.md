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
3. **PM:** should the exported folder also get a generated model card (README.md) now, or later (M2)?
4. **fixcode:** EXP-001 naming and i18n text.
