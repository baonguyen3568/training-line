# 021 — Bring your own model / dataset from Hugging Face (ADR-021, stream H), backend side

- Author: bao-backend · 2026-10-14 · Status: APPROVED by PM (v2: A1 + §7.1 decided); security review pending
- Decision: docs/adr/ADR-021-bring-your-own-model-data.md (user, 1.0). Error codes: fixcode (list agreed 2026-10-14).
- Error codes (fixcode, agreed 2026-10-14): NET-005 HUB_REPO_UNAVAILABLE {repo, kind, reason: not_found |
  not_found_or_private | gated_no_token | gated_not_accepted | network}; MODEL-003 MODEL_ARCH_UNSUPPORTED {repo,
  revision, architecture, model_type}; MODEL-004 MODEL_NEEDS_REMOTE_CODE {repo, revision}; SEC-005
  LICENSE_NOT_CONFIRMED {name, repo, license}; DATA-009 DATASET_HAS_SCRIPT {repo, has_parquet_export}; DATA-010
  DATASET_COLUMNS_MISSING {repo, missing, available}; DATA-011 DATASET_COLUMN_TYPE {repo, column, role, expected,
  found}. DATA-008 stays reserved for P020 (DATASET_NOT_FOR_NODE).

## A1 (user, 2026-10-14) — folded in below

- License unknown (no card id): the download is allowed, **export is blocked** for that asset and everything trained
  from it (lineage check in the export worker: SEC-005 with `license: unknown`); a known license still needs the
  confirmation before download.
- **40 GB cap** per user-added model (`user_assets.max_model_gb` in the settings DB, default 40, the user may raise
  it in Settings); over it → the existing large-download refusal with the cap named.
- **WebDataset `.tar` / `.zip` in 1.0**: the hub job downloads the shards; extraction runs in a second job with
  `--network none` (the OCRBench extraction path: `download/extracted.py`), limits from security (max members, max
  member size, total expanded size, no links / absolute / `..` paths, image types only + json/txt sidecars).

## 0. Principle: a user-added asset becomes a catalog entry, so nothing downstream forks

`check_support`, the workers, the estimates, the downloads and the cache all take a `CatalogEntry` today. A user-added
model is turned into the same `CatalogEntry` at add time (tier `unverified`, `ram_source: estimated`,
`remote_code: none`, `security: user`) and stored in the workspace, not in `configs/catalog`. One code path, one
`check_support` (ADR-021 §6); "verified" is the existing catalog.

## 1. Model: add flow (one API call, then the normal download)

`POST /api/user-assets/models {repo, revision?}` (no network in the request thread longer than a few s; the hub
calls run inside the existing egress-proxied download job, never from the app process):

1. **Resolve** (a `plan_only` hub job, the G-5 path that already exists): pin `revision` to the 40-char sha
   (`HfApi.model_info(revision=...)`), list files + sizes, read `gated`, `cardData.license`, and fetch exactly
   `config.json` (+ `preprocessor_config.json`, `generation_config.json` if present) into the stage dir. Budget: the
   small plan budget (0.05 GB) used by `verify_existing`.
2. **Parse on the host** (`download/resolver.py`, new): `json.loads` with a size cap (1 MB), depth cap, only these
   keys read: `architectures[0]`, `model_type`, `auto_map`, `torch_dtype`, `vision_config.model_type`,
   `text_config.{hidden_size,num_hidden_layers}`, `tie_word_embeddings`. Never `AutoConfig`, never an import.
   - `auto_map` present, or any `*.py` in the file list referenced by it → **MODEL-004** (blocked, ADR-021 §3).
   - `architectures[0]` not in research's family table → **MODEL-003**.
   - no `*.safetensors` → existing "no safetensors" refusal (job exit 4, now SYS-000 → a user error code here).
3. **Family → entry + template**: research's table (`configs/catalog/families.yaml`, research owns) maps an
   architecture to: `auto_class`, `attn`, freeze defaults, LoRA exclude/target, `qlora_skip_modules`, chat template
   handling, and the RAM formula. The backend renders a training template
   (`<workspace>/user_models/<id>/train.yaml`) from the family's recipe file, and a `CatalogEntry` with
   `params_b` from the safetensors index (`metadata.total_size` / dtype bytes) and `ram_gb_train` = research's
   upper-bound formula.
4. **License / gated**: the API answer carries `license` (model card id or `unknown`) and `gated`; the UI shows the
   dialog; `POST .../confirm-license` records it. Download refused before that → **SEC-005**. Gated without a token
   → **NET-005 gated_no_token**.
5. **Download**: the normal `DownloadService.start(kind="model", ...)` with the user entry (budget from the file list
   at the pinned sha, verify, install). Nothing new on the network side.

Id: `user--<org>--<name>--<sha8>` (stable, never collides with catalog ids). Storage: `<data root>/.cache/user_assets/
models/<id>.json` (entry + license confirmation + who/when), validated by pydantic on every read (hand edits fail
closed).

## 2. `model.hf_pick`: catalog OR user model

The node keeps its type and version bump 1.1.0: `model` stays the catalog enum (default unchanged); a new param
`user_model` (an id from the user list, empty by default) wins when set. The option list in `/api/node-types` gains
the user models with `option_meta.verified = false` (frontend: badge "Chưa kiểm chứng"). Workflows that name a user
model on another machine: CFG error "model not added here" with an action "add it" (repo + revision are in the id
record, so the action can re-add it).

## 3. Dataset: any HF repo (`data.hf_load` 1.1.0)

Params: `repo` (pattern = HF repo id), `revision?` (pinned at add time like models), `config?`, `split` (default
`train`), `columns` (mapping role → column: `image`, `question`, `answer` or `messages`), `max_samples`, `seed`.
`cord-v2` stays the default and keeps its loader.

1. **Inspect** (`POST /api/user-assets/datasets/inspect {repo, revision?, config?}`): a plan-only hub job lists files
   at the pinned sha. Only `*.parquet`, `*.arrow`, `*.json[l]`, and imagefolder layouts are accepted; a repo whose
   data is only a loading script (`<name>.py`) → **DATA-009** (suggest `refs/convert/parquet` when the hub has it).
   The job fetches the first parquet row group (or first 64 KB of json) of the chosen split; the host reads the
   schema with `pyarrow` (no `datasets.load_dataset`, no script) and returns columns + types + 3 sample rows
   (images as small thumbnails written by the host, text truncated).
2. **Column mapping**: auto-suggested by name/type (image feature → image; `messages` list → messages;
   question/prompt/query → question; answer/text/label/ground_truth → answer); the UI confirms. Missing → **DATA-010**,
   wrong type → **DATA-011**.
3. **Download**: the same `DownloadService` with kind dataset (budget = files of the chosen config/split only).
4. **Load**: a new loader turns the mapped columns into the same `Record` the local folder produces, so DATA-V
   (dedupe, train/test leak, image size, token length) runs unchanged.

## 4. Verified / unverified + quick trial

- `option_meta.verified` and `GET /api/node-types` carry it for models and datasets.
- Estimates for unverified models are research's upper bound; memguard + 80 % rule unchanged (ADR-005).
- "Chạy thử nhanh": the existing train node with `max_steps: 5` and `val_every_steps` off, as a template action on
  the node (no new node). The first real run of an unverified model records `peak_mem_gb`; research may later
  promote it to the catalog with measured numbers.

## 5. Security points to review (security)

1. config.json parse: size/depth caps, read keys only, no code paths (`AutoConfig`, `trust_remote_code`) anywhere in
   the user path; a test asserts `trust_remote_code` is never true for a user entry and that `auto_map` blocks.
2. Revision pinned to a sha at add time; the stored record is the only source afterwards.
3. Downloads: same proxy, budget from the pinned file list, verify, install; dataset scripts never fetched (the plan
   filters `*.py` for datasets entirely).
4. License confirmation recorded per asset; gated assets need the user's token (existing secret store).
5. Repo ids, column names, architecture strings: regex-checked and length-capped before any log or UI.
6. The user asset records live under the data root, never in `configs/` (code review never sees them, so the app
   validates them on read).

## 6. Order and estimate (backend days; research/frontend/fixcode in parallel)

| step | what | est. |
|---|---|---|
| H1 | resolver (parse config, family lookup, entry + template render) + tests | 0.5 d |
| H2 | user model store + `/api/user-assets/models` (+ license confirm) + plan-only resolve job | 0.5 d |
| H3 | `model.hf_pick` 1.1.0 (user_model), catalog/entry lookup everywhere goes through one `resolve_entry()` | 0.5 d |
| H4 | dataset inspect job + schema/sample API + column mapping + loader → Record | 1 d |
| H5 | `data.hf_load` 1.1.0, DATA-V on hub data, quick trial action | 0.5 d |
| H5b | WebDataset tar/zip: extraction job (network none, security limits) + `__key__` grouping → Record | 0.75 d |
| H5c | A1: export lineage block for unknown licence + 40 GB setting | 0.25 d |
| H6 | E2E with tester: 3 repos outside the catalog (research list), one remote-code repo blocked | 0.5 d |

Total ≈ 4.5 backend days (A1 adds ≈ 1 d); H1–H3 can start as soon as research's family table has its first family (Qwen3.5
native is enough to test the path). Depends on: research `families.yaml` + recipes, fixcode codes, frontend dialogs.

## 7. Open questions

1. ~~PM~~ DECIDED 2026-10-14: an error with the action "Thêm model này": it opens the add dialog prefilled with the
   workflow's repo + 40-char sha (the author's revision); the user still reads and ticks the licence. Never silent.
2. research: RAM upper bound formula per family (params_b × bytes + activations at the default pixel cap)?
3. security: should a user-added dataset's images also go through the existing image decode limits (DATA-V
   max pixels) before any thumbnail is written? (proposed: yes, same limits.)
