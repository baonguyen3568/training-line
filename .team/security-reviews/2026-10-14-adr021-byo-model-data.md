# ADR-021 (bring your own model/data): security checklist before code (bao-security, 2026-10-14)

User decision: any HF repo for VLM models and datasets in 1.0; `trust_remote_code` blocked outside the catalog.
This list is what the review will check; build it in from the start.

## 1. Where the network happens
- [ ] The app process never calls Hugging Face. Metadata (file list, sizes, LFS sha256, git blob ids, card license,
      gated flag) and the small config files come from a **plan-only download job** (G-5 path: egress proxy, network
      allowlist, token as a file), written to the job's run dir. The host reads them with the same no-follow, size-capped
      readers as G-5.
- [ ] One download/inspection at a time (the existing `downloads/active` lock).

## 2. Repo id and revision
- [ ] Repo id `^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$`, no `--`, no `..`; checked before any job.
- [ ] Revision: the user's ref (default `main`) is resolved to the 40-hex commit sha in the plan job, shown, and
      stored in the workflow/node params. Every later use reads that sha, never `main`. A sha that later disappears from
      the hub is an error, not a silent re-resolve.

## 3. Files of the repo
- [ ] Every sibling name passes `download_rules.name_ok` (relative, no `..`, no absolute, no `\`, no control chars,
      ≤ 512) and the allow/ignore patterns. Nothing under `.git*` or `.cache/`. Case-insensitive duplicates refused.
- [ ] Caps: ≤ 2 000 files, total within the G-5 budget (quota, free disk − reserve, confirm above the threshold).
- [ ] Models: `*.safetensors` required; `.bin/.pt/.pth/.ckpt/.pkl/.npy/.gguf/.onnx` never fetched. Load with
      `use_safetensors=True` (and vLLM `load_format="safetensors"`).
- [ ] `*.py` is never fetched for a user-added repo, whatever the patterns say.

## 4. Remote code detection (block, never guess)
Refuse with a clear catalog error ("this model needs its own code; only reviewed catalog models may run code") if ANY:
- [ ] `auto_map` anywhere (recursive, including nested `text_config` / `vision_config` / `audio_config` …) in
      `config.json`, `processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json`,
      `video_preprocessor_config.json`, `generation_config.json`;
- [ ] any value of the form `org/repo--module.Class` (cross-repo code reference);
- [ ] the repo file list contains any `*.py`;
- [ ] `architectures` / `model_type` / `processor_class` / `tokenizer_class` / `image_processor_type` not resolvable to
      a NATIVE transformers class in the train image (lookup done in a `--network none` job against that image's
      transformers, never by importing anything from the repo), or `architectures` not in research's family table;
- [ ] `trust_remote_code` appears as a key anywhere.
- [ ] Every loader call for user-added models passes `trust_remote_code=False` explicitly (test).

## 5. Parsing the JSON files (host side)
- [ ] `read_small_json` (no-follow, regular file, cap 1 MiB per config; 8 MiB for `model.safetensors.index.json`).
- [ ] Plain JSON only (`json.loads` with `parse_constant` rejecting NaN/Infinity), nesting depth ≤ 32, object keys are
      strings ≤ 256 chars.
- [ ] Only the fields we need are read, with types checked (pydantic `extra="ignore"` for the config, strict types on
      what we use). Numbers that size memory (hidden size, layers, image size, `max_pixels`…) are bounded; image limits
      are clamped to the catalog image policy, never raised by a repo.
- [ ] Parameter count for the RAM estimate from the safetensors index `metadata.total_size` (bounded int) or the
      header of each shard read in the job, never by loading weights in the app.

## 6. Chat template and processor
- [ ] Chat template (jinja) only rendered inside jobs (transformers' sandboxed environment), never in the app process.
- [ ] Processor/tokenizer files are data; a `tokenizer_class` outside the native list refuses (§4).

## 7. Licence and gated repos
- [ ] Card licence shown verbatim as plain text with the badge "not reviewed by the team"; if it is one of
      `configs/licenses/obligations.yaml` ids, show those obligations; else class `unknown`.
- [ ] The user ticks a confirmation before any weight is fetched; stored with the asset (licence string, sha, time,
      "confirmed in the UI") and carried into lineage/export. `unknown` → export/share blocked (P012).
- [ ] Gated: the plan job reports `gated`; without a token in Settings → clear error with the gated walkthrough;
      the app never accepts terms. Token only through the G-5 file path.

## 8. Datasets
- [ ] Only parquet / arrow / json / jsonl / csv / imagefolder (png/jpeg/webp). Any `*.py` loading script in the repo →
      refuse ("needs a loading script"). No tar/zip in 1.0 (webdataset stays catalog-only).
- [ ] Schema read, column suggestions and preview run in a `--network none` job (pyarrow ≥ 14.0.1), never in the app.
      Preview: ≤ 20 rows, text truncated, images decoded by `common.image_rules` and re-encoded to PNG.
- [ ] Preview text is untrusted: React plain text, no HTML/Markdown rendering.
- [ ] Row/field caps for json/csv (field ≤ 1 MiB, row count bounded by the budget).
- [ ] DATA-007 still applies (no training on data/eval), and a one-time warning: an arbitrary repo may be a benchmark
      test set.

## 9. UI
- [ ] Repo id and card text are displayed escaped; links to huggingface.co built from the validated id only.
- [ ] Badge "Chưa kiểm chứng" plus the quick trial run before a real training (ADR-021 §2); limits of ADR-005 apply.

## Decisions for the user (via PM)
1. Unknown licence (`license` missing on the card): allow the download with a warning and block export/share (my
   default), or refuse the download?
2. Default size cap for a user-added model (separate from the global quota), e.g. 40 GB unless the user raises it in
   Settings?
3. Datasets as tar/zip (webdataset) from any repo: keep out of 1.0 (my default)?
