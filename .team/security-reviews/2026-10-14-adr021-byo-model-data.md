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

## 10. Archives (WebDataset tar/zip), user decision A1 (2026-10-14): thresholds
Principle: **do not extract when you can stream.** WebDataset shards are read sequentially; the conversion job
reads each member, applies the rules below and writes OUR records (+ images re-encoded by `common.image_rules`) into
its run dir. Nothing from the archive is written under its own name. If a real extraction is ever needed, the same
rules apply and every file is created through a dir fd with `O_CREAT|O_EXCL|O_NOFOLLOW` in a fresh dir.
Where: a job with `--network none`, archive mounted read-only, rw = its run dir only, swap-exempt CPU job
(≤ 4 GB, `--memory-swap = --memory`, `--pids-limit`), wall-clock timeout.

| Rule | Threshold / behaviour |
|---|---|
| Container formats | `.tar` (preferred), `.tar.gz`/`.tgz`, `.zip` (stored or deflate). Anything else (7z, rar, xz, zstd, bz2, nested archives) → refuse. An archive inside an archive is never opened (skipped, counted). |
| Member types | Regular files only (tar REGTYPE/AREGTYPE; zip entries whose mode is not S_IFLNK/S_IFCHR/S_IFBLK/S_IFIFO). Directories are ignored. Symlink, hardlink, device, FIFO, GNU sparse, tar "contiguous" → **refuse the whole shard**. |
| Names | Relative POSIX only: no leading `/`, no `..` component, no `\`, no drive letter (`C:`), no NUL/control chars, each component ≤ 255 bytes, full path ≤ 512; NFC-normalised; duplicates (also case-insensitive after NFC) in one shard → refuse. Tar PAX/GNU long-name headers are fine; the resolved name is what is checked. |
| Extensions kept | Images `.jpg .jpeg .png .webp`; text/labels `.txt .json .jsonl .cls .caption .text`. Any other extension is skipped (counted in the report), never written. |
| Per member | ≤ 20 MiB image (same as uploads), ≤ 16 MiB json/jsonl, ≤ 1 MiB other text. Images must open under `image_rules` (40 MP, bomb = error) or the sample is dropped. |
| Per shard | ≤ 200 000 members, uncompressed ≤ 8 GiB. |
| Whole dataset | ≤ 2 000 000 samples; uncompressed total ≤ the download budget × 3 and ≤ quota/free-disk rules of G-5 (checked before writing, re-checked while streaming). |
| Compression ratio | zip: per member `file_size / compress_size` ≤ 100, archive total ≤ 20; gzip tar: bytes out / bytes in ≤ 20, tracked while reading, abort at the limit (bomb). |
| Zip specifics | Reject encrypted entries, multi-disk, and archives whose central directory disagrees with local headers (sizes/names) or that overlap (zip-in-zip "quines"). Use `zipfile` only through the per-member checks, never `extractall`. |
| Errors | Any refusal → the shard is rejected with a reason code, nothing of that shard is kept; the report lists shard + reason, never file contents. |

Tests the PR must carry: `../x`, `/abs`, `a\b`, symlink, hardlink, device, duplicate names, case-only duplicate, a
42.zip-style ratio bomb, a gzip bomb, a nested archive, an unknown extension, an over-size image, an encrypted zip.

## 11. After 1.0 (licence backlog)
- `qwen-research` (Qwen RESEARCH LICENSE AGREEMENT, e.g. Qwen2.5-VL-3B-Instruct): vendor the text, add it to
  `obligations.yaml` as `non_commercial`, and only then let a card with no `license` but a known `license_name` count.
  Until then such a card is `unknown` (export blocked by SEC-006) and the UI shows the name from `license_card`.
  Reason: moving unknown → non_commercial unblocks export, so it needs the text read first (decided 2026-10-14).
