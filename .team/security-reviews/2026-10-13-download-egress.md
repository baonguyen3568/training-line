# Job "download": egress, token, file filter, integrity, budget (bao-security, 2026-10-13)

Release 1.0 blocker (PM): the app needs a way to fetch models and datasets. The design below keeps every other job at
`--network none` and gives the download job only Hugging Face, through a proxy that enforces it.

## 1. Egress: proxy on an internal network (no host firewall change)

```
download job ──(internal net "ntc-training_egress", internal: true)──> vlm-egress-proxy ──(default net)──> HF
every other job: --network none (unchanged)
```

- `compose.yaml` declares the network `egress` with `internal: true`, so it has no route out. It also declares a service
  `egress-proxy` attached to `egress` and to the default network, and nothing else is on `egress`. The app is NOT
  on `egress`. Both the network and the proxy are static (compose), never created by a request.
- The runner attaches ONLY `job_kind == "download"` to that fixed network name (a constant in code, not a request
  field). Every other kind keeps `--network none`. Test: the docker argv for each kind.
- Job env: `HTTPS_PROXY=http://egress-proxy:3128`, `NO_PROXY=` (empty), `HF_HUB_DISABLE_XET=1` (plain HTTPS: one
  client, one proxy path; xet can come later after its proxy behaviour is checked), `HF_HUB_DISABLE_TELEMETRY=1`.
- **Proxy = our own small stdlib CONNECT proxy** (runs in the runner image, same reviewed code base; no new third-party
  image to vet). Rules:
  - only `CONNECT host:443`; plain HTTP, other ports, and anything other than CONNECT → 403;
  - host allowlist (exact or dot-suffix): `huggingface.co`, `hf.co` (covers `cdn-lfs*.hf.co`, `cdn-lfs*.huggingface.co`,
    `cas-bridge.xethub.hf.co`…). IP literals → 403;
  - the proxy resolves the name itself and refuses if ANY address is private, loopback, link-local, multicast, or
    reserved (RFC1918, 127/8, 169.254/16, ::1, fc00::/7, fe80::/10, 100.64/10). This blocks SSRF and DNS rebinding to
    the host or LAN. It connects to the address it checked, not to a second resolution;
  - no TLS interception (the token stays end-to-end). The proxy logs host, port, bytes and duration per tunnel,
    and never logs headers or payload (it cannot see them anyway);
  - **byte budget per job**: the runner registers `(job id → max_bytes)` with the proxy (over its local socket,
    not over the network). The proxy closes the job's tunnels once the bytes received pass the limit. This is the hard
    backstop of §5 and does not depend on the client.
  - container: non-root, read-only, cap_drop ALL, no-new-privileges, mem_limit small.

## 2. Token: file, not env

- Source: the HF token stored encrypted in Settings (D-7). The app decrypts it only to build ONE download submit.
- Transport: a `hf_token` field in the runner `submit`, allowed only when `job_kind == "download"` (protocol refuses
  it on any other kind). The runner audit line never contains it (audit already omits env/argv; add a test that the
  token string never appears in runner.jsonl or in the job log).
- Delivery into the job: **a file, never an env var.** `docker inspect` shows env (and our S6 evidence dumps
  Config.Env). The runner writes the token to `<runner state>/secrets/<run_id>` (tmpfs or 0700 dir, file 0400 owned
  by the job uid, O_EXCL|O_NOFOLLOW) and bind-mounts that single file read-only at `/run/secrets/hf_token`. Job env:
  `HF_TOKEN_PATH=/run/secrets/hf_token`. The runner deletes the file when the container exits (and at runner start
  for any leftovers).
- The job never prints the token. huggingface_hub does not log it, and our download module must not log headers
  or request objects. The host scan_tokens pass at job end covers run dir + logs (already a release item).
- Gated repos: the user accepts the terms on huggingface.co themself; the app never accepts for them. A 401/403 →
  catalog error "accept the model's terms on Hugging Face, then retry", no token detail.
- A token with WRITE scope: one-time warning in Settings (P003 D12); never required for download.

## 3. What is fetched: pinned revision + include/exclude

- `repo` + `revision` = 40-hex commit SHA from the catalog (models.yaml / datasets.yaml). A repo outside the catalog
  is not offered in 1.0. If a "custom repo" comes later: the host resolves `main` → SHA, shows it, and the same
  filter applies with remote code always refused.
- `allow_patterns` (models): `*.safetensors`, `*.safetensors.index.json`, `config.json`, `generation_config.json`,
  `*processor_config.json`, `tokenizer*`, `special_tokens_map.json`, `added_tokens.json`, `vocab.*`, `merges.txt`,
  `*.model` (sentencepiece), `*.tiktoken`, `chat_template.*`, `*.jinja`, `README.md`, `LICENSE*`, `NOTICE*`.
  `*.py` ONLY when the repo+revision is in `configs/security/remote_code_allowlist.yaml`, and then only the listed
  files.
- `allow_patterns` (datasets): `*.parquet`, `*.arrow`, `*.jsonl`, `*.json`, `*.csv`, `*.tar` (webdataset shards),
  `*.png|*.jpg|*.jpeg|*.webp`, `README.md`, `LICENSE*`. No loading scripts (`*.py`) ever.
- `ignore_patterns` (always, wins over allow): `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.pickle`, `*.joblib`,
  `*.npy`, `*.npz`, `*.h5`, `*.msgpack`, `*.gguf`, `*.onnx`, `*.zip`, `*.7z`, `*.tar.gz`, `*.tgz`, `*.sh`, `*.exe`,
  `*.so`, `.git*`, `*.py` (unless allowlisted above).
- Models without `*.safetensors` → refused before download (security-model-vetting A.3; donut-base needs the offline
  conversion path instead).

## 4. Where it lands: staging, then host verification and move

- The job's only writable mount is its run dir. It downloads into `<run dir>/stage/` with
  `local_dir=<run dir>/stage/files`, `cache_dir=<run dir>/stage/cache`. It never writes into the shared
  `.cache/hf/hub`, where a hostile job could otherwise poison another repo's blobs.
- After exit 0 the HOST (app process, never the job):
  1. walks `stage/files` with the no-follow tree check: regular files only, no symlink/fifo/device, names match the
     allow patterns and not the ignore ones, no file outside the expected list;
  2. checks every LFS file's sha256 against the hub metadata `lfs.sha256` fetched by the job and recorded in
     `stage/manifest.json` (integrity). Where the repo has a pin (`configs/models/pinned/*.sha256`, dataset
     `files_sha256`), it checks against the PIN instead (authenticity). Any mismatch → refuse, keep nothing;
  3. moves the verified files into the canonical location (`.cache/hf/hub/<models|datasets>--org--name/` in the
     HF layout, or `.cache/prepared/<bundle>`) with rename on the same filesystem. A target that already exists is
     verified again, never overwritten blindly.
- `stage/` is then deleted (or left to retention on failure).

## 5. Size budget

- Before any blob is fetched, the job writes the plan (file list + sizes from `model_info(files_metadata=True)`
  after the filter), and the host checks: `planned_bytes ≤ min(user download quota from Settings, free disk of the
  data root − reserve)`. Reserve = max(50 GB, 10 % of the disk) on a shared host (`shared_dev`), 20 GB on `personal`.
- Above the confirm threshold (Settings, default 20 GB) the UI asks for a typed confirmation and shows the size. On
  `shared_dev` the admin policy can cap it (`[download] max_gb`). Env may only lower it, like the other policy keys.
- The proxy enforces `planned_bytes × 1.05 + 100 MB` per job as the hard limit (§1). The job is then killed with a
  catalog error, and the staging dir goes to retention.
- One download job at a time (it does not need the GPU lock, but it holds a download lock, so two big pulls never
  race for disk).

## 6. Tests the PR must carry

- argv: only kind `download` has `--network ntc-training_egress`, all others `--network none`; the token is a
  mounted file, never in `-e`.
- protocol: `hf_token` on a non-download kind → refused; the token never appears in runner.jsonl or logs.
- proxy: non-443, plain HTTP, IP literal, a non-HF host, a host resolving to 127.0.0.1 / 10.x / 169.254.x → refused;
  the byte cap closes the tunnel.
- staging: a symlink, an `*.bin`, an unexpected file, or a sha mismatch → nothing moved; the happy path moves into
  the HF layout.
- budget: plan over the quota or free disk → refused before any blob.
