---
name: security-model-vetting
description: Use before downloading, loading or adding to a config ANY model checkpoint, tokenizer/processor, container image or dataset from Hugging Face / NGC / elsewhere — and whenever code sets trust_remote_code=True, calls torch.load/pickle, or bumps a model revision or image tag. Verifies official source, pins commit SHA / image digest, enforces safetensors-only and reviews remote code.
---

# Model / image / dataset vetting (supply chain gate)

Knowledge & sources: `.team/knowledge/security.md` §2–§4. Output of this skill = entries in
`configs/security/remote_code_allowlist.yaml`, `configs/security/licenses.yaml`, `docker/IMAGES.lock`.

## A. Hugging Face model / processor

1. **Source is official?** Org must be the publisher (`nvidia/`, `Qwen/`, `OpenGVLab/`, `google/`,
   `meta-llama/`, `microsoft/`, `HuggingFaceM4/`, `allenai/`…). Ask bao-research if unsure. Re-uploads → reject.
2. **Resolve the commit SHA** (never pin `main`):
   ```bash
   uvx --from huggingface_hub python -I -c "
   from huggingface_hub import HfApi; import sys
   i = HfApi().model_info(sys.argv[1], files_metadata=True)
   print('sha', i.sha); print('gated', i.gated); print('license', (i.card_data or {}).get('license'))
   for s in i.siblings: print(f'{s.size or 0:>14}  {s.rfilename}')
   " <org/repo>
   ```
3. **File inventory check** (from listing above):
   - Has `*.safetensors`? If NO → stop; either reject or convert in a `--network none` sandbox (PM sign-off).
   - Any `*.bin *.pt *.pth *.ckpt *.pkl *.npy(object) *.joblib *.7z *.zip`? → exclude via `allow_patterns`; never load.
   - Any `*.py` / `auto_map` in `config.json` (INCLUDING nested sub-configs: `vision_config`, `audio_config`, `llm_config`…) / `processor_config.json` / `tokenizer_config.json`? → remote code, go to step 4.
   - `auto_map` value of form `org/other-repo--module.Class` = code pulled from ANOTHER repo at `main` (unpinned; transformers does NOT pass `revision` to the other repo — verified 5.12.1; real case: nvidia/C-RADIOv2-H main changed 2026-01-30 after Parse/Nano-VL shipped) → vet that repo too and pin by rewriting the auto_map prefix to a local hash-verified dir (`/abs/dir--module.Class`) + `HF_HUB_OFFLINE=1`. See `.team/security-reviews/2026-10-12-nemotron-3-nano-omni.md`.
   - Total size > 20GB → tell PM before download (CLAUDE.md).
4. **trust_remote_code review** (only if no native `transformers` class exists — check
   `transformers` version in our container first):
   - Download only code files at the SHA: `allow_patterns=["*.py","*.json"]`.
   - Read every `.py`. Red flags: `os.system`, `subprocess`, `eval/exec`, `socket`, `requests`/`urllib` at import,
     `pickle`/`torch.load` without `weights_only=True`, base64/zlib blobs, writing outside cache, env-var reads (`*_TOKEN`, `*_KEY`),
     `trust_remote_code=True` passed further down, dynamic `importlib` of remote modules.
   - Record in `configs/security/remote_code_allowlist.yaml`:
     ```yaml
     - repo: OpenGVLab/InternVL3_5-8B
       revision: <40-char sha>
       files: {modeling_internvl_chat.py: sha256:<...>, ...}   # sha256sum of each .py
       reviewer: bao-security
       date: 2026-10-12
       notes: "no network/exec; only torch/transformers imports"
     ```
   - Any SHA change ⇒ re-review (diff old vs new `.py`).
5. **Download** (pinned, safetensors only):
   ```python
   snapshot_download(repo_id, revision=SHA, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"] + (["*.py"] if allowlisted else []))
   ```
6. **First load offline**: run the first `from_pretrained` inside the container with `--network none`
   (`HF_HUB_OFFLINE=1`), `use_safetensors=True`, `trust_remote_code` only per allowlist.
7. Optional extra layer (never a substitute for 1–6): `uvx modelscan -p <dir>`. Scanners are bypassable (picklescan CVE-2025-10155..7).

## B. NGC / container image

```bash
docker buildx imagetools inspect nvcr.io/nvidia/<image>:<tag>        # find linux/arm64 manifest digest
docker buildx imagetools inspect --raw nvcr.io/nvidia/<image>:<tag> | python3 -I -m json.tool | grep -B2 -A6 arm64
```
- Registry must be `nvcr.io/nvidia/*` (or other approved official). No `:latest`.
- Append to `docker/IMAGES.lock`: `<name> <tag> <arm64 digest> <date> <reviewer> <ngc page url>`.
- Reference in Dockerfile/compose as `image@sha256:<digest>` (tag kept as comment).
- Scan (report, not hard gate), no docker socket (`:ro` on a socket does not limit the API): `docker save <ref> -o <scratch>/img.tar`
  then `docker run --rm -v <scratch>:/scan:ro aquasec/trivy@sha256:<pinned> image --input /scan/img.tar --severity CRITICAL,HIGH --ignore-unfixed`.
- Host check: `nvidia-ctk --version` — must be ≥ 1.19.1 (CVE-2026-24260). If lower: only pinned official images with `--gpus`.

## C. Dataset
- Official source + revision SHA (HF datasets are git repos too) + license entry (use `security-license-check`).
- Prefer parquet/arrow/webdataset tar of images; reject dataset loading scripts (`*.py` with `trust_remote_code`) unless reviewed like A.4.
- Record shard sha256 manifest after curation.
- Eval benchmarks: eval-only, must not enter train mix.

## Verdict format (reply to requester + log line)
`VET <id>@<sha|digest>: APPROVED | APPROVED-WITH-CONDITIONS (<list>) | REJECTED (<reason>)`
Append to `.team/log/YYYY-MM-DD.md`: `HH:MM [bao-security → <who>] VET ...`.

## D. Python packages: network at import ("phone home")
Some packages call the network when imported (update checks, telemetry). Real case: albumentations 2.0.8
`__init__.py` -> `check_for_updates()` fetches pypi.org at import (gate: `NO_ALBUMENTATIONS_UPDATE=1`).
Check every new package:
```bash
docker run --rm --network none --read-only --cap-drop ALL --user 1000:1000 --entrypoint bash <image> -c \
  '/opt/venv/bin/python -X importtime -c "import <pkg>" 2>&1 | grep -iE "urlopen|fetch|resolution|update|telemetry"'
grep -rnE "urlopen|requests\.(get|post)|http.client|socket\.|getaddrinfo" <site-packages>/<pkg>/__init__.py <site-packages>/<pkg>/*version*.py
```
Baseline env for every image/worker: `NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=offline`.
Probe tip (bao-tester): count `socket.getaddrinfo` + `connect` in a fresh interpreter, with a CONTROL run without the env var
(must show >= 1 lookup) — counting only connect gives 0 under `--network none` because DNS fails first.
Limit: libraries with a native network process (wandb >= 0.2x uses the Go `wandb-core`) are invisible to Python socket probes;
for those rely on `--network none` + `WANDB_MODE=offline|disabled`, never on the probe.
