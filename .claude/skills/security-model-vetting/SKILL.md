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
   - Any `*.py` / `auto_map` in `config.json` / `processor_config.json` / `tokenizer_config.json`? → remote code, go to step 4.
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
- Scan (report, not hard gate): `docker run --rm -v /var/run/docker.sock:/var/run/docker.sock:ro aquasec/trivy@sha256:<pinned> image --severity CRITICAL,HIGH --ignore-unfixed <ref>`
  (the trivy container is the ONLY one allowed the docker socket, read-only, and only for scanning).
- Host check: `nvidia-ctk --version` — must be ≥ 1.19.1 (CVE-2026-24260). If lower: only pinned official images with `--gpus`.

## C. Dataset
- Official source + revision SHA (HF datasets are git repos too) + license entry (use `security-license-check`).
- Prefer parquet/arrow/webdataset tar of images; reject dataset loading scripts (`*.py` with `trust_remote_code`) unless reviewed like A.4.
- Record shard sha256 manifest after curation.
- Eval benchmarks: eval-only, must not enter train mix.

## Verdict format (reply to requester + log line)
`VET <id>@<sha|digest>: APPROVED | APPROVED-WITH-CONDITIONS (<list>) | REJECTED (<reason>)`
Append to `.team/log/YYYY-MM-DD.md`: `HH:MM [bao-security → <who>] VET ...`.
