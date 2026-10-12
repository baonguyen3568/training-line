# Pinned remote-code hashes

Owner: **bao-security**. Each `<name>@<sha8>.sha256` lists the sha256 of every `.py` file security reviewed for that
repo at that exact commit (format: `sha256sum` output). `src/vlm_pipeline/train/prepare.py` refuses to prepare a
model if any `.py` is unvetted, missing or mismatched.

Rules:
- Changing or adding a file here = a new security review. Commit message must reference the review record in
  `.team/security-reviews/`. Never regenerate these files from a fresh download "to make it pass".
- One file per repo@commit, including *bridged* repos pulled via `auto_map` (`org/repo--module.Class`).

| File | Repo @ commit | Review |
|---|---|---|
| `nemotron-parse-v1.1@fd1e3c6b.sha256` | nvidia/NVIDIA-Nemotron-Parse-v1.1@fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4 | `.team/security-reviews/2026-10-12-nvidia-template-candidates.md` |
| `c-radiov2-h@0d8f4c18.sha256` | nvidia/C-RADIOv2-H@0d8f4c18c877166eda07ddae1386bcad256b7a6a | same |
| `nemotron-parse-2.0@b6742064.sha256` | nvidia/NVIDIA-Nemotron-Parse-2.0@b6742064f4a8cf22a10383ece5e7fbead355ac04 (top-level `.py` only; `logitsprocs/` is vLLM-only and not loaded) | same; re-confirmed 2026-10-12: still `main`, C-RADIOv2-H still @0d8f4c18 |
| `nemotron-parse-2.0@b6742064.vllm-extras.sha256` | same repo: `vllm_tied_patch/sitecustomize.py`, `logitsprocs/nemotron_parse_vllm_logitprocs.py` (vLLM serve only) | reviewed: benign (monkey-patches vLLM `load_weights` to tie lm_head; logits processor). **Rule:** never put an HF-cache directory on `PYTHONPATH`; vendor these files into `third_party/patches/vllm/` and verify against this list |

## Rule: no downloaded code on `PYTHONPATH`
A `sitecustomize.py` on `PYTHONPATH` runs automatically at **every** Python interpreter start. Model READMEs that say
"download X and `export PYTHONPATH=…`" must instead be handled by vendoring the reviewed file into the repo
(`third_party/patches/<tool>/`), hash-verified against `configs/models/pinned/`, and mounted read-only only into the
container that needs it (e.g. the vLLM serve container), never on the host.
