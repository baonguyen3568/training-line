"""The download job (P018, security download-egress): fetch one pinned catalog repo into its own run dir.

``python -m vlm_pipeline.download.hub --run-dir /workspace/outputs/runs/<run_id>`` inside the approved train image,
started only by vlm-runner (kind ``download``: the egress proxy is its only way out, the HF token a mounted file).
It reads ``<run_dir>/download_spec.json`` (written by the host from the CATALOG: repo, 40-hex revision, kind, extra
remote-code files, byte budget) and writes, all inside ``<run_dir>/stage/``:

* ``plan.json``: the filtered file list with sizes and LFS sha256 from the hub, BEFORE any blob (the host and the
  proxy budget check it); over the budget -> exit ``EXIT_OVER_BUDGET``, nothing fetched;
* ``files/``: the files (plain files, ``local_dir``); ``progress.jsonl``: one line per finished file;
* ``manifest.json``: name, size and the sha256 of what landed (the host re-hashes anyway: the job is not trusted).

Never logs the token, headers or request objects. The host decides what is installed (``download.install``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from vlm_pipeline.common.download_rules import Kind, check_repo, wanted
from vlm_pipeline.common.runinfo import read_small_json

SPEC_NAME = "download_spec.json"
EXIT_OVER_BUDGET = 3
EXIT_NO_SAFETENSORS = 4
EXIT_BAD_SPEC = 5


def read_spec(run_dir: Path) -> dict[str, Any]:
    spec = read_small_json(run_dir / SPEC_NAME, max_bytes=64 * 1024)
    if not isinstance(spec, dict) or spec.get("kind") not in ("model", "dataset"):
        raise ValueError("bad spec")
    check_repo(str(spec.get("repo")), str(spec.get("revision")))
    budget = spec.get("max_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("bad budget")
    return spec


def plan_files(siblings: Iterable[Any], kind: Kind, remote_code_files: Iterable[str]) -> list[dict[str, Any]]:
    """The hub's file list filtered by the shared rules, with size and LFS sha256 (None for small git files)."""
    allowed = tuple(remote_code_files)
    plan = []
    for sib in siblings:
        name = getattr(sib, "rfilename", "")
        if not wanted(name, kind, allowed):
            continue
        lfs = getattr(sib, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        size = getattr(sib, "size", None) or (getattr(lfs, "size", None) if lfs is not None else None) or 0
        blob = getattr(sib, "blob_id", None)  # git object id: checks the small non-LFS files (configs, tokenizer)
        plan.append({"name": name, "size": int(size), "sha256": sha if isinstance(sha, str) else None,
                     "git_sha1": blob if isinstance(blob, str) and lfs is None else None})  # fmt: skip
    return sorted(plan, key=lambda f: f["name"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token() -> str | None:
    path = os.environ.get("HF_TOKEN_PATH")
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def run(run_dir: Path, *, api: Any = None, fetch: Callable[..., str] | None = None) -> int:
    """The whole job; ``api`` / ``fetch`` are injectable (unit tests never touch the network)."""
    try:
        spec = read_spec(run_dir)
    except (OSError, ValueError):
        return EXIT_BAD_SPEC
    kind: Kind = spec["kind"]
    token = _token()
    if api is None or fetch is None:  # pragma: no cover - the real hub, inside the job only
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]  # container-only

        api, fetch = HfApi(token=token), hf_hub_download
    info = (api.model_info if kind == "model" else api.dataset_info)(
        spec["repo"], revision=spec["revision"], files_metadata=True
    )
    files = plan_files(info.siblings or (), kind, spec.get("remote_code_files") or ())
    stage = run_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for f in files)
    (stage / "plan.json").write_text(json.dumps({"files": files, "total_bytes": total}, indent=1), encoding="utf-8")
    if kind == "model" and not any(f["name"].endswith(".safetensors") for f in files):
        return EXIT_NO_SAFETENSORS  # pickle-only weights are never fetched (security A.3)
    if spec.get("plan_only") is True:  # "Check": the host compares an existing snapshot with this plan, no blob
        return 0
    if total > spec["max_bytes"]:
        return EXIT_OVER_BUDGET
    out = stage / "files"
    done = 0
    with (stage / "progress.jsonl").open("a", encoding="utf-8") as progress:
        for n, f in enumerate(files, start=1):
            fetch(repo_id=spec["repo"], filename=f["name"], revision=spec["revision"], repo_type=kind,
                  local_dir=str(out), cache_dir=str(stage / "cache"), token=token)  # fmt: skip
            done += f["size"]
            progress.write(json.dumps({"ts": time.time(), "files_done": n, "files_total": len(files),
                                       "bytes_done": done, "bytes_total": total}) + "\n")  # fmt: skip
            progress.flush()
    manifest = [{"name": f["name"], "size": (out / f["name"]).stat().st_size, "sha256": _sha256(out / f["name"])}
                for f in files]  # fmt: skip
    (stage / "manifest.json").write_text(json.dumps({"files": manifest}, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - the job entrypoint
    parser = argparse.ArgumentParser(prog="python -m vlm_pipeline.download.hub")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.run_dir)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
