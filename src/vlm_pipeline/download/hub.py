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

Resolve mode (``"resolve": true``, ADR-021 / P021 H2, a user-added repo): ``revision`` may be a ref (``main``, a
branch, a tag); the job pins it to the commit sha, writes ``stage/resolve.json`` (sha, gated, card licence, EVERY file
name for the remote-code check, the filtered plan and its total) and fetches only the small JSON files the resolver
reads (``download.resolver.CONFIG_FILES`` + the safetensors index) into ``stage/meta/``, each under its size cap.
A hub refusal writes ``stage/error.json`` ``{reason}`` (NET-005) and exits ``EXIT_HUB``. No weight is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from vlm_pipeline.common.download_rules import Kind, check_repo, name_ok, wanted
from vlm_pipeline.common.runinfo import read_small_json

SPEC_NAME = "download_spec.json"
EXIT_OVER_BUDGET = 3
EXIT_NO_SAFETENSORS = 4
EXIT_BAD_SPEC = 5
EXIT_HUB = 6  # resolve: the hub refused (not found, gated, network): stage/error.json says why
_REF = re.compile(r"^(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")  # a branch / tag / sha the user typed
META_FILES = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)  # = download.resolver.CONFIG_FILES (the job image does not import the host-side resolver)
META_MAX = 1 << 20
TOKENIZER_CONFIG_MAX = 4 << 20  # added_tokens + chat template: a few hundred KB in real VLM repos
INDEX_NAME = "model.safetensors.index.json"
INDEX_MAX = 8 << 20


def meta_caps() -> dict[str, int]:
    """Each small JSON the resolver reads, with its size cap (the host reads them with the same caps)."""
    return {**dict.fromkeys(META_FILES, META_MAX), "tokenizer_config.json": TOKENIZER_CONFIG_MAX, INDEX_NAME: INDEX_MAX}


MAX_REPO_FILES = 2000  # security checklist §3
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def read_spec(run_dir: Path) -> dict[str, Any]:
    spec = read_small_json(run_dir / SPEC_NAME, max_bytes=512 * 1024)  # only_files: up to 2 000 names
    if not isinstance(spec, dict) or spec.get("kind") not in ("model", "dataset"):
        raise ValueError("bad spec")
    if spec.get("resolve") is True:  # a ref the job pins; the repo id rules are the same
        check_repo(str(spec.get("repo")), "0" * 40)
        ref = str(spec.get("revision"))
        if not _REF.fullmatch(ref):
            raise ValueError("bad ref")
        if ref.startswith("refs/pr/"):  # anyone can open a hub PR: never pin unmerged code under the repo's name
            raise ValueError("an unmerged pull request ref is not a revision of the repo")
        for key in ("config", "split"):  # a dataset's choice (H4): a plain name, or absent
            value = spec.get(key)
            if value is not None and not (isinstance(value, str) and _NAME.fullmatch(value)):
                raise ValueError(f"bad {key}")
    else:
        check_repo(str(spec.get("repo")), str(spec.get("revision")))
    budget = spec.get("max_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("bad budget")
    only = spec.get("only_files")  # a user dataset: the files of ONE split, from the host's checked record
    if only is not None and not (isinstance(only, list) and 0 < len(only) <= MAX_REPO_FILES
                                 and all(isinstance(n, str) and name_ok(n) for n in only)):  # fmt: skip
        raise ValueError("bad only_files")
    return spec


def plan_files(
    siblings: Iterable[Any], kind: Kind, remote_code_files: Iterable[str], *, archives: bool = False
) -> list[dict[str, Any]]:
    """The hub's file list filtered by the shared rules, with size and LFS sha256 (None for small git files)."""
    allowed = tuple(remote_code_files)
    plan = []
    for sib in siblings:
        name = getattr(sib, "rfilename", "")
        if not wanted(name, kind, allowed, archives=archives):
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


def _stop(code: int, why: str) -> int:
    """Say why on stderr (the run log) before a refusing exit: never a silent exit code (tester 2026-10-14)."""
    print(f"download.hub: {why} (exit {code})", file=sys.stderr, flush=True)
    return code


def hub_failure_reason(exc: BaseException, *, has_token: bool) -> str:
    """NET-005 reason of a hub exception, by class name (the host never imports huggingface_hub to classify)."""
    name = type(exc).__name__
    if name == "GatedRepoError":
        return "gated_not_accepted" if has_token else "gated_no_token"
    if name in ("RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError"):
        # Without a token the hub answers 401 for a private repo exactly as for a missing one.
        return "not_found" if has_token else "not_found_or_private"
    return "network"


def _card_license(info: Any) -> str | None:
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    value = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    return value if isinstance(value, str) else None


def _dataset_choice(
    spec: dict[str, Any], api: Any, files: list[dict[str, Any]], names: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """A dataset's configs / splits / chosen pair / format and the sample to fetch (download.dataset_layout)."""
    from vlm_pipeline.download.dataset_layout import choose, has_script, layout, sample_files

    sizes = {f["name"]: f["size"] for f in files}
    tree = layout(sizes)  # planned files only: a *.py is never planned (its presence is has_script)
    try:  # the hub's own parquet conversion (refs/convert/parquet): DATA-009 can offer it instead of a script
        refs = api.list_repo_refs(spec["repo"], repo_type="dataset")
        export = any(getattr(r, "ref", "") == "refs/convert/parquet" for r in getattr(refs, "converts", None) or ())
    except Exception:  # noqa: BLE001 - optional: absent means "not offered"
        export = False
    block: dict[str, Any] = {
        "configs": sorted(tree), "has_script": has_script(names), "has_parquet_export": export,
        "config": None, "split": None, "splits": [], "format": "unknown", "sample": [],
    }  # fmt: skip
    if not tree:
        return block, []
    config, split = choose(tree, spec.get("config"), spec.get("split"))
    fmt, sample = sample_files(tree[config][split], sizes, sizes)
    block.update(config=config, split=split, splits=sorted(tree[config]), format=fmt, sample=sample,
                 split_bytes=sum(sizes.get(f.name, 0) for f in tree[config][split]))  # fmt: skip
    return block, sample


def resolve(run_dir: Path, spec: dict[str, Any], *, api: Any, fetch: Callable[..., str], token: str | None) -> int:
    """Resolve mode (module docstring): pin the ref, list every file, fetch only the small JSON files."""
    kind: Kind = spec["kind"]
    stage = run_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        info = (api.model_info if kind == "model" else api.dataset_info)(
            spec["repo"], revision=spec["revision"], files_metadata=True
        )
    except Exception as exc:  # noqa: BLE001 - every hub refusal becomes a reason, never a traceback
        reason = hub_failure_reason(exc, has_token=token is not None)
        (stage / "error.json").write_text(json.dumps({"reason": reason}), encoding="utf-8")
        return _stop(EXIT_HUB, f"the hub refused {spec['repo']}: {reason}")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return _stop(EXIT_BAD_SPEC, "the hub returned no commit sha")
    siblings = list(info.siblings or ())
    if len(siblings) > MAX_REPO_FILES:
        (stage / "error.json").write_text(json.dumps({"reason": "too_many_files"}), encoding="utf-8")
        return _stop(EXIT_HUB, f"the repo has {len(siblings)} files (over {MAX_REPO_FILES})")
    names = sorted(str(getattr(sib, "rfilename", "")) for sib in siblings)
    files = plan_files(siblings, kind, (), archives=kind == "dataset")  # never a *.py for a user-added repo (§3)
    gated = getattr(info, "gated", False)
    sizes = {f["name"]: f["size"] for f in files}
    caps = meta_caps() if kind == "model" else {}
    # A file the resolver must read that is there but too big (or not planned) is NEVER "absent": the host refuses
    # the repo (security H2: an oversized tokenizer_config.json could hide an auto_map).
    unreadable = sorted(n for n, cap in caps.items() if n in names and (n not in sizes or sizes[n] > cap))
    meta = {"sha": sha, "gated": bool(gated), "license": _card_license(info), "all_files": names,
            "files": files, "total_bytes": sum(f["size"] for f in files), "unreadable": unreadable}  # fmt: skip
    sample: list[str] = []
    if kind == "dataset":
        meta["dataset"], sample = _dataset_choice(spec, api, files, names)
    (stage / "resolve.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for name in sample:  # one bounded file of the chosen split (imagefolder: its metadata + a few images)
        fetch(repo_id=spec["repo"], filename=name, revision=sha, repo_type=kind, local_dir=str(stage / "sample"),
              cache_dir=str(stage / "cache"), token=token)  # fmt: skip
    shutil.rmtree(stage / "sample" / ".cache" / "huggingface", ignore_errors=True)
    for name in caps:
        if name not in sizes or name in unreadable:
            continue
        fetch(repo_id=spec["repo"], filename=name, revision=sha, repo_type=kind, local_dir=str(stage / "meta"),
              cache_dir=str(stage / "cache"), token=token)  # fmt: skip
    shutil.rmtree(stage / "meta" / ".cache" / "huggingface", ignore_errors=True)
    shutil.rmtree(stage / "cache", ignore_errors=True)
    return 0


def run(run_dir: Path, *, api: Any = None, fetch: Callable[..., str] | None = None) -> int:  # noqa: PLR0911 - one exit per refusal
    """The whole job; ``api`` / ``fetch`` are injectable (unit tests never touch the network)."""
    try:
        spec = read_spec(run_dir)
    except (OSError, ValueError) as exc:
        return _stop(EXIT_BAD_SPEC, f"bad download spec: {exc}")
    kind: Kind = spec["kind"]
    token = _token()
    if api is None or fetch is None:  # pragma: no cover - the real hub, inside the job only
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]  # container-only

        api, fetch = HfApi(token=token), hf_hub_download
    if spec.get("resolve") is True:
        return resolve(run_dir, spec, api=api, fetch=fetch, token=token)
    info = (api.model_info if kind == "model" else api.dataset_info)(
        spec["repo"], revision=spec["revision"], files_metadata=True
    )
    files = plan_files(info.siblings or (), kind, spec.get("remote_code_files") or (),
                       archives=spec.get("archives") is True and kind == "dataset")  # fmt: skip
    if spec.get("only_files") is not None:
        wanted_names = set(spec["only_files"])
        files = [f for f in files if f["name"] in wanted_names]
        if {f["name"] for f in files} != wanted_names:  # a file the record names is gone or not allowed: refuse
            return _stop(EXIT_BAD_SPEC, "a file of the split is not in the repo at this revision")
    stage = run_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for f in files)
    (stage / "plan.json").write_text(json.dumps({"files": files, "total_bytes": total}, indent=1), encoding="utf-8")
    if kind == "model" and not any(f["name"].endswith(".safetensors") for f in files):
        return _stop(EXIT_NO_SAFETENSORS, "no .safetensors weights in the repo")  # pickle-only: never (security A.3)
    if spec.get("plan_only") is True:  # "Check": the host compares an existing snapshot with this plan, no blob
        return 0
    if total > spec["max_bytes"]:
        return _stop(EXIT_OVER_BUDGET, f"the planned files are {total} bytes, over this job's budget of "
                                       f"{spec['max_bytes']} bytes (the catalog size is too small)")  # fmt: skip
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
    # huggingface_hub keeps its own bookkeeping under <local_dir>/.cache/huggingface (metadata, locks, CACHEDIR.TAG):
    # not part of the repo. Removed here so files/ holds exactly the plan (PM 2026-10-14: the host's check refused
    # 21 extra files with NET-003). The host check stays exact: anything else left behind is still refused.
    shutil.rmtree(out / ".cache" / "huggingface", ignore_errors=True)
    if (out / ".cache").is_dir() and not any((out / ".cache").iterdir()):
        (out / ".cache").rmdir()
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
