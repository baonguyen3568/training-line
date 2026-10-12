"""Print vetting-relevant Hugging Face metadata for model repos.

Reports SHA, license tag, safetensors size, .py / pickle files, and every ``auto_map`` and
``_name_or_path`` found recursively in config.json (root AND every sub-config).

Usage: python3 -I scripts/research/hf_model_meta.py org/repo [org/repo ...]

An ``auto_map`` value such as ``other-org/other-repo--module.Class`` or a foreign ``_name_or_path``
means code is pulled from ANOTHER repo (at ``main`` unless snapshotted): flag it to bao-security.
Also grep the repo's .py for ``get_class_from_dynamic_module`` / ``trust_remote_code``.
Gated repos return no config without an accepted token.

Exit code: 0 if every repo was fully read, 1 if any repo id was invalid or its metadata/config could
not be fetched (a vetting tool must not succeed silently), 2 on usage error.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import Any

HF = "https://huggingface.co"
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".tar")
# org/name; a segment may not start with "." (rejects "..", "../x", hidden names) - stricter than a plain charset
REPO_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def fetch_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host, validated repo id
            data: dict[str, Any] = json.load(resp)
            return data
    except Exception as exc:  # noqa: BLE001 - report any fetch failure inline
        return {"_err": str(exc)}


def auto_map_targets(value: Any) -> list[str] | str:
    """Return foreign repos referenced by an auto_map dict, or 'local' when all classes live in this repo."""
    if not isinstance(value, dict):
        return "local"
    repos = sorted({str(v).split("--")[0] for v in value.values() if "--" in str(v)})
    return repos or "local"


def collect_bridges(node: Any, path: str, out: dict[str, Any]) -> None:
    """Record every auto_map (root included) and every sub-config _name_or_path in a config tree."""
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        where = f"{path}.{key}" if path else key
        if key == "auto_map":
            out[where] = auto_map_targets(value)
        elif key == "_name_or_path" and path and value:
            out[where] = value
        collect_bridges(value, where, out)


def describe(repo_id: str) -> tuple[str, bool]:
    """Return (report, ok). ok is False when the repo id is invalid or meta/config could not be read."""
    if not REPO_ID_RE.match(repo_id):
        return f"{repo_id} ERR invalid repo id (expected org/name)", False
    meta = fetch_json(f"{HF}/api/models/{repo_id}?blobs=true")
    sha = meta.get("sha")
    if "_err" in meta or not sha:
        return f"{repo_id} ERR metadata: {meta.get('_err', 'missing sha')}", False
    files = meta.get("siblings", [])
    size_gb = sum(f.get("size") or 0 for f in files if f["rfilename"].endswith(".safetensors")) / 1e9
    py_files = [f["rfilename"] for f in files if f["rfilename"].endswith(".py")]
    pickles = [f["rfilename"] for f in files if f["rfilename"].endswith(PICKLE_SUFFIXES)]
    cfg = fetch_json(f"{HF}/{repo_id}/resolve/{sha}/config.json")
    bridges: dict[str, Any] = {}
    collect_bridges(cfg, "", bridges)
    licenses = [t for t in meta.get("tags", []) if t.startswith("license")]
    license_name = (meta.get("cardData") or {}).get("license_name")
    report = (
        f"{repo_id}\n"
        f"  sha={sha} gated={meta.get('gated')} {licenses} license_name={license_name} size={size_gb:.1f}GB\n"
        f"  arch={cfg.get('architectures')} model_type={cfg.get('model_type')} bridges={bridges or None} "
        f"py={len(py_files)} pickle={pickles[:3]} config_err={cfg.get('_err')} "
        f"modified={(meta.get('lastModified') or '')[:10]}"
    )
    return report, "_err" not in cfg


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    all_ok = True
    for repo_id in argv:
        report, ok = describe(repo_id)
        print(report)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
