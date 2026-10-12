"""Print vetting-relevant Hugging Face metadata for model repos.

Reports SHA, license tag, safetensors size, .py / pickle files, and every ``auto_map`` and
``_name_or_path`` found recursively in config.json (any sub-config, not only the root).

Usage: python3 -I scripts/research/hf_model_meta.py org/repo [org/repo ...]

A sub-config entry such as ``other-org/other-repo--module.Class`` or a foreign ``_name_or_path``
means code is pulled from ANOTHER repo (at ``main`` unless snapshotted): flag it to bao-security.
Also grep the repo's .py for ``get_class_from_dynamic_module`` / ``trust_remote_code``.
Gated repos return no config without an accepted token.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import Any

HF = "https://huggingface.co"
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".tar")


def fetch_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
            data: dict[str, Any] = json.load(resp)
            return data
    except Exception as exc:  # noqa: BLE001 - report any fetch failure inline
        return {"_err": str(exc)}


def collect_bridges(node: Any, path: str, out: dict[str, Any]) -> None:
    """Record auto_map / _name_or_path found below the root of a config tree."""
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if path and key == "auto_map" and isinstance(value, dict):
            repos = sorted({str(v).split("--")[0] for v in value.values() if "--" in str(v)})
            out[f"{path}.auto_map"] = repos or "local"
        if path and key == "_name_or_path" and value:
            out[f"{path}._name_or_path"] = value
        collect_bridges(value, f"{path}.{key}" if path else key, out)


def describe(repo_id: str) -> str:
    meta = fetch_json(f"{HF}/api/models/{repo_id}?blobs=true")
    if "_err" in meta:
        return f"{repo_id} ERR {meta['_err']}"
    files = meta.get("siblings", [])
    size_gb = sum(f.get("size") or 0 for f in files if f["rfilename"].endswith(".safetensors")) / 1e9
    py_files = [f["rfilename"] for f in files if f["rfilename"].endswith(".py")]
    pickles = [f["rfilename"] for f in files if f["rfilename"].endswith(PICKLE_SUFFIXES)]
    cfg = fetch_json(f"{HF}/{repo_id}/resolve/{meta['sha']}/config.json")
    bridges: dict[str, Any] = {}
    collect_bridges(cfg, "", bridges)
    licenses = [t for t in meta.get("tags", []) if t.startswith("license")]
    license_name = (meta.get("cardData") or {}).get("license_name")
    return (
        f"{repo_id}\n"
        f"  sha={meta['sha']} gated={meta.get('gated')} {licenses} license_name={license_name} "
        f"size={size_gb:.1f}GB\n"
        f"  arch={cfg.get('architectures')} model_type={cfg.get('model_type')} "
        f"auto_map={'yes' if cfg.get('auto_map') else 'no'} subconfig_bridges={bridges or None} "
        f"py={len(py_files)} pickle={pickles[:3]} config_err={cfg.get('_err')} "
        f"modified={(meta.get('lastModified') or '')[:10]}"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for repo_id in argv:
        print(describe(repo_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
