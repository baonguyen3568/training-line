"""Can an exported model folder be loaded on another machine without running code we did not ship (ADR-012 B3).

Decided on the folder's REAL content, never on the catalog alone: any ``auto_map`` at any level of any ``config*.json``
or any ``*.py`` next to the weights means the model runs remote code, which a native export refuses (fail closed) until
the Parse patch PR makes that code portable. Also: no absolute path may survive in a config (it leaks the machine's
layout and breaks elsewhere).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_PATH_MARKERS = ("/workspace", "/home/", "/root/", "/tmp/")  # noqa: S108 - markers searched for, never used
_REMOTE_KEYS = frozenset({"auto_map", "custom_pipelines"})
TEXT_SUFFIXES = frozenset({".json", ".txt", ".jinja", ".md"})


def _walk(node: Any, where: str) -> Iterator[tuple[str, Any]]:
    yield where, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{where}.{key}" if where else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{where}[{i}]")


def remote_code_findings(folder: Path) -> list[str]:
    """Why loading ``folder`` would execute model code from it (empty = native weights + configs only)."""
    findings = [f"{p.name}: Python file next to the weights" for p in sorted(folder.glob("*.py"))]
    for cfg in sorted(folder.glob("*.json")):  # every JSON, not only *config*: auto_map can sit in any of them
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            findings.append(f"{cfg.name}: unreadable ({type(exc).__name__}): cannot rule out remote code")
            continue
        findings += [
            f"{cfg.name}: {where or '<root>'} has {key}"
            for where, node in _walk(data, "")
            if isinstance(node, dict)
            for key in sorted(_REMOTE_KEYS & node.keys())
        ]
    return findings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_as_upstream(path: Path, upstream: Path | None) -> bool:
    """``path`` is byte-identical to the file of the same name in the base model's own snapshot."""
    if upstream is None:
        return False
    original = upstream / path.name
    return original.is_file() and original.stat().st_size == path.stat().st_size and _sha256(original) == _sha256(path)


def absolute_path_findings(folder: Path, *, upstream: Path | None = None) -> list[str]:
    """Machine paths a shared folder must not carry (they leak the user name and break elsewhere).

    Every text file (json/txt/jinja/md) is scanned for path markers; JSON ``_name_or_path`` must not be absolute.
    A file byte-identical to the one in ``upstream`` (the base model's snapshot) is not marker-scanned: it holds
    nothing of this machine (Qwen's vocab has a "/workspace" token).
    """
    findings: list[str] = []
    for text_file in sorted(p for p in folder.iterdir() if p.suffix in TEXT_SUFFIXES and p.is_file()):
        if _same_as_upstream(text_file, upstream):
            continue
        text = text_file.read_bytes().decode("utf-8", errors="replace")
        findings += [f"{text_file.name}: contains a machine path ({m})" for m in _PATH_MARKERS if m in text]
    for cfg in sorted(folder.glob("*.json")):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # reported by remote_code_findings
        findings += [
            f"{cfg.name}: {where} is a machine path"
            for where, node in _walk(data, "")
            if isinstance(node, str) and where.endswith("_name_or_path") and node[:1] in {"/", "~"}
        ]
    return findings


def _set_name_or_path(node: Any, value: str, where: str = "") -> list[str]:
    """Set every ``_name_or_path`` under ``node`` to ``value`` in place; returns the keys that changed."""
    hits: list[str] = []
    if isinstance(node, dict):
        for key, sub in node.items():
            here = f"{where}.{key}" if where else key
            if key == "_name_or_path" and isinstance(sub, str):
                if sub != value:
                    node[key] = value
                    hits.append(here)
            else:
                hits += _set_name_or_path(sub, value, here)
    elif isinstance(node, list):
        for i, sub in enumerate(node):
            hits += _set_name_or_path(sub, value, f"{where}[{i}]")
    return hits


def neutralize_name_or_path(folder: Path, value: str) -> list[str]:
    """Set every ``_name_or_path`` in the folder's configs to ``value`` (e.g. ``org/model@sha``); returns changed keys.

    Training writes the container path of the base model there; an export names its origin, never a location.
    """
    changed: list[str] = []
    for cfg in sorted(folder.glob("*config*.json")):
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if hits := _set_name_or_path(data, value):
            tmp = cfg.with_name(cfg.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(cfg)
            changed += [f"{cfg.name}: {h}" for h in hits]
    return changed


_CACHE_FILES = ("config.json", "generation_config.json")
# Top level + a VLM's language-model config. A model that keeps use_cache elsewhere (e.g. Parse's "decoder",
# "language_config") needs its scope added here.
_CACHE_SCOPES = ("", "text_config")


def _scope(data: Any, name: str) -> dict[str, Any] | None:
    node = data if not name else (data.get(name) if isinstance(data, dict) else None)
    return node if isinstance(node, dict) else None


def follow_base_use_cache(folder: Path, upstream: Path | None) -> list[str]:
    """``use_cache`` in the export's configs as the base model has it (set, or removed); returns changed keys.

    Training may save ``use_cache: false`` (e.g. the Parse decoder patch, needed to train and resume): an export is
    for inference, where false makes HF generate ~3x slower without a word (fixcode / tester 2026-10-13).
    """
    changed: list[str] = []
    for name in _CACHE_FILES:
        cfg = folder / name
        if not cfg.is_file():
            continue
        base_file = upstream / name if upstream is not None else None
        try:
            base = json.loads(base_file.read_text(encoding="utf-8")) if base_file and base_file.is_file() else {}
        except (OSError, ValueError):
            base = {}
        data = json.loads(cfg.read_text(encoding="utf-8"))
        hits: list[str] = []
        for scope in _CACHE_SCOPES:
            node, base_node = _scope(data, scope), _scope(base, scope)
            if node is None or "use_cache" not in node:
                continue
            where = f"{scope}.use_cache" if scope else "use_cache"
            if base_node is not None and "use_cache" in base_node:
                if node["use_cache"] != base_node["use_cache"]:
                    node["use_cache"] = base_node["use_cache"]
                    hits.append(where)
            else:
                del node["use_cache"]  # the library default (true for generation)
                hits.append(where)
        if hits:
            tmp = cfg.with_name(cfg.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(cfg)
            changed += [f"{name}: {h}" for h in hits]
    return changed


__all__ = ["absolute_path_findings", "follow_base_use_cache", "neutralize_name_or_path", "remote_code_findings"]
