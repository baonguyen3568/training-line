"""P016 §3: dump every option of the pinned NeMo AutoModel engine that our train templates can reach (one JSON).

Runs INSIDE the engine image, offline: ``docker run --rm --network none -v <repo>/configs/train/templates:/t:ro
-v <repo>/scripts/probe:/p:ro -v <out dir>:/out vlm-automodel:<tag> python /p/engine_schema.py /t /out/engine.json``
(a file, not stdout: the NGC entrypoint prints a banner). Imports engine code, never loads
weights. The host filters the result through ``vlm_pipeline.nodes.advanced_denylist`` before committing it to
``configs/engine/automodel-<tag>.json`` (scripts/probe/engine_schema_write.py).

Sources, per key: ``signature`` (the ``_target_`` callable's parameters: dataclass configs, managers, schedulers),
``template`` (keys our templates set, for ``from_pretrained``-style ``**kwargs`` targets and plain sections) and
``recipe`` (``cfg.get("a.b", default)`` reads in the VLM finetune recipe). Output is sorted, without timestamps, so
an image bump shows exactly which knobs changed.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import json
import re
import sys
import typing
from pathlib import Path
from typing import Any

import yaml

SCALARS = (bool, int, float, str)


def _resolve(target: str) -> Any:
    module, _, attr = target.rpartition(".")
    try:
        obj: Any = importlib.import_module(module)
        return getattr(obj, attr)
    except (ImportError, AttributeError):
        parent, _, cls = module.rpartition(".")  # Class.method targets: "pkg.mod.Class.from_pretrained"
        try:
            return getattr(getattr(importlib.import_module(parent), cls), attr)
        except (ImportError, AttributeError):
            return None


def _kind(annotation: Any, default: Any) -> tuple[str, list[Any] | None]:
    """(type name, enum choices) of a parameter; ``other`` for anything not offered as an Advanced value."""
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if origin is typing.Literal:
        return "enum", list(typing.get_args(annotation))
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return "enum", [m.value for m in annotation]
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)) and len(args) == 1:
        return _kind(args[0], default)
    for py, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if annotation is py or (annotation is inspect.Parameter.empty and type(default) is py):
            return name, None
    if origin in (list, tuple) and args and all(a in (bool, int, float) for a in args):
        return "list", None  # never list[str]: free strings could carry a path / module / URL (security D1)
    return "other", None


def _json_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        value = value.value
    if value is None or isinstance(value, SCALARS):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, SCALARS) for v in value):
        return list(value)
    return None  # not representable: no default offered (the UI shows "engine default")


def _signature_keys(prefix: str, target: str) -> dict[str, dict[str, Any]]:
    obj = _resolve(target)
    if obj is None:
        return {}
    if dataclasses.is_dataclass(obj):
        hints = typing.get_type_hints(obj) if inspect.isclass(obj) else {}
        fields = {f.name: (hints.get(f.name, f.type), f.default) for f in dataclasses.fields(obj)}
    else:
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return {}
        fields = {
            name: (p.annotation, p.default)
            for name, p in sig.parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and name not in ("self", "cls")
        }
    keys: dict[str, dict[str, Any]] = {}
    for name, (annotation, raw_default) in fields.items():
        missing = raw_default is dataclasses.MISSING or raw_default is inspect.Parameter.empty
        default = None if missing else raw_default
        kind, choices = _kind(annotation, default)
        keys[f"{prefix}{name}"] = {"type": kind, "default": _json_default(default), "choices": choices,
                                   "source": "signature", "target": target}  # fmt: skip
    return keys


def _walk(node: Any, prefix: str, keys: dict[str, dict[str, Any]]) -> None:
    if not isinstance(node, dict):
        return
    target = node.get("_target_")
    if isinstance(target, str):
        for key, entry in _signature_keys(prefix, target).items():
            keys.setdefault(key, entry)
    for name, value in node.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            _walk(value, f"{key}.", keys)
        elif key not in keys:
            kind, _ = _kind(inspect.Parameter.empty, value)
            keys[key] = {"type": kind, "default": None, "choices": None, "source": "template", "target": target}


def _recipe_keys() -> dict[str, dict[str, Any]]:
    from nemo_automodel.recipes.vlm import finetune

    source = inspect.getsource(finetune)
    found: dict[str, dict[str, Any]] = {}
    for key in sorted(set(re.findall(r"""cfg\.get\(\s*["']([A-Za-z_][\w.]*)["']""", source))):
        found[key] = {"type": "other", "default": None, "choices": None, "source": "recipe", "target": None}
    return found


def main(templates_dir: str, out_path: str) -> int:
    keys: dict[str, dict[str, Any]] = {}
    templates: dict[str, set[str]] = {}  # key -> template ids it applies to (optimizers differ per model)
    for path in sorted(Path(templates_dir).glob("*.yaml")):
        found: dict[str, dict[str, Any]] = {}
        _walk(yaml.safe_load(path.read_text(encoding="utf-8")), "", found)
        for key, entry in found.items():
            keys.setdefault(key, entry)
            templates.setdefault(key, set()).add(path.stem)
    for key, entry in _recipe_keys().items():
        keys.setdefault(key, entry)
    import nemo_automodel

    out = {
        "engine": "nemo_automodel",
        "engine_version": getattr(nemo_automodel, "__version__", "unknown"),
        "keys": [{"key": k, **keys[k], "templates": sorted(templates.get(k, []))} for k in sorted(keys)],
    }
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
