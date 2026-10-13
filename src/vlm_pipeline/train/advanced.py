"""Advanced engine options of the train nodes (P016 §3), checked against the committed engine file of the image.

``configs/engine/automodel-<tag>.json`` lists what the security denylist allows (scripts/probe/engine_schema_write.py):
per dotted key its type, enum choices, number range and the templates (catalog model ids) it applies to. The node
param ``advanced`` is checked here at validation; the train support check refuses a key the wired model's template
does not have (the model is only known there); build_recipe sets the values by dotted path.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic_core import PydanticCustomError

from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.paths import configs_root

ENGINE_DIR = Path("configs") / "engine"
MAX_KEYS = 100  # a hand-written file cannot smuggle thousands of overrides into one recipe


class AdvancedNotForModelError(VlmPipelineError):
    """An Advanced key that the wired model's training template does not have (support check, CFG-005)."""

    code: ClassVar[str] = "CFG-005"


@functools.cache  # read once per process like the templates digest: the file ships with the app
def engine_options(root: Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    path = configs_root(root) / ENGINE_DIR / f"automodel-{DEFAULT_TRAIN_IMAGE.rsplit(':', 1)[-1]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["key"]: entry for entry in data["keys"]}


def _number(entry: Mapping[str, Any], value: object) -> str | None:
    numeric = isinstance(value, int) if entry["type"] == "int" else isinstance(value, int | float)
    if isinstance(value, bool) or not numeric:
        return "type"
    low, high = entry.get("min"), entry.get("max")
    return None if low is not None and high is not None and low <= value <= high else "range"


def _numbers(value: object) -> bool:
    return isinstance(value, list) and len(value) <= 16 and all(isinstance(v, int | float) for v in value)


def _bool(entry: Mapping[str, Any], value: object) -> str | None:
    return None if isinstance(value, bool) else "type"


def _enum(entry: Mapping[str, Any], value: object) -> str | None:
    return None if value in (entry.get("choices") or []) else "choice"


def _list(entry: Mapping[str, Any], value: object) -> str | None:
    return None if _numbers(value) else "type"


Check = Callable[[Mapping[str, Any], object], "str | None"]
_CHECKS: Mapping[str, Check] = {"bool": _bool, "enum": _enum, "int": _number, "float": _number, "list": _list}


def _problem(entry: Mapping[str, Any], value: object) -> str | None:
    if isinstance(value, str) and ("${" in value or "oc." in value):
        return "resolver"  # ${oc.env:HF_TOKEN} would leak a secret into the recipe (security)
    check = _CHECKS.get(entry["type"])
    return "type" if check is None else check(entry, value)


ERROR_TYPE = "advanced_options"  # pydantic error type; ctx["problems"] = [{"key", "rule"}] (graph.validation)
_SAFE_KEY = re.compile(
    r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*){0,7}$"
)  # an unknown key is echoed only if it looks like one


def check_advanced(values: Mapping[str, object]) -> dict[str, object]:
    """``values`` when every key is offered and every value fits, else ONE error listing every bad key.

    Raises ``PydanticCustomError(ERROR_TYPE)`` with ``ctx["problems"] = [{"key": dotted key, "rule": rule}]``, rule in
    not_offered | type | range | choice | resolver | too_many, so the UI shows each problem under its own row. A key
    that is not offered comes from the file: it is echoed only when it has the shape of a key (else ``""``).
    """
    problems: list[dict[str, str]] = []
    if len(values) > MAX_KEYS:
        problems.append({"key": "", "rule": "too_many"})
    else:
        options = engine_options()
        for key, value in values.items():
            entry = options.get(key)
            if entry is None:
                shown = key if len(key) <= 120 and _SAFE_KEY.fullmatch(key) else ""
                problems.append({"key": shown, "rule": "not_offered"})
            elif (rule := _problem(entry, value)) is not None:
                problems.append({"key": key, "rule": rule})
    if problems:
        raise PydanticCustomError(
            ERROR_TYPE, "{count} advanced option(s) invalid", {"count": len(problems), "problems": problems}
        )
    return dict(values)


def keys_not_for(model_id: str, values: Mapping[str, object]) -> list[str]:
    """Keys of ``values`` that the training template of ``model_id`` does not have (optimizers differ per model)."""
    options = engine_options()
    return sorted(k for k in values if k in options and model_id not in options[k].get("templates", ()))


def set_dotted(recipe: dict[str, Any], key: str, value: object) -> None:
    """``recipe["a"]["b"] = value`` for ``"a.b"``.

    A missing section is created (a constructor kwarg the template does not spell out); an existing NON-section parent
    is an error, never overwritten (it would silently drop e.g. a collate_fn).
    """
    *parents, leaf = key.split(".")
    node = recipe
    for depth, part in enumerate(parents):
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"{key}: {'.'.join(parents[: depth + 1])} is not a section")
        node = child
    node[leaf] = value


__all__ = ["ERROR_TYPE", "AdvancedNotForModelError", "check_advanced", "engine_options", "keys_not_for", "set_dotted"]
