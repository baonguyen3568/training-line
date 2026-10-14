"""User-facing copy (vi/en): one source of truth in ``i18n/{vi,en}.yaml`` (owner: bao-design; Proposal 009 §2.2).

Keys are dotted strings. Node type ids contain dots themselves and are a single YAML key under ``node``
(``node["train.lora_vlm"]["title"]``), so the canonical key is ``node.<type_id>.<suffix>`` where suffix is one of
``title``, ``desc``, ``port.<name>``, ``param.<name>.label`` or ``param.<name>.help``; every other branch is a plain
dotted path. A missing key renders as the key itself (never crashes); a missing placeholder keeps ``{name}`` and logs
a warning so it is caught in tests rather than shown silently.
"""

from __future__ import annotations

import functools
import logging
import re
import string
from collections.abc import Mapping
from importlib import resources
from typing import Any, Literal

Locale = Literal["vi", "en"]
LOCALES: tuple[Locale, ...] = ("vi", "en")
DEFAULT_LOCALE: Locale = "vi"

_log = logging.getLogger(__name__)
_NODE_SUFFIX = re.compile(r"^node\.(?P<type_id>.+?)\.(?P<suffix>title|desc|port\.[^.]+|param\.[^.]+\.(?:label|help))$")


def _flatten(tree: Mapping[str, Any], prefix: str, out: dict[str, str]) -> None:
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten(value, path, out)
        elif isinstance(value, str):
            out[path] = value
        else:
            raise TypeError(f"i18n value at {path!r} must be a string, got {type(value).__name__}")


@functools.cache
def load(locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """Flat ``{canonical_key: text}`` for ``locale`` (cached; treat the result as read-only)."""
    if locale not in LOCALES:
        raise ValueError(f"unsupported locale {locale!r}; expected one of {LOCALES}")
    import yaml

    raw = resources.files(__package__).joinpath(f"{locale}.yaml").read_text(encoding="utf-8")
    tree = yaml.safe_load(raw) or {}
    if not isinstance(tree, Mapping):
        raise TypeError(f"{locale}.yaml must contain a mapping at the top level")
    out: dict[str, str] = {}
    _flatten(tree, "", out)
    return out


def split_key(key: str) -> tuple[str, ...]:
    """Path tuple for a canonical key; ``node.<type_id>.<suffix>`` keeps the dotted type id as one element."""
    m = _NODE_SUFFIX.match(key)
    if m:
        return ("node", m["type_id"], *m["suffix"].split("."))
    return tuple(key.split("."))


class _KeepMissing(dict[str, Any]):
    def __init__(self, params: Mapping[str, Any], key: str) -> None:
        super().__init__(params)
        self._key = key

    def __missing__(self, name: str) -> str:
        _log.warning("i18n: missing param %r for key %r", name, self._key)
        return "{" + name + "}"


def placeholders(text: str) -> set[str]:
    """Placeholder names used in ``text`` (``{name}``)."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def t(key: str, locale: str = DEFAULT_LOCALE, /, **params: Any) -> str:
    """Render ``key`` in ``locale``. Unknown key -> the key itself (warning); missing param -> ``{name}`` kept.

    ``key`` / ``locale`` are positional-only: an error's own params may be named ``key`` or ``locale`` (CFG-005's
    engine option ``{key}``) and must reach the text, never collide with them (E2E 2026-10-14: validate 500).
    """
    text = load(locale).get(key)
    if text is None:
        _log.warning("i18n: missing key %r (%s)", key, locale)
        return key
    if not params and not placeholders(text):
        return text
    return text.format_map(_KeepMissing(params, key))


__all__ = ["DEFAULT_LOCALE", "LOCALES", "Locale", "load", "placeholders", "split_key", "t"]
