"""Detect silently-ignored model config fields (renamed upstream, read via ``getattr(cfg, name, default)``).

Case that motivated it (fixcode KB §2.6b): Nemotron-Parse v1.2/2.0 renamed ``encoder.image_processor_normalizes``
to ``processor_normalizes``; AutoModel's native class still reads the old name with default ``False`` and
normalizes images twice. Nothing crashes; accuracy drops. Pure, CPU-only, works on ``config.json`` dicts.

LIMITATION: this checks field NAMES only, not behaviour. It reports Parse v1.1 as clean, yet v1.1 is also
normalised twice on AutoModel 26.08-1 because ``NemotronParseConfig`` copies only keys that ``RADIOConfig``
already has (benchmarks/sanity/parse-v1.1-normalize-26.08-1.json). Behaviour is gated by the white-image probe
(scripts/qa/probe_double_normalize.py). RULES describe the native class of AutoModel 26.08-1; update them per
AutoModel version once the patched image (26.08-2) reads both names.

Rules per field (checked on the root config and on every nested sub-config dict):
- ``conflict`` (fail): two aliases are both present with different values.
- ``ignored`` (fail): only a non-native alias is present and its value differs from the native default, so the
  model would silently run with the default.
- ``ignored_same`` (warn): only a non-native alias is present but its value equals the native default.
- ``unsupported`` (fail): the effective value is one the native class cannot run (e.g. tied embeddings).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Level = Literal["fail", "warn"]


@dataclass(frozen=True)
class FieldRule:
    """One semantic field: every name it has had upstream, the name the native class reads and its default."""

    aliases: tuple[str, ...]
    native_name: str
    native_default: Any = None
    unsupported_values: tuple[Any, ...] = ()
    paths: tuple[str, ...] | None = None  # sub-config paths the native class reads it from ("" = root); None = all


@dataclass(frozen=True)
class Finding:
    level: Level
    kind: Literal["conflict", "ignored", "ignored_same", "unsupported"]
    path: str  # dotted sub-config path, "" for the root
    field: str  # native name
    detail: str


# Per model family: rules for the native AutoModel class (versions read: AutoModel 26.08 / main@829445d).
RULES: dict[str, tuple[FieldRule, ...]] = {
    "nemotron_parse": (
        FieldRule(
            ("image_processor_normalizes", "processor_normalizes"),
            "image_processor_normalizes",
            False,
            paths=("encoder",),
        ),
        # native class supports untied text embeddings only (TieSupport.UNTIED_ONLY);
        # the vision encoder's own flag is unused
        FieldRule(
            ("tie_word_embeddings",), "tie_word_embeddings", False, unsupported_values=(True,), paths=("", "decoder")
        ),
    ),
}


def _sub_configs(cfg: Mapping[str, Any], path: str = "") -> Iterator[tuple[str, Mapping[str, Any]]]:
    yield path, cfg
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            yield from _sub_configs(value, f"{path}.{key}" if path else key)


def check_config(cfg: Mapping[str, Any], rules: tuple[FieldRule, ...]) -> list[Finding]:
    findings: list[Finding] = []
    for path, sub in _sub_configs(cfg):
        for rule in rules:
            if rule.paths is not None and path not in rule.paths:
                continue
            present = {name: sub[name] for name in rule.aliases if name in sub}
            if not present:
                continue
            if len({repr(v) for v in present.values()}) > 1:
                findings.append(Finding("fail", "conflict", path, rule.native_name, f"aliases disagree: {present}"))
                continue
            value = next(iter(present.values()))
            if rule.native_name not in present:
                kind: Literal["ignored", "ignored_same"] = "ignored" if value != rule.native_default else "ignored_same"
                findings.append(
                    Finding(
                        "fail" if kind == "ignored" else "warn",
                        kind,
                        path,
                        rule.native_name,
                        f"config sets {sorted(present)}={value!r} but the native class reads "
                        f"{rule.native_name!r} (default {rule.native_default!r})",
                    )
                )
                value = rule.native_default  # what the model will actually use
            if value in rule.unsupported_values:
                findings.append(
                    Finding("fail", "unsupported", path, rule.native_name, f"{rule.native_name}={value!r} unsupported")
                )
    return findings


def check_family(cfg: Mapping[str, Any], family: str) -> list[Finding]:
    """Check a ``config.json`` dict against the rules of ``family`` (unknown family -> no rules, no findings)."""
    return check_config(cfg, RULES.get(family, ()))
