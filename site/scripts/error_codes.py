"""Error-code reference for /docs/ma-loi, generated from the app's own catalog + text (spec: bao-design, M2).

Sources: src/vlm_pipeline/common/error_catalog.yaml (codes, severity, actions; owner bao-fixcode) and
src/vlm_pipeline/i18n/{vi,en}.yaml (title / what / action labels / error_group.<PREFIX>; owner bao-design).
Output (generated, git-ignored): src/generated/error-codes.json, grouped by prefix, one entry per code with
``anchor`` = the code (the app's error card links to /docs/ma-loi#<CODE>).

Public on purpose: only the user-facing text. Never ``match`` (log regexes), ``slug`` or params. Placeholders such as
``{node}`` stay as they are; the page renders them in italics. A prefix without an ``error_group.<PREFIX>`` text fails
the build so new groups get their copy before they ship. Deprecated codes are left out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SITE = Path(__file__).resolve().parent.parent
ROOT = SITE.parent
LOCALES = ("vi", "en")
# Codes with a longer walkthrough on the troubleshooting page (bao-cleanup): link to its <a id="CODE"> anchor. Only
# codes that really have that anchor in the page are linked (read from the source, never assumed).
GUIDE_PAGE = "/docs/xu-ly-su-co-bo-nho/"
GUIDE_SOURCE = SITE / "src/content/docs/docs/xu-ly-su-co-bo-nho.mdx"


def _guide_anchors() -> set[str]:
    import re

    if not GUIDE_SOURCE.is_file():
        return set()
    return set(re.findall(r'<a id="([A-Z]+-\d{3})"', GUIDE_SOURCE.read_text(encoding="utf-8")))


def _what(code: str, texts: dict[str, str]) -> list[str]:
    """The 'what' text: one sentence, or one per variant (key order of the i18n file)."""
    bare = texts.get(f"error.{code}.what")
    if bare:
        return [bare]
    prefix = f"error.{code}.what."
    return [text for key, text in texts.items() if key.startswith(prefix)]


def build() -> dict[str, Any]:
    from vlm_pipeline.common.error_catalog import load_catalog
    from vlm_pipeline.i18n import load

    texts = {loc: load(loc) for loc in LOCALES}
    guides = _guide_anchors()
    groups: dict[str, dict[str, Any]] = {}
    for code, entry in sorted(load_catalog().items()):
        if entry.deprecated:
            continue
        prefix = code.split("-", 1)[0]
        if prefix not in groups:
            missing = [loc for loc in LOCALES if not texts[loc].get(f"error_group.{prefix}")]
            if missing:
                raise SystemExit(f"error_codes: no error_group.{prefix} text in {missing} (ask bao-design)")
            groups[prefix] = {
                "prefix": prefix,
                "title": {loc: texts[loc][f"error_group.{prefix}"] for loc in LOCALES},
                "codes": [],
            }
        groups[prefix]["codes"].append(
            {
                "code": code,
                "anchor": code,
                "severity": entry.severity,
                "title": {loc: texts[loc][f"error.{code}.title"] for loc in LOCALES},
                "what": {loc: _what(code, texts[loc]) for loc in LOCALES},
                "actions": [{loc: texts[loc][f"action.{a.id}"] for loc in LOCALES} for a in entry.actions],
                "guide": f"{GUIDE_PAGE}#{code}" if code in guides else None,
            }
        )
    return {"groups": list(groups.values())}


def main() -> None:
    data = build()
    dest = SITE / "src/generated/error-codes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    count = sum(len(g["codes"]) for g in data["groups"])
    print(f"error_codes: {count} codes in {len(data['groups'])} groups", file=sys.stderr)


if __name__ == "__main__":
    main()
