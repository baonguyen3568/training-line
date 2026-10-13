"""No made-up numbers on the site: every number in a page's prose must appear in one of the files its ``sources`` list.

check-sources (bao-cleancode) proves the listed paths exist; this proves the numbers came from them. For each MDX page
under site/src/content, numbers are taken from the prose (frontmatter, imports, code and link targets excluded),
normalized (Vietnamese decimal comma ``0,94`` -> ``0.94``, thousands separators dropped) and searched in the sources'
text with the same normalization. A number found in none of them is reported with its line.

  uv run python scripts/qa/check_site_numbers.py [--root site/src/content] [--strict]

Limits: a match means the digits occur somewhere in a source file, not that they mean the same thing there (0.94
may match an unrelated field of a long YAML). Passing is a first filter, not a semantic check: key numbers (landing,
benchmark results) still get a human review, and pages should list the specific benchmark file, not a folder.
Scope: Markdown/MDX under --root only; .astro pages and site/src/data are not scanned yet (P011 G1).

Exit 1 with --strict when a number is unsourced (report-only otherwise, while pages are being written).
Numbers that need no source (step "1." of a list, years in a date, ...) are skipped by IGNORE; a page can whitelist
more with frontmatter ``numbers_without_source: ["3", ...]`` and should say why in the prose.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
CODE_FENCE = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
TAG_ATTR = re.compile(r"<[A-Za-z][^>]*>")  # JSX/HTML tags: attribute values are not prose
IMPORT = re.compile(r"^(import|export) .*$", re.M)
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")  # prose: standalone numbers only
SOURCE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")  # sources: also "128GB", "CI95", "42.0"
CODE_ID = re.compile(r"\b[A-Z]{2,}-\d+\b")  # error codes (CFG-010) are identifiers, not claims
IGNORE = {str(i) for i in range(11)}  # list positions, "1 model", "2 bước": too common to source, too small to mislead


def normalize(token: str) -> str:
    """``0,94`` -> ``0.94``; ``1.024`` / ``1,024`` with 3-digit groups -> ``1024``; ``20`` stays."""
    parts = re.split(r"[.,]", token)
    if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and not token.startswith("0"):
        return "".join(parts)
    value = token.replace(",", ".")
    return value[:-2] if value.endswith(".0") else value  # "42.0" in a JSON record is the "42" on the page


def numbers_in(text: str) -> set[str]:
    return {normalize(m.group()) for m in SOURCE_NUMBER.finditer(text)}


def prose(body: str) -> str:
    for pattern in (CODE_FENCE, IMPORT, INLINE_CODE, LINK_TARGET, TAG_ATTR, CODE_ID):
        body = pattern.sub(lambda m: "\n" * m.group().count("\n"), body)
    return body


def check_page(page: Path, repo: Path = REPO) -> dict[str, object]:
    text = page.read_text(encoding="utf-8")
    match = FRONTMATTER.match(text)
    meta = yaml.safe_load(match.group(1)) if match else {}
    meta = meta or {}
    body = text[match.end() :] if match else text
    offset = text[: match.end()].count("\n") if match else 0
    sources = [str(s) for s in meta.get("sources") or []]
    known: set[str] = set()
    missing_sources = []
    for source in sources:
        path = repo / source
        if path.is_file():
            known |= numbers_in(path.read_text(encoding="utf-8", errors="replace"))
        else:
            missing_sources.append(source)
    allowed = IGNORE | {normalize(str(n)) for n in meta.get("numbers_without_source") or []}
    unsourced = []
    for lineno, line in enumerate(prose(body).splitlines(), start=offset + 1):
        for m in NUMBER.finditer(line):
            value = normalize(m.group())
            if value not in allowed and value not in known:
                unsourced.append({"line": lineno, "number": m.group(), "context": line.strip()[:100]})
    return {
        "page": str(page.relative_to(repo)),
        "sources": sources,
        "missing_sources": missing_sources,
        "unsourced": unsourced,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="site/src/content")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    pages = sorted((REPO / args.root).rglob("*.mdx")) + sorted((REPO / args.root).rglob("*.md"))
    results = [check_page(p) for p in pages]
    bad = [r for r in results if r["unsourced"] or r["missing_sources"]]
    print(
        json.dumps(
            {"pages": len(results), "pages_with_problems": len(bad), "results": bad}, indent=1, ensure_ascii=False
        )
    )
    return 1 if args.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
