"""Generate the release list for /thay-doi from git tags (design sitemap §5: never hand-written).

A release is a tag ``v<semver>``; its notes are the tag's ANNOTATION (written once when releasing, reviewed like any
public text). Commit subjects are never published: they name internal roles and work in progress. Output (generated,
git-ignored): src/generated/changelog.json = ``{"releases": [{"version", "date", "notes"}...]}``, newest first.
No tag yet, or no git (a source tarball): an empty list, and the page says there is no release yet.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
ROOT = SITE.parent
OUT = SITE / "src" / "generated" / "changelog.json"
TAG = re.compile(r"^v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$")
FIELD_SEP, RECORD_SEP = "\x1f", "\x1e"


def releases(root: Path = ROOT) -> list[dict[str, str]]:
    fmt = FIELD_SEP.join(["%(refname:short)", "%(creatordate:short)", "%(contents)"]) + RECORD_SEP
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no user input
            ["git", "-C", str(root), "tag", "--list", "v*", "--sort=-creatordate", f"--format={fmt}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for record in filter(str.strip, out.split(RECORD_SEP)):
        name, date, contents = [*record.strip("\n").split(FIELD_SEP), "", ""][:3]
        if TAG.match(name):  # anything else (typos, internal tags) never reaches the public page
            notes = contents.split("-----BEGIN PGP SIGNATURE-----")[0].strip()
            found.append({"version": name.removeprefix("v"), "date": date, "notes": notes})
    return found


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"releases": releases()}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"changelog: {len(data['releases'])} release(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
