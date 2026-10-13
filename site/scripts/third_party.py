"""Generate the site's third-party notices from configs/security/licenses.yaml (owner: bao-security).

Only entries with scope `site` ship code in site/dist (security review 2026-10-13, incl. vendored mark.js inside
pagefind). Outputs (generated, git-ignored):
  public/THIRD_PARTY_LICENSES.txt   full notices + license texts, served as-is
  src/generated/third-party.json    the list rendered on /chinh-sach/ben-thu-ba/
Fails when a `site` entry has no license text, so a new shipped dependency cannot slip through unattributed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parent.parent
ROOT = SITE.parent
MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def license_text(repo_id: str, version: str, license_id: str, notice: str) -> str:
    kind, _, name = repo_id.partition(":")
    if kind == "npm":
        pkg = SITE / "node_modules" / name
        installed = json.loads((pkg / "package.json").read_text(encoding="utf-8")).get("version")
        if installed != version:  # node_modules out of sync with the lock: never attach another version's text
            raise SystemExit(f"third_party: {name} installed {installed}, licenses.yaml says {version} (run npm ci)")
        for path in sorted(pkg.glob("*")):
            if not path.name.lower().startswith(("license", "licence", "copying")):
                continue
            files = sorted(f for f in path.iterdir() if f.is_file()) if path.is_dir() else [path]  # pagefind: a dir
            if files:
                return "\n\n".join(f.read_text(encoding="utf-8", errors="replace").strip() for f in files)
        if license_id == "MIT":  # MIT package that ships no LICENSE file (e.g. @pagefind/default-ui)
            author = json.loads((pkg / "package.json").read_text(encoding="utf-8")).get("author")
            author = author.get("name") if isinstance(author, dict) else author
            return f"Copyright (c) {author or name} (package.json author; no LICENSE file in the package)\n\n{MIT}"
    if kind == "vendored" and license_id == "MIT":  # bundled inside another package, no LICENSE file of its own
        return f"{notice}\n\n{MIT}".strip()
    raise SystemExit(f"third_party: no license text for {repo_id} ({license_id})")


def main() -> None:
    assets = yaml.safe_load((ROOT / "configs/security/licenses.yaml").read_text(encoding="utf-8"))["assets"]
    shipped = sorted((a for a in assets if "site" in (a.get("scope") or [])), key=lambda a: a["repo_id"].lower())
    items, blocks = [], []
    for a in shipped:
        name = a["repo_id"].partition(":")[2]
        notice = a.get("notice") or ""
        items.append({"name": name, "version": str(a["version"]), "license": a["license"], "notice": notice})
        blocks.append(f"{name} {a['version']} ({a['license']})\n{'-' * 72}\n{license_text(a['repo_id'], str(a['version']), a['license'], notice)}\n")
    head = "Third-party software shipped with this site. Generated from configs/security/licenses.yaml.\n\n"
    (SITE / "public").mkdir(exist_ok=True)
    (SITE / "public/THIRD_PARTY_LICENSES.txt").write_text(head + "\n\n".join(blocks), encoding="utf-8")
    out = SITE / "src/generated/third-party.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"third_party: {len(items)} shipped packages", file=sys.stderr)


if __name__ == "__main__":
    main()
