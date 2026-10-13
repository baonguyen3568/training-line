"""No secret or bulky folder ever enters a docker build context (ADR-015 M3, bao-security).

Every Dockerfile under docker/ (and a root Dockerfile) is built with the repository root as context. BuildKit uses
``<Dockerfile>.dockerignore`` next to it when present, else the root ``.dockerignore``. For each Dockerfile:
1. that ignore file exists and starts with ``*`` (allowlist: only ``!`` lines enter the context);
2. its last lines deny ``.env*`` and ``**/.env*`` again (last match wins: no ``!`` above can bring them back);
3. none of FORBIDDEN (env files, git, outputs, caches, data, team state, local settings, keys) is in the context,
   evaluated with the docker pattern rules (``**``, ``*``, ``?``; a matched directory covers its contents).

  uv run python scripts/qa/check_build_context.py        # exit 1 and list the problems when one fails
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED_TAIL = (".env*", "**/.env*")
FORBIDDEN = (
    ".env",
    ".env.local",
    "docker/.env.prod",
    "src/vlm_pipeline/.env",
    ".git/config",
    ".git/objects/pack/p.pack",
    ".venv/bin/python",
    "outputs/system/cleanup.jsonl",
    ".claude/agents/x.md",
    "frontend/.npmrc",
    "site/.npmrc",
    ".pypirc",
    "home/.netrc",
    "docker/credentials.json",
    "third_party/patches/server.key",
    "outputs/runs/r1/run.json",
    ".cache/hf/hub/x.safetensors",
    "data/uploads/u1/a.png",
    ".team/GPU_LOCK",
    ".claude/settings.local.json",
    "frontend/node_modules/x/index.js",
    "secrets/hf_token",
    "id_ed25519",
    "keys/server.pem",
)


def patterns(text: str) -> list[str]:
    return [s for line in text.splitlines() if (s := line.strip()) and not s.startswith("#")]


def _regex(pattern: str) -> re.Pattern[str]:
    if pattern in ("", "."):  # the context root itself: everything
        return re.compile(".*")
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):  # any number of folders, including none
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        else:
            c = pattern[i]
            out, i = out + ("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c)), i + 1
    return re.compile(f"{out}(?:/.*)?")  # a matched directory excludes/includes everything under it


def in_context(path: str, rules: list[str]) -> bool:
    """Docker semantics: the last matching rule decides; ``!rule`` includes, ``rule`` excludes; no match -> included."""
    included = True
    for rule in rules:
        negate = rule.startswith("!")
        body = (rule[1:] if negate else rule).strip().removeprefix("./").strip("/")  # like filepath.Clean
        if _regex(body).fullmatch(path):
            included = negate
    return included


def ignore_file(dockerfile: Path, repo: Path) -> Path:
    own = dockerfile.with_name(dockerfile.name + ".dockerignore")
    return own if own.exists() else repo / ".dockerignore"


def check(dockerfile: Path, repo: Path) -> list[str]:
    ignore = ignore_file(dockerfile, repo)
    name = dockerfile.relative_to(repo).as_posix()
    if not ignore.is_file():
        return [f"{name}: no {ignore.relative_to(repo)} (the whole repository would be sent)"]
    rules = patterns(ignore.read_text(encoding="utf-8"))
    problems = []
    if not rules or rules[0] != "*":
        problems.append(f"{name}: {ignore.name} must start with '*' (allowlist)")
    if tuple(rules[-len(REQUIRED_TAIL) :]) != REQUIRED_TAIL:
        problems.append(f"{name}: {ignore.name} must end with {', '.join(REQUIRED_TAIL)}")
    problems += [f"{name}: '{p}' would be in the build context" for p in FORBIDDEN if in_context(p, rules)]
    return problems


def dockerfiles(repo: Path) -> list[Path]:
    found = [p for p in (repo / "docker").rglob("*Dockerfile*") if p.is_file() and not p.name.endswith(".dockerignore")]
    return sorted(found + [p for p in (repo / "Dockerfile",) if p.is_file()])


def main(repo: Path = REPO) -> int:
    problems = [problem for df in dockerfiles(repo) for problem in check(df, repo)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
