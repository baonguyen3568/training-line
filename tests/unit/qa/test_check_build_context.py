from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "check_build_context.py"
TAIL = ".env*\n**/.env*\n"


def _checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_build_context", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path, own: str | None = None, root: str | None = None) -> Path:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "x.Dockerfile").write_text("FROM scratch\n")
    if own is not None:
        (tmp_path / "docker" / "x.Dockerfile.dockerignore").write_text(own)
    if root is not None:
        (tmp_path / ".dockerignore").write_text(root)
    return tmp_path


def test_the_repository_passes() -> None:
    assert _checker().main() == 0


@pytest.mark.parametrize(
    ("path", "rules", "expected"),
    [
        (".env.local", ["*", "!src/"], False),
        ("src/a.py", ["*", "!src/"], True),
        ("src/.env", ["*", "!src/", *TAIL.split()], False),  # the tail wins over a wide include
        ("src/.env", ["*", "!src/"], True),
        ("a/b/.env.prod", ["**/.env*"], False),
        (".env", ["**/.env*"], False),  # "**/" also matches zero folders
        ("outputs/runs/r/run.json", ["outputs"], False),  # a matched folder covers its content
        ("docs/x.md", [], True),
        ("docker/automodel/extras.lock.txt", ["*", "!docker/automodel/extras.lock.txt"], True),
    ],
)
def test_docker_pattern_semantics(path: str, rules: list[str], expected: bool) -> None:
    assert _checker().in_context(path, rules) is expected


def test_allowlist_with_tail_is_clean(tmp_path: Path) -> None:
    repo = _repo(tmp_path, own="*\n!docker/automodel/\n" + TAIL)
    assert _checker().main(repo) == 0


@pytest.mark.parametrize(
    ("own", "root", "problem"),
    [
        (None, None, "no .dockerignore"),
        ("outputs\n" + TAIL, None, "must start with '*'"),
        ("*\n!docker/\n", None, "must end with"),
        ("*\n!**\n" + TAIL, None, "'.git/config' would be in the build context"),
        (None, "*\n!.\n" + TAIL, "'outputs/runs/r1/run.json' would be in the build context"),
    ],
)
def test_unsafe_contexts_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], own: str | None, root: str | None, problem: str
) -> None:
    assert _checker().main(_repo(tmp_path, own=own, root=root)) == 1
    assert problem in capsys.readouterr().err
