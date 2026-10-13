"""Every container job the app can start mounts only what it needs (P017 step 6, security S6, release 1.0).

A job built with ``narrow=False`` gets the whole project mounted (read-only): .git, .team, other runs' outputs. The
2026-10-13 merge_lora job (n5) did, because the app process predated step 6. Guard: every ``ContainerJob(...)`` call
in the code the app runs passes ``narrow=True`` literally; the only exceptions are team CLI entrypoints, listed with
their reason. A new worker that forgets it fails here, not in a mount audit. Behaviour (what is mounted) is tested per
worker and in tests/unit/common/test_jobrun.py.
In compose the runner (runner/server.py) builds every job with narrow=True at the trust boundary; this guard is the
layer for dev mode (LocalDockerLauncher). A behavioural check through template_e2e comes after 1.0.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "vlm_pipeline"
# Team CLI entrypoints (never reachable from the app): they keep the project mount until they move to the runner.
ALLOWED_WIDE = {
    "common/jobrun_cli.py": "team CLI (jobrun_cli), eval/bench scripts run by members",
    "train/smoke.py": "team CLI (train.smoke) for recipe smoke runs",
}


def _container_job_calls() -> list[tuple[str, int, ast.Call]]:
    calls = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "common/jobrun.py":  # the definition (and its own helpers)
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                if name == "ContainerJob":
                    calls.append((rel, node.lineno, node))
    return calls


def _modules() -> list[tuple[str, ast.Module]]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel != "common/jobrun.py":
            out.append((rel, ast.parse(path.read_text(encoding="utf-8"))))
    return out


def _unpacked(call: ast.Call) -> bool:
    return any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords)


def _narrow_true(call: ast.Call) -> bool:
    return any(
        kw.arg == "narrow" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords
    )


def test_every_app_container_job_is_narrow() -> None:
    calls = [(rel, line, call) for rel, line, call in _container_job_calls() if rel not in ALLOWED_WIDE]
    unchecked = [f"{rel}:{line}" for rel, line, call in calls if _unpacked(call)]
    assert unchecked == [], f"ContainerJob(*args / **kwargs) cannot be checked for narrow=True: {unchecked}"
    wide = [f"{rel}:{line}" for rel, line, call in calls if not _narrow_true(call)]
    assert wide == [], f"ContainerJob without narrow=True (whole project mounted): {wide}"


def test_the_scan_sees_the_known_workers() -> None:
    """The guard must not pass because it found nothing (a rename of ContainerJob, a moved worker)."""
    files = {rel for rel, _, _ in _container_job_calls()}
    for expected in (
        "app/workers.py",
        "app/trial_worker.py",
        "app/bench_worker.py",
        "app/quantize_worker.py",
        "eval/node_worker.py",
        "serve/vllm_runner.py",
        "runner/server.py",
    ):
        assert expected in files, f"{expected} no longer builds a ContainerJob: update this test"
    assert set(ALLOWED_WIDE) <= files, "an allowlisted file no longer builds a ContainerJob: drop it from the list"


def test_narrow_is_never_turned_off_after_the_job_is_built() -> None:
    """dataclasses.replace(job, narrow=False), setattr(job, "narrow", ...) or a variable passed as narrow= would bypass
    the literal check above: every narrow= keyword in the code must be the constant True, and no code sets the
    attribute by name."""
    problems = []
    for rel, tree in _modules():
        if rel in ALLOWED_WIDE:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "narrow" and not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
                    problems.append(f"{rel}:{node.lineno} narrow={ast.unparse(kw.value)}")
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in ("setattr", "__setattr__") and any(
                isinstance(a, ast.Constant) and a.value == "narrow" for a in node.args
            ):
                problems.append(f"{rel}:{node.lineno} {name}(..., 'narrow', ...)")
    assert problems == [], f"narrow set other than by the literal narrow=True: {problems}"
