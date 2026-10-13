"""Host side of a download (P018): nothing reaches the shared cache unless every staged file checks out."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vlm_pipeline.download.install import DownloadIntegrityError, install, pinned_hashes, snapshot_dir

REV = "b" * 40


def _stage(run_dir: Path, files: dict[str, bytes], *, plan_sha: dict[str, str | None] | None = None) -> None:
    stage = run_dir / "stage"
    (stage / "files").mkdir(parents=True)
    plan = []
    for name, data in files.items():
        path = stage / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lfs = name.endswith(".safetensors")
        sha = (plan_sha or {}).get(name, hashlib.sha256(data).hexdigest() if lfs else None)
        git = None if lfs else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324
        plan.append({"name": name, "size": len(data), "sha256": sha, "git_sha1": git})
    (stage / "plan.json").write_text(json.dumps({"files": plan}))


def _install(root: Path, run_dir: Path, **kw: object) -> Path:
    args = {"kind": "model", "repo": "org/m", "revision": REV, "catalog_id": "m", "name": "M", **kw}
    return install(root, run_dir, **args)  # type: ignore[arg-type]


def test_a_verified_stage_becomes_the_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "dl"
    _stage(run_dir, {"model.safetensors": b"w", "config.json": b"{}", "sub/tokenizer.json": b"{}"})
    target = _install(tmp_path, run_dir)
    assert target == snapshot_dir(tmp_path, "model", "org/m", REV)
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) == [
        "config.json", "model.safetensors", "sub/tokenizer.json"]  # fmt: skip
    assert not any(p.is_symlink() for p in target.rglob("*"))
    assert not list(target.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    "tamper",
    ["sha_mismatch", "symlink", "extra_file", "pickle", "missing", "git_mismatch"],
)
def test_any_doubt_installs_nothing(tmp_path: Path, tamper: str) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "dl"
    _stage(run_dir, {"model.safetensors": b"w", "config.json": b"{}"},
           plan_sha={"model.safetensors": "0" * 64} if tamper == "sha_mismatch" else None)  # fmt: skip
    files = run_dir / "stage" / "files"
    if tamper == "symlink":
        (files / "config.json").unlink()
        (files / "config.json").symlink_to("/etc/hostname")
    elif tamper == "extra_file":
        (files / "extra.json").write_text("{}")
    elif tamper == "pickle":
        (files / "pytorch_model.bin").write_bytes(b"x")
    elif tamper == "missing":
        (files / "config.json").unlink()
    elif tamper == "git_mismatch":
        (files / "config.json").write_bytes(b'{"x": 1}')  # a small git file changed after the plan
    with pytest.raises(DownloadIntegrityError) as err:
        _install(tmp_path, run_dir)
    assert err.value.code == "NET-003"
    assert not snapshot_dir(tmp_path, "model", "org/m", REV).exists()


def test_remote_code_must_match_the_security_pin(tmp_path: Path) -> None:
    pinned = tmp_path / "configs" / "models" / "pinned"
    pinned.mkdir(parents=True)
    good = b"print('vetted')\n"
    (pinned / f"m@{REV[:8]}.sha256").write_text(f"{hashlib.sha256(good).hexdigest()}  modeling.py\n")
    assert pinned_hashes(tmp_path, "m", REV) == {"modeling.py": hashlib.sha256(good).hexdigest()}
    run_dir = tmp_path / "outputs" / "runs" / "ok"
    _stage(run_dir, {"model.safetensors": b"w", "modeling.py": good})
    _install(tmp_path, run_dir)
    bad_dir = tmp_path / "outputs" / "runs" / "bad"
    _stage(bad_dir, {"model.safetensors": b"w", "modeling.py": b"import os\n"})
    with pytest.raises(DownloadIntegrityError):
        _install(tmp_path, bad_dir, revision="c" * 40)


@pytest.mark.parametrize("bad", ["link", "huge", "bad_sha", "bad_name", "dupe"])
def test_a_plan_the_job_tampered_with_installs_nothing(tmp_path: Path, bad: str) -> None:
    """security D2: the plan is the job's file: read no-follow, bounded, schema-checked."""
    run_dir = tmp_path / "outputs" / "runs" / "dl"
    _stage(run_dir, {"model.safetensors": b"w", "config.json": b"{}"})
    plan_path = run_dir / "stage" / "plan.json"
    plan = json.loads(plan_path.read_text())
    if bad == "link":
        real = tmp_path / "elsewhere.json"
        real.write_text(plan_path.read_text())
        plan_path.unlink()
        plan_path.symlink_to(real)
    elif bad == "huge":
        plan_path.write_text(json.dumps({"files": plan["files"], "pad": "x" * (17 * 1024 * 1024)}))
    elif bad == "bad_sha":
        plan["files"][0]["sha256"] = "XYZ"
        plan_path.write_text(json.dumps(plan))
    elif bad == "bad_name":
        plan["files"][0]["name"] = "../model.safetensors"
        plan_path.write_text(json.dumps(plan))
    elif bad == "dupe":
        plan["files"].append(dict(plan["files"][0]))
        plan_path.write_text(json.dumps(plan))
    with pytest.raises(DownloadIntegrityError):
        _install(tmp_path, run_dir)
    assert not snapshot_dir(tmp_path, "model", "org/m", REV).exists()


def test_an_existing_snapshot_is_used_only_if_it_matches_the_plan(tmp_path: Path) -> None:
    """security D3: never trust what is already in the cache; never overwrite it either."""
    first = tmp_path / "outputs" / "runs" / "dl1"
    _stage(first, {"model.safetensors": b"w", "config.json": b"{}"})
    target = _install(tmp_path, first)
    again = tmp_path / "outputs" / "runs" / "dl2"
    _stage(again, {"model.safetensors": b"w", "config.json": b"{}"})
    assert _install(tmp_path, again) == target  # matches: used
    (target / "config.json").write_text('{"tampered": true}')
    third = tmp_path / "outputs" / "runs" / "dl3"
    _stage(third, {"model.safetensors": b"w", "config.json": b"{}"})
    with pytest.raises(DownloadIntegrityError):
        _install(tmp_path, third)
    assert (target / "config.json").read_text() == '{"tampered": true}'  # not overwritten: the user decides
