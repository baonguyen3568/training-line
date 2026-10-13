from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath as P

import pytest

from vlm_pipeline.common.host_paths import (
    HostPaths,
    PathOutsideRootError,
    UnsafeMountSourceError,
    check_mount_source,
)

PATHS = HostPaths(P("/app"), P("/home/u/Trainning-model"), P("/srv/ntc-data"))


@pytest.mark.parametrize(
    ("app_path", "host"),
    [
        ("/app/outputs/runs/r1", "/srv/ntc-data/outputs/runs/r1"),
        ("/app/.cache/hf/hub", "/srv/ntc-data/.cache/hf/hub"),
        ("/app/data/uploads/u1", "/srv/ntc-data/data/uploads/u1"),
        ("/app/uploads/" + "a" * 32 + ".png", "/srv/ntc-data/uploads/" + "a" * 32 + ".png"),
        ("/app/outputs", "/srv/ntc-data/outputs"),
        ("/app/src/vlm_pipeline", "/home/u/Trainning-model/src/vlm_pipeline"),
        ("/app/configs/train/x.yaml", "/home/u/Trainning-model/configs/train/x.yaml"),
        ("/app/outputs-old/x", "/home/u/Trainning-model/outputs-old/x"),  # a name prefix is not the data dir
        ("/app", "/home/u/Trainning-model"),
    ],
)
def test_data_dirs_go_to_the_data_root_the_rest_to_the_project(app_path: str, host: str) -> None:
    assert PATHS.to_host(app_path) == P(host)


def test_without_a_data_root_everything_stays_in_the_project() -> None:
    paths = HostPaths(P("/app"), P("/home/u/Trainning-model"))
    assert paths.to_host("/app/outputs/runs/r1") == P("/home/u/Trainning-model/outputs/runs/r1")


@pytest.mark.parametrize(
    "app_path",
    [
        "/etc/passwd",
        "/home/u/.config/ntc-training/policy.toml",
        "/application/outputs",  # shares a string prefix with /app, not a parent
        "/app/outputs/../../etc",
        "/app/../etc",
        "outputs/runs/r1",  # relative: never resolved against a guessed cwd
        "/",
    ],
)
def test_paths_outside_the_app_root_are_refused(app_path: str) -> None:
    with pytest.raises(PathOutsideRootError):
        PATHS.to_host(app_path)


@pytest.mark.parametrize("bad", [P("relative"), P("/srv/../etc")])
def test_roots_must_be_absolute_and_clean(bad: P) -> None:
    with pytest.raises(ValueError, match="absolute"):
        HostPaths(P("/app"), P("/home/u/p"), bad)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "outputs" / "runs" / "A").mkdir(parents=True)
    (root / "outputs" / "runs" / "A" / "result.json").write_text("{}")
    return root


def test_a_real_source_is_accepted(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    run = root / "outputs" / "runs" / "A"
    assert check_mount_source(root, run, directory=True) == run
    assert check_mount_source(root, run / "result.json", directory=False) == run / "result.json"
    assert check_mount_source(root, root, directory=True) == root


def test_a_symlinked_source_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    secret = tmp_path / "ssh"
    secret.mkdir()
    (root / "outputs" / "runs" / "A" / "x").symlink_to(secret)  # job A planted it
    with pytest.raises(UnsafeMountSourceError, match="symlink"):
        check_mount_source(root, root / "outputs" / "runs" / "A" / "x", directory=True)


def test_a_symlinked_parent_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "B").mkdir(parents=True)
    (root / "outputs" / "link").symlink_to(elsewhere)
    with pytest.raises(UnsafeMountSourceError, match="outputs/link is a symlink"):
        check_mount_source(root, root / "outputs" / "link" / "B", directory=True)


def test_a_symlinked_root_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (tmp_path / "alias").symlink_to(root)
    with pytest.raises(UnsafeMountSourceError, match="not a real directory"):
        check_mount_source(tmp_path / "alias", tmp_path / "alias" / "outputs", directory=True)


@pytest.mark.parametrize(
    ("rel", "directory", "match"),
    [
        ("outputs/runs/missing", True, "does not exist"),  # the daemon would create it as root
        ("outputs/runs/A/result.json", True, "not a directory"),
        ("outputs/runs/A", False, "not a regular file"),
        ("outputs/runs/A/result.json/x", False, "is not a directory"),
        ("../etc", True, "absolute paths"),
    ],
)
def test_missing_or_wrong_kind_is_refused(tmp_path: Path, rel: str, directory: bool, match: str) -> None:
    root = _tree(tmp_path)
    with pytest.raises(UnsafeMountSourceError, match=match):
        check_mount_source(root, root / rel, directory=directory)


def test_a_source_outside_the_root_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(UnsafeMountSourceError, match="outside"):
        check_mount_source(root, tmp_path, directory=True)


def test_mount_unit_of_hub_and_prepared_paths(tmp_path: Path) -> None:
    """A prepared model's config names sibling folders of its bundle (Parse: the RADIO encoder): mount the bundle."""
    from vlm_pipeline.common.host_paths import hub_repo_dir

    hub = tmp_path / ".cache" / "hf" / "hub" / "models--org--m"
    assert hub_repo_dir(tmp_path, hub / "snapshots" / "rev") == hub
    bundle = tmp_path / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b"
    assert hub_repo_dir(tmp_path, bundle / "model") == bundle
    prepared_root = tmp_path / ".cache" / "prepared"
    assert hub_repo_dir(tmp_path, prepared_root) == prepared_root  # never widened to the root itself
    assert hub_repo_dir(tmp_path, tmp_path / "data" / "mine") == tmp_path / "data" / "mine"
