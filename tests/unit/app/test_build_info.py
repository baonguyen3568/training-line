"""/api/health names the running code; the public bug-report card carries only what security approved."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.app.test_api import FakeServices

from vlm_pipeline.app import build_info
from vlm_pipeline.app.main import create_app
from vlm_pipeline.common.paths import project_root


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, str]:
    from vlm_pipeline.artifacts.testing import seed_demo_runs

    seed_demo_runs(tmp_path / "outputs" / "runs")
    monkeypatch.setattr(build_info, "git_state", lambda _root: ("c0ffee" + "0" * 34, True))
    monkeypatch.setattr(build_info, "gpu_name", lambda: "NVIDIA GB10")
    svc = FakeServices(tmp_path)
    run_id = svc.list_runs()[0]["run_id"]
    return TestClient(create_app(svc, secure=False)), run_id  # type: ignore[arg-type]


def test_health_names_the_commit_read_at_startup(
    client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    c, _ = client
    monkeypatch.setattr(build_info, "git_state", lambda _root: ("f" * 40, False))  # a merge after startup
    body = c.get("/api/health").json()
    assert body["ok"] is True
    assert body["commit"] == "c0ffee" + "0" * 34
    assert body["dirty"] is True
    assert body["app_version"].endswith("+c0ffee000000-dirty")


def test_bug_report_without_a_run_has_exactly_the_approved_fields(client: tuple[TestClient, str]) -> None:
    c, _ = client
    body = c.get("/api/bug-report", params={"code": "EXP-001", "locale": "en"}).json()
    assert set(body) == {"code", "error_title", "app_version", "gpu"}
    assert body["error_title"]
    assert "{" not in body["error_title"]
    assert body["gpu"] == "NVIDIA GB10"


def test_bug_report_of_a_run_adds_only_a_hashed_reference(client: tuple[TestClient, str]) -> None:
    c, run_id = client
    body = c.get(f"/api/runs/{run_id}/bug-report", params={"code": "SYS-003"}).json()
    assert set(body) == {"code", "error_title", "app_version", "gpu", "ref_code"}
    assert body["ref_code"] != hashlib.sha256(run_id.encode()).hexdigest()[:8]  # keyed: no dictionary reversal
    assert run_id not in str(body)
    again = c.get(f"/api/runs/{run_id}/bug-report", params={"code": "DATA-007"}).json()
    assert again["ref_code"] == body["ref_code"]  # same install, same run -> same reference


def test_the_ref_key_is_created_once_private_and_never_followed(tmp_path: Path) -> None:
    key = build_info.ref_key(tmp_path)
    path = tmp_path / build_info.REF_KEY_REL
    assert len(key) == 32
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert build_info.ref_key(tmp_path) == key
    path.unlink()
    path.symlink_to(tmp_path / "planted")
    (tmp_path / "planted").write_bytes(b"x" * 32)
    with pytest.raises(OSError, match="Too many levels of symbolic links"):  # O_NOFOLLOW: ELOOP
        build_info.ref_key(tmp_path)


@pytest.mark.parametrize("path", ["/api/bug-report", "/api/runs/{run}/bug-report"])
def test_unknown_codes_and_runs_are_404_and_never_echoed(client: tuple[TestClient, str], path: str) -> None:
    c, run_id = client
    r = c.get(path.format(run=run_id), params={"code": "<script>EVIL"})
    assert r.status_code == 404
    assert "EVIL" not in r.text
    assert c.get("/api/runs/no-such-run/bug-report", params={"code": "SYS-003"}).status_code == 404


def test_the_image_build_commit_is_used_without_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(build_info, "git_state", lambda _root: (None, None))
    monkeypatch.setenv(build_info.BUILD_COMMIT_ENV, "a" * 40)
    info = build_info.read_build_info(tmp_path)
    assert (info.commit, info.dirty) == ("a" * 40, False)
    monkeypatch.delenv(build_info.BUILD_COMMIT_ENV)
    assert build_info.read_build_info(tmp_path).app_version == info.version


def test_real_repo_reads_its_head() -> None:
    info: Any = build_info.read_build_info(project_root())
    assert info.commit is None or len(info.commit) == 40
