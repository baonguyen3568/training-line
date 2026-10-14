"""ADR-021 / P021 H2: user-added models: store, host-side resolve, the download-slot job, the API."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app import downloads, user_assets
from vlm_pipeline.app.downloads import DownloadService
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.download import resolver

SHA = "b" * 40
QWEN = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "torch_dtype": "bfloat16"}


def _stage(run_dir: Path, *, config: dict[str, Any] | None = None, files: list[tuple[str, int]] | None = None,
           unreadable: list[str] | None = None, license_id: str | None = "apache-2.0",
           index: dict[str, Any] | None = None, extra_names: list[str] | None = None) -> Path:  # fmt: skip
    files = files if files is not None else [("config.json", 100), ("model.safetensors", 4_000_000_000)]
    meta = run_dir / "stage" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(json.dumps(config if config is not None else QWEN))
    names = [n for n, _ in files] + (extra_names or [])
    if index is not None:
        (meta / "model.safetensors.index.json").write_text(json.dumps(index))
        names.append("model.safetensors.index.json")
        files = [*files, ("model.safetensors.index.json", 50)]
    plan = [{"name": n, "size": s, "sha256": None, "git_sha1": None} for n, s in files]
    body = {"sha": SHA, "gated": False, "license": license_id, "all_files": sorted(names), "files": plan,
            "total_bytes": sum(s for _, s in files), "unreadable": unreadable or []}  # fmt: skip
    (run_dir / "stage" / "resolve.json").write_text(json.dumps(body))
    return run_dir


# ------------------------------------------------------------------------------------------- finish (host side)
def test_a_native_repo_becomes_an_unconfirmed_user_record(tmp_path: Path) -> None:
    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "org/My_Model")
    entry = record.entry
    assert entry.id == "user--org--my-model--bbbbbbbb"
    assert (entry.revision, entry.license, entry.tier, entry.params_b) == (SHA, "apache-2.0", "user", 2.0)
    assert (record.family, record.license_confirmed, record.trial_run_id) == ("qwen3_5", False, None)


def test_params_come_from_the_index_when_there_is_one(tmp_path: Path) -> None:
    run = _stage(tmp_path / "run", index={"metadata": {"total_size": 4_600_000_000}})
    assert user_assets.finish_model(tmp_path, run, "o/x").entry.params_b == 2.3


def test_an_unknown_licence_is_kept_as_unknown(tmp_path: Path) -> None:
    run = _stage(tmp_path / "run", license_id=None)
    assert user_assets.finish_model(tmp_path, run, "o/x").entry.license == "unknown"


def test_an_unreadable_config_is_refused_like_remote_code(tmp_path: Path) -> None:
    run = _stage(tmp_path / "run", unreadable=["tokenizer_config.json"], extra_names=["tokenizer_config.json"])
    with pytest.raises(resolver.ModelNeedsRemoteCodeError) as info:
        user_assets.finish_model(tmp_path, run, "o/x")
    assert info.value.params == {"repo": "o/x", "revision": SHA[:12], "reason": "unreadable_config",
                                 "file": "tokenizer_config.json"}  # fmt: skip


def test_a_python_file_anywhere_blocks_the_repo(tmp_path: Path) -> None:
    run = _stage(tmp_path / "run", extra_names=["sub/modeling_x.py"])
    with pytest.raises(resolver.ModelNeedsRemoteCodeError):
        user_assets.finish_model(tmp_path, run, "o/x")


def test_no_safetensors_is_refused(tmp_path: Path) -> None:
    run = _stage(tmp_path / "run", files=[("config.json", 100), ("pytorch_model.bin", 10)])
    with pytest.raises(resolver.ModelArchUnsupportedError) as info:
        user_assets.finish_model(tmp_path, run, "o/x")
    assert info.value.params["reason"] == "no_safetensors"


def test_over_the_cap_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(user_assets, "max_model_gb", lambda root: 1.0)
    with pytest.raises(user_assets.UserModelTooLargeError) as info:
        user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    assert info.value.params == {"size_gb": 3.7, "quota_gb": 1.0}


@pytest.mark.parametrize(
    "tamper",
    [
        lambda b: b.update(sha="main"),
        lambda b: b.update(total_bytes=1),
        lambda b: b.update(all_files=["../etc/passwd"]),
        lambda b: b.update(unreadable=["anything.json"]),
        lambda b: b.update(extra=1),
    ],
)
def test_a_resolve_json_that_does_not_validate_is_refused(tmp_path: Path, tamper: Any) -> None:
    run = _stage(tmp_path / "run")
    path = run / "stage" / "resolve.json"
    body = json.loads(path.read_text())
    tamper(body)
    path.write_text(json.dumps(body))
    with pytest.raises(resolver.RepoFilesInvalidError):
        user_assets.finish_model(tmp_path, run, "o/x")


# ------------------------------------------------------------------------------------------- store
def test_the_store_round_trips_and_confirms_the_licence(tmp_path: Path) -> None:
    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    user_assets.save(tmp_path, record)
    assert user_assets.load(tmp_path, record.entry.id) == record
    confirmed = user_assets.confirm_license(tmp_path, record.entry.id)
    assert confirmed is not None
    assert confirmed.license_confirmed
    assert [r.entry.id for r in user_assets.list_all(tmp_path)] == [record.entry.id]
    assert user_assets.delete(tmp_path, record.entry.id)
    assert user_assets.load(tmp_path, record.entry.id) is None


def test_a_hand_edited_record_never_becomes_a_model(tmp_path: Path) -> None:
    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    user_assets.save(tmp_path, record)
    path = tmp_path / user_assets.STORE_REL / f"{record.entry.id}.json"
    body = json.loads(path.read_text())
    body["entry"]["remote_code"] = "pinned_prepared"  # an attempt to run code
    path.write_text(json.dumps(body))
    assert user_assets.load(tmp_path, record.entry.id) is None
    assert user_assets.list_all(tmp_path) == []


@pytest.mark.parametrize("model_id", ["../x", "qwen3.5-2b", "user--../x", "user--A"])
def test_only_user_ids_are_looked_up(tmp_path: Path, model_id: str) -> None:
    assert user_assets.load(tmp_path, model_id) is None
    assert not user_assets.delete(tmp_path, model_id)


# ------------------------------------------------------------------------------------------- the service + API
@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DownloadService:
    monkeypatch.setattr(downloads, "read_policy", lambda: None)
    monkeypatch.setattr(downloads, "read_disk", lambda _r: SimpleNamespace(free_gb=500.0, total_gb=1000.0))
    return DownloadService(tmp_path, emit=lambda i, e: None, token=lambda: None)


def _fake_resolve_job(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], *, exit_code: int = 0,
                      reason: str | None = None) -> None:  # fmt: skip
    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        seen["spec"] = json.loads((run_dir / "download_spec.json").read_text())
        seen["take_gpu_lock"] = kw.get("take_gpu_lock")
        if reason is not None:
            (run_dir / "stage").mkdir(parents=True)
            (run_dir / "stage" / "error.json").write_text(json.dumps({"reason": reason}))
        else:
            _stage(run_dir)
        status = RunStatus.COMPLETED if exit_code == 0 else RunStatus.FAILED
        return manifest.model_copy(update={"status": status, "exit_code": exit_code})

    monkeypatch.setattr(downloads, "run_job", run_job)


def _done(service: DownloadService, resolve_id: str) -> Any:
    for _ in range(200):
        d = service.downloads[resolve_id]
        if d.status in ("completed", "failed"):
            return d
        threading.Event().wait(0.02)
    raise AssertionError("resolve did not finish")


def test_a_resolve_runs_the_plan_only_job_in_the_download_slot(service: DownloadService,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    seen: dict[str, Any] = {}
    _fake_resolve_job(monkeypatch, seen)
    d = service.start_resolve("model", "o/x", "main", lambda run_dir: user_assets.finish_model(
        service.root, run_dir, "o/x").to_api())  # fmt: skip
    done = _done(service, d.download_id)
    assert done.status == "completed"
    assert done.result["id"] == "user--o--x--bbbbbbbb"
    assert (seen["spec"]["resolve"], seen["spec"]["revision"], seen["spec"]["remote_code_files"]) == (True, "main", [])
    assert seen["take_gpu_lock"] is False
    assert not (service.root / downloads.ACTIVE_REL).exists()  # the slot is free again


@pytest.mark.parametrize(("reason", "code"), [("gated_no_token", "NET-005"), ("bogus", "NET-001")])
def test_a_hub_refusal_is_net005_with_its_reason(service: DownloadService, monkeypatch: pytest.MonkeyPatch,
                                                 reason: str, code: str) -> None:  # fmt: skip
    from vlm_pipeline.download.hub import EXIT_HUB

    _fake_resolve_job(monkeypatch, {}, exit_code=EXIT_HUB, reason=reason)
    d = _done(service, service.start_resolve("model", "o/x", "main", lambda run_dir: None).download_id)
    assert d.status == "failed"
    assert d.error["code"] == code  # an unknown reason from the job is never echoed
    if code == "NET-005":
        assert d.error["params"] == {"repo": "o/x", "kind": "model", "reason": reason}


def _client(service: DownloadService) -> TestClient:
    from fastapi import FastAPI

    from vlm_pipeline.app.user_assets_api import register_user_asset_routes

    app = FastAPI()
    app.state.downloads = service
    register_user_asset_routes(app)
    return TestClient(app)


def test_the_api_adds_lists_confirms_and_deletes(service: DownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_resolve_job(monkeypatch, {})
    client = _client(service)
    reply = client.post("/api/user-assets/models", json={"repo": "o/x"})
    assert reply.status_code == 202
    resolve_id = reply.json()["resolve_id"]
    _done(service, resolve_id)
    status = client.get(f"/api/user-assets/resolves/{resolve_id}").json()
    assert status["status"] == "done"
    model = status["model"]
    assert (model["id"], model["license_confirmed"], model["verified"]) == ("user--o--x--bbbbbbbb", False, False)
    assert [m["id"] for m in client.get("/api/user-assets/models").json()["models"]] == [model["id"]]
    assert client.post(f"/api/user-assets/models/{model['id']}/confirm-license",
                       json={"accept": True}).json()["model"]["license_confirmed"]  # fmt: skip
    assert client.post(f"/api/user-assets/models/{model['id']}/confirm-license",
                       json={"accept": False}).status_code == 422  # fmt: skip
    assert client.delete(f"/api/user-assets/models/{model['id']}").status_code == 200
    assert client.get("/api/user-assets/models").json() == {"models": []}


@pytest.mark.parametrize(
    "body",
    [
        {"repo": "nope"},
        {"repo": "o/x--y"},
        {"repo": "o/x", "revision": "refs/pr/1"},
        {"repo": "o/x", "revision": "../main"},
        {"repo": "o/x", "extra": 1},
    ],
)
def test_the_api_refuses_a_bad_repo_or_ref_before_any_job(service: DownloadService, monkeypatch: pytest.MonkeyPatch,
                                                          body: dict[str, Any]) -> None:  # fmt: skip
    seen: dict[str, Any] = {}
    _fake_resolve_job(monkeypatch, seen)
    assert _client(service).post("/api/user-assets/models", json=body).status_code in (400, 422)
    assert seen == {}


def test_a_busy_slot_is_409(service: DownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    service._take_active("someone-else")
    assert _client(service).post("/api/user-assets/models", json={"repo": "o/x"}).status_code == 409


def test_an_unknown_resolve_id_is_404(service: DownloadService) -> None:
    client = _client(service)
    assert client.get("/api/user-assets/resolves/20261014-1200-rs-deadbeef").status_code == 404
    assert client.get("/api/user-assets/resolves/..%2Fx").status_code == 404


def test_the_store_never_writes_through_a_symlink(tmp_path: Path) -> None:
    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "user_assets").symlink_to(outside)
    with pytest.raises(OSError, match="symlink"):
        user_assets.save(tmp_path, record)
