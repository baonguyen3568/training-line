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
@pytest.fixture(autouse=True)
def _no_leaked_lookup() -> Any:
    """register_user_asset_routes sets a process-wide lookup: never let it reach another test."""
    from vlm_pipeline.common.catalog import set_user_model_lookup

    yield
    set_user_model_lookup(None)


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


# ------------------------------------------------------------------------------------------- H3: download + lookups
def _added(service: DownloadService, monkeypatch: pytest.MonkeyPatch, *, confirm: bool) -> str:
    _fake_resolve_job(monkeypatch, {})
    client = _client(service)
    resolve_id = client.post("/api/user-assets/models", json={"repo": "o/x"}).json()["resolve_id"]
    model_id = _done(service, resolve_id).result["id"]
    if confirm:
        client.post(f"/api/user-assets/models/{model_id}/confirm-license", json={"accept": True})
    return str(model_id)


def test_no_weight_before_the_licence_is_confirmed(service: DownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = _added(service, monkeypatch, confirm=False)
    with pytest.raises(user_assets.LicenseNotConfirmedError) as info:
        service.start("model", model_id)
    assert info.value.code == "SEC-005"
    assert info.value.params == {"name": "x", "repo": "o/x", "license": "apache-2.0", "catalog_id": model_id,
                                 "kind": "model"}  # fmt: skip


def test_a_confirmation_for_another_revision_does_not_count(service: DownloadService,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    model_id = _added(service, monkeypatch, confirm=True)
    record = user_assets.load(service.root, model_id)
    assert record is not None
    user_assets.save(service.root, record.model_copy(update={"license_confirmed_sha": "c" * 40}))
    with pytest.raises(user_assets.LicenseNotConfirmedError):
        service.start("model", model_id)


def test_a_confirmed_user_model_downloads_its_pinned_revision(service: DownloadService,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    model_id = _added(service, monkeypatch, confirm=True)
    launched: list[Any] = []
    monkeypatch.setattr(service, "_launch", lambda asset, planned, plan_only: launched.append(asset) or {})
    service.start("model", model_id, confirm=True)
    [asset] = launched
    assert (asset.repo, asset.revision, asset.catalog_id) == ("o/x", SHA, model_id)


def test_the_one_model_lookup_finds_user_models_and_their_template(
    service: DownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_pipeline.common.catalog import model_entry, set_user_model_lookup
    from vlm_pipeline.train.config import LoraParams
    from vlm_pipeline.train.recipe import build_recipe

    model_id = _added(service, monkeypatch, confirm=False)
    try:
        entry = model_entry(model_id)
        assert entry is not None
        assert entry.tier == "user"
        assert model_entry("qwen3.5-2b") is not None  # the catalog first, unchanged
        assert model_entry("user--not--added--00000000") is None
        template = user_assets.training_template(service.root, entry)
        recipe = build_recipe(template, LoraParams(quantize_base="nf4"), model_dir="m", dataset_dir="d", run_dir="r")
        assert recipe["quantization"]["llm_int8_skip_modules"] == ["model.visual"]
        assert user_assets.delete(service.root, model_id)
        assert not (service.root / user_assets.STORE_REL / f"{model_id}.train.yaml").exists()
    finally:
        set_user_model_lookup(None)
    assert model_entry(model_id) is None  # no app, no user models


def test_a_user_template_edited_on_disk_never_trains(service: DownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    """security H3: a recipe can name Python callables (_target_): only the bytes recorded at add time are used."""
    from vlm_pipeline.common.catalog import model_entry
    from vlm_pipeline.train.recipe import RecipeError, template_path

    model_id = _added(service, monkeypatch, confirm=False)
    entry = model_entry(model_id)
    assert entry is not None
    assert user_assets.training_template(service.root, entry)  # as added: fine
    with pytest.raises(RecipeError, match="verified loader"):
        template_path(service.root, entry)  # no bare path for a user model
    path = service.root / user_assets.STORE_REL / f"{model_id}.train.yaml"
    path.write_bytes(path.read_bytes() + b"\nmodel: {_target_: os.system}\n")
    with pytest.raises(RecipeError, match="changed on disk"):
        user_assets.training_template(service.root, entry)
    path.unlink()
    path.symlink_to(service.root / "elsewhere.yaml")
    with pytest.raises(RecipeError):
        user_assets.training_template(service.root, entry)


# ------------------------------------------------------------------------------------------- H5c
def test_the_model_cap_is_a_setting_the_user_can_raise(service: DownloadService, monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:  # fmt: skip
    client = _client(service)
    assert client.get("/api/user-assets/limits").json()["max_model_gb"] == 40.0
    for bad in ({"max_model_gb": 0}, {"max_model_gb": 5000}, {"max_model_gb": 50, "x": 1}):
        assert client.put("/api/user-assets/limits", json=bad).status_code == 422
    assert client.put("/api/user-assets/limits", json={"max_model_gb": 2.0}).json() == {"max_model_gb": 2.0}
    assert user_assets.max_model_gb(service.root) == 2.0
    with pytest.raises(user_assets.UserModelTooLargeError):  # 3.7 GB over the user's 2 GB
        user_assets.finish_model(service.root, _stage(tmp_path / "run"), "o/x")
    (service.root / user_assets.LIMITS_REL).write_text('{"max_model_gb": 999999}')  # hand-edited: never trusted
    assert user_assets.max_model_gb(service.root) == user_assets.DEFAULT_MAX_MODEL_GB


def _record(root: Path, license_id: str | None) -> Any:
    record = user_assets.finish_model(root, _stage(root / "run", license_id=license_id), "o/x")
    user_assets.save(root, record)
    return record


@pytest.mark.parametrize(("license_id", "blocked"), [(None, True), ("apache-2.0", False)])
def test_an_unknown_licence_never_leaves_the_app(tmp_path: Path, license_id: str | None, blocked: bool) -> None:
    """ADR-021 A1: a user model whose card names no licence trains and runs, but export is refused (SEC-006)."""
    from vlm_pipeline.app.export_worker import ExportLicenseUnknownError
    from vlm_pipeline.app.workers import _check_export
    from vlm_pipeline.graph.worker import SupportQuery
    from vlm_pipeline.nodes import REGISTRY

    entry = _record(tmp_path, license_id).entry
    spec = REGISTRY.get("export.hf_dir")
    assert spec is not None
    query = SupportQuery("x", spec, spec.validate_params({}), entry, {})
    if not blocked:
        _check_export(query)
        return
    with pytest.raises(ExportLicenseUnknownError) as info:
        _check_export(query)
    assert (info.value.code, info.value.params) == ("SEC-006", {"node": "x", "name": "x", "repo": "o/x"})


def test_a_user_model_is_a_base_model_for_every_worker(tmp_path: Path) -> None:
    """base_of() resolved catalog models only: a user model's train, merge, eval or export failed (UnknownModel)."""
    from vlm_pipeline.app.workers import RealWorker
    from vlm_pipeline.graph.worker import ArtifactRef

    entry = _record(tmp_path, "mit").entry
    worker = RealWorker(tmp_path)
    assert worker._entry_for(ArtifactRef("Model", "", f"{entry.repo}@{entry.revision}")) == entry


def test_the_admin_download_cap_bounds_the_setting(service: DownloadService, monkeypatch: pytest.MonkeyPatch) -> None:
    """security H5c: on a shared host the policy's [download] max_gb is the ceiling; Settings only tighten."""
    monkeypatch.setattr("vlm_pipeline.common.app_config.read_policy", lambda: {"download": {"max_gb": 60}})
    client = _client(service)
    assert client.get("/api/user-assets/limits").json()["admin_cap_gb"] == 60.0
    refused = client.put("/api/user-assets/limits", json={"max_model_gb": 100})
    assert (refused.status_code, refused.json()["detail"]) == (422, {"code": "NET-004", "params": {
        "size_gb": 100.0, "quota_gb": 60.0}})  # fmt: skip
    assert client.put("/api/user-assets/limits", json={"max_model_gb": 50}).status_code == 200
    user_assets.save_limits(service.root, user_assets.UserLimits(max_model_gb=500))  # saved before a stricter policy
    assert user_assets.max_model_gb(service.root) == 60.0
