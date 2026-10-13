"""G-5 (P018): the app's download service: catalog only, budget, confirm, one at a time, verify then install."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlm_pipeline.app import downloads
from vlm_pipeline.app.downloads import Asset, DownloadService
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.download.install import snapshot_dir

REV = "c" * 40
SMALL = Asset("model", "m", "org/m", REV, "M", 1.0, "apache-2.0", gated=False)


@pytest.fixture
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DownloadService, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(downloads, "resolve", lambda kind, cid: SMALL)
    monkeypatch.setattr(downloads, "read_policy", lambda: None)
    monkeypatch.setattr(downloads, "read_disk", lambda _r: SimpleNamespace(free_gb=500.0, total_gb=1000.0))
    service = DownloadService(tmp_path, emit=lambda i, e: events.append((i, e)), token=lambda: "hf_" + "x" * 30)
    return service, events


def _fake_job(
    monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], *, status: RunStatus = RunStatus.COMPLETED
) -> None:
    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        seen.update(job=job, kw=kw, spec=json.loads((run_dir / "download_spec.json").read_text()))
        stage = run_dir / "stage" / "files"
        stage.mkdir(parents=True)
        data = b"weights"
        (stage / "model.safetensors").write_bytes(data)
        plan = [{"name": "model.safetensors", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "git_sha1": None}]  # fmt: skip
        (run_dir / "stage" / "plan.json").write_text(json.dumps({"files": plan}))
        return manifest.model_copy(update={"status": status})

    monkeypatch.setattr(downloads, "run_job", run_job)


def _wait(service: DownloadService, download_id: str) -> str:
    for _ in range(200):
        status = service.downloads[download_id].status
        if status in ("completed", "failed"):
            return status
        threading.Event().wait(0.02)
    raise AssertionError("download did not finish")


def test_a_download_runs_without_the_gpu_lock_then_installs(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service, events = svc
    seen: dict[str, Any] = {}
    _fake_job(monkeypatch, seen)
    started = service.start("model", "m")
    assert _wait(service, started["download_id"]) == "completed"
    job = seen["job"]
    assert (job.gpu, job.narrow, job.download_budget > GIB) == (False, True, True)
    assert job.hf_token.startswith("hf_")
    assert "hf_" not in repr(job)  # never in a repr
    assert seen["kw"]["take_gpu_lock"] is False
    assert seen["spec"] == {"kind": "model", "repo": "org/m", "revision": REV, "remote_code_files": [],
                            "max_bytes": job.download_budget, "plan_only": False}  # fmt: skip
    assert (snapshot_dir(service.root, "model", "org/m", REV) / "model.safetensors").read_bytes() == b"weights"
    statuses = [e["status"] for _i, e in events if e.get("type") == "status"]
    assert statuses == ["downloading", "verifying", "completed"]
    assert not (service.root / downloads.ACTIVE_REL).exists()  # the next download may start


def test_a_snapshot_the_app_never_checked_is_not_called_verified(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """security P3: a manual copy / old cache is usable but marked unverified until "Check" passes."""
    service, _events = svc
    target = snapshot_dir(service.root, "model", "org/m", REV)
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"weights")
    reply = service.start("model", "m")
    assert (reply["status"], reply["verified"]) == ("completed", False)
    assert service.downloads == {}
    seen: dict[str, Any] = {}
    _fake_job(monkeypatch, seen)
    checking = service.start("model", "m", check=True)  # plan only: nothing downloaded again
    assert _wait(service, checking["download_id"]) == "completed"
    assert seen["spec"]["plan_only"] is True
    again = service.start("model", "m")
    assert (again["status"], again["verified"]) == ("completed", True)
    (target / "model.safetensors").write_bytes(b"weightsX")  # changed afterwards: no longer verified
    assert service.start("model", "m")["verified"] is False


def test_a_download_lock_left_by_a_dead_app_is_replaced(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """security P2: a killed app must not leave every later download "busy" forever."""
    service, _events = svc
    _fake_job(monkeypatch, {})
    active = service.root / downloads.ACTIVE_REL
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({"download_id": "old", "pid": 999_999_999, "boot_id": "another-boot",
                                  "pidns": "pid:[1]", "starttime": "1"}))  # fmt: skip
    started = service.start("model", "m")
    assert _wait(service, started["download_id"]) == "completed"


def test_one_download_at_a_time(svc: Any) -> None:
    service, _events = svc
    active = service.root / downloads.ACTIVE_REL
    active.parent.mkdir(parents=True)
    active.write_text("other")  # unreadable holder: cannot tell it is gone -> busy
    with pytest.raises(downloads.DownloadBusyError):
        service.start("model", "m")


def test_a_big_download_needs_a_confirmation(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _events = svc
    big = Asset("model", "b", "org/b", REV, "B", 40.0, "apache-2.0", gated=False)
    monkeypatch.setattr(downloads, "resolve", lambda kind, cid: big)
    reply = service.start("model", "b")
    assert (reply["needs_confirm"], reply["threshold_gb"]) == (True, downloads.CONFIRM_GB)
    assert service.downloads == {}


def test_budget_refusals_come_before_any_job(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _events = svc
    monkeypatch.setattr(downloads, "read_policy", lambda: {"download": {"max_gb": 0.5}})
    with pytest.raises(downloads.DownloadQuotaError) as quota:
        service.start("model", "m")
    assert quota.value.params["quota_gb"] == 0.5
    monkeypatch.setattr(downloads, "read_policy", lambda: None)
    monkeypatch.setattr(downloads, "read_disk", lambda _r: SimpleNamespace(free_gb=20.5, total_gb=100.0))
    with pytest.raises(downloads.DownloadDiskError):  # 20 GB reserve on a personal machine
        service.start("model", "m")


def test_a_failed_job_or_bad_files_install_nothing(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _events = svc
    _fake_job(monkeypatch, {}, status=RunStatus.FAILED)
    started = service.start("model", "m")
    assert _wait(service, started["download_id"]) == "failed"
    assert not snapshot_dir(service.root, "model", "org/m", REV).exists()
    assert not (service.root / downloads.ACTIVE_REL).exists()


def test_only_catalog_ids_resolve() -> None:
    with pytest.raises(downloads.DownloadUnknownError):
        downloads.resolve("model", "not-a-model")
    asset = downloads.resolve("model", "qwen3.5-0.8b")
    assert asset.repo.startswith("Qwen/")
    assert len(asset.revision) == 40


def test_the_default_template_names_its_catalog_assets() -> None:
    from vlm_pipeline.common.paths import project_root

    template = json.loads((project_root() / "configs" / "templates" / "invoice_to_json.json").read_text())
    assert downloads.template_assets(template) == [("model", "nemotron-parse-v1.1"), ("dataset", "cord-v2")]
