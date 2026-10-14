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


def test_a_bridged_model_fetches_its_bridge_then_is_prepared(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """5b: Parse (pinned_prepared) needs C-RADIOv2-H and the host preparation, in the same download."""
    service, events = svc
    parse = Asset("model", "nemotron-parse-v1.1", "nvidia/NVIDIA-Nemotron-Parse-v1.1", "f" * 40, "Parse", 3.8,
                  "nvidia-open-model", gated=False)  # fmt: skip
    monkeypatch.setattr(downloads, "resolve", lambda kind, cid: parse)
    fetched: list[str] = []
    seen: dict[str, Any] = {}
    _fake_job(monkeypatch, seen)
    real_fetch = DownloadService._fetch

    def fetch(self: DownloadService, d: Any, part: Asset, run_id: str, planned: int, plan_only: bool) -> bool:
        fetched.append(part.repo)
        return real_fetch(self, d, part, run_id, planned, plan_only)

    prepared: list[str] = []
    monkeypatch.setattr(DownloadService, "_fetch", fetch)
    monkeypatch.setattr(downloads, "prepare_catalog_model", lambda root, cid, repo, rev: prepared.append(cid))
    started = service.start("model", "nemotron-parse-v1.1")
    assert started["size_gb"] == 3.8  # the asset; the budget counted the bridge too
    assert _wait(service, started["download_id"]) == "completed"
    assert fetched == ["nvidia/NVIDIA-Nemotron-Parse-v1.1", "nvidia/C-RADIOv2-H"]
    assert prepared == ["nemotron-parse-v1.1"]
    assert "installing" in [e["status"] for _i, e in events if e.get("type") == "status"]


def test_the_slot_is_free_when_the_ui_sees_the_end(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """frontend chains "download all missing": the next POST goes out on the completed event.

    security: the first download's cleanup must then never remove the second one's lock.
    """
    service, _events = svc
    _fake_job(monkeypatch, {})
    fake = downloads.run_job
    hold = threading.Event()
    calls: list[str] = []

    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        calls.append(run_dir.name)
        if len(calls) == 2:
            hold.wait(5)  # the second download is still running while the first one finishes
        return fake(job, manifest, run_dir, **kw)

    monkeypatch.setattr(downloads, "run_job", run_job)
    active = service.root / downloads.ACTIVE_REL
    second: list[str] = []

    def emit(_i: str, e: dict[str, Any]) -> None:
        if e.get("status") == "completed" and not second:
            second.append(service.start("model", "m2")["download_id"])  # would be SYS-002 if the slot were held

    service._emit = emit
    monkeypatch.setattr(downloads, "resolve", lambda kind, cid: SMALL if cid == "m" else
                        Asset("model", "m2", "org/m2", REV, "M2", 1.0, "apache-2.0", gated=False))  # fmt: skip
    first = service.start("model", "m")
    assert _wait(service, first["download_id"]) == "completed"
    for _ in range(100):  # the first download's thread ends (its finally ran)
        if not any(t.name == f"download-{first['download_id']}" for t in threading.enumerate()):
            break
        threading.Event().wait(0.02)
    assert second
    assert json.loads(active.read_text())["download_id"] == second[0]  # still the second one's lock
    hold.set()
    assert _wait(service, second[0]) == "completed"
    assert not active.exists()


def test_eval_data_is_extracted_by_a_no_network_job_then_checked(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """tester 2026-10-13: OCRBench is fetched, then extracted into data/eval/ (host-checked) for the eval node."""
    from vlm_pipeline.download.extracted import MANIFEST, eval_data_dir

    service, _events = svc
    from vlm_pipeline.common.catalog import load_dataset_catalog

    entry = load_dataset_catalog().get("ocrbench")
    assert entry is not None  # the fixture stubs resolve(): build the real asset from the catalog
    ocr = Asset("dataset", entry.id, entry.repo, entry.revision, entry.display_name, entry.disk_gb, entry.license,
                gated=False)  # fmt: skip
    monkeypatch.setattr(downloads, "resolve", lambda kind, cid: ocr)
    jobs: list[Any] = []

    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        jobs.append((job, kw))
        if "--parquet" in job.argv:  # the extraction job: one image, the questions, their manifest
            out = run_dir / "out"
            (out / "images").mkdir(parents=True)
            (out / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff a")
            (out / "questions.jsonl").write_bytes(b"{}\n")
            sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
                    for p in (out / "images" / "0000.jpg", out / "questions.jsonl")]  # fmt: skip
            (out / MANIFEST).write_text("\n".join(sums) + "\n")
        else:  # the download job: the parquet as planned
            stage = run_dir / "stage" / "files" / "data"
            stage.mkdir(parents=True)
            (stage / "test-00000-of-00001.parquet").write_bytes(b"pq")
            plan = [{"name": "data/test-00000-of-00001.parquet", "size": 2,
                     "sha256": hashlib.sha256(b"pq").hexdigest(), "git_sha1": None}]  # fmt: skip
            (run_dir / "stage" / "plan.json").write_text(json.dumps({"files": plan}))
        return manifest.model_copy(update={"status": RunStatus.COMPLETED})

    from vlm_pipeline.download import extracted

    files = {"images/0000.jpg": b"\xff\xd8\xff a", "questions.jsonl": b"{}\n"}  # what the fake job writes
    sums = "".join(f"{hashlib.sha256(data).hexdigest()}  {rel}\n" for rel, data in sorted(files.items()))
    pinned = entry.model_copy(update={"extracted_manifest_sha256": hashlib.sha256(sums.encode()).hexdigest()})
    monkeypatch.setattr(extracted.ExtractSpec, "entry", lambda self: pinned)
    monkeypatch.setattr(downloads, "run_job", run_job)
    monkeypatch.setattr(downloads, "check_source", lambda root, cid: snapshot_dir(
        root, "dataset", ocr.repo, ocr.revision) / "data" / "test-00000-of-00001.parquet")  # fmt: skip
    started = service.start("dataset", "ocrbench")
    assert _wait(service, started["download_id"]) == "completed"
    extract, kw = jobs[1]
    assert (extract.gpu, extract.narrow, extract.egress, extract.hf_token) == (False, True, False, None)
    assert extract.argv[:3] == ("python", "-m", "vlm_pipeline.eval.ocrbench_extract")
    assert kw["take_gpu_lock"] is False
    assert (eval_data_dir(service.root, "ocrbench") / "questions.jsonl").is_file()
    assert service.start("dataset", "ocrbench")["status"] == "completed"  # installed + extracted: nothing to do


def _held_by_another_container(active: Path, *, age_s: float) -> None:
    import os
    import time

    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(json.dumps({"download_id": "old", "pid": 1, "boot_id": "same-boot?", "pidns": "pid:[999]",
                                  "starttime": "1"}))  # fmt: skip
    then = time.time() - age_s
    os.utime(active, (then, then))


def test_a_lock_nobody_refreshes_is_replaced(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM 2026-10-14: the app container was replaced mid-download; its lock (another pid namespace: "cannot
    tell") blocked every later download. A holder refreshes the lock; 5 minutes untouched means nobody holds it."""
    service, _events = svc
    _fake_job(monkeypatch, {})
    monkeypatch.setattr(downloads, "holder_gone", lambda body: None)  # another container: cannot tell
    active = service.root / downloads.ACTIVE_REL
    _held_by_another_container(active, age_s=downloads.ACTIVE_STALE_S + 60)
    started = service.start("model", "m")
    assert _wait(service, started["download_id"]) == "completed"


def test_a_lock_refreshed_recently_is_respected(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _events = svc
    monkeypatch.setattr(downloads, "holder_gone", lambda body: None)
    _held_by_another_container(service.root / downloads.ACTIVE_REL, age_s=10)
    with pytest.raises(downloads.DownloadBusyError):
        service.start("model", "m")


def test_a_running_download_refreshes_its_lock(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    service, _events = svc
    monkeypatch.setattr(downloads, "ACTIVE_HEARTBEAT_S", 0.02)
    release = threading.Event()
    _fake_job(monkeypatch, {})
    fake = downloads.run_job

    def slow(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        release.wait(5)
        return fake(job, manifest, run_dir, **kw)

    monkeypatch.setattr(downloads, "run_job", slow)
    started = service.start("model", "m")
    active = service.root / downloads.ACTIVE_REL
    os.utime(active, (1, 1))  # pretend it is old
    for _ in range(100):
        if os.lstat(active).st_mtime > 1:
            break
        threading.Event().wait(0.02)
    assert os.lstat(active).st_mtime > 1  # refreshed by the heartbeat
    release.set()
    assert _wait(service, started["download_id"]) == "completed"
    service._stop.set()


def test_a_download_waiting_for_memory_says_so(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM 2026-10-14: the prefetch waited 17 min with no word; now every memory poll is an event (need vs avail)."""
    service, events = svc
    _fake_job(monkeypatch, {})
    fake = downloads.run_job

    def waits(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        kw["on_wait"]({"need_gb": 10.0, "avail_gb": 3.9, "waited_s": 30.0, "since": 1.0})
        assert kw["memory_wait_s"] == service.memory_wait_s
        return fake(job, manifest, run_dir, **kw)

    monkeypatch.setattr(downloads, "run_job", waits)
    started = service.start("model", "m")
    assert _wait(service, started["download_id"]) == "completed"
    [waiting] = [e for _i, e in events if e.get("type") == "waiting"]
    assert waiting == {"type": "waiting", "waiting_for": "memory", "need_gb": 10.0, "avail_gb": 3.9, "waited_s": 30.0}


_PART = Asset(kind="dataset", catalog_id="x", repo="org/x", revision="a" * 40, name="x", size_gb=0.1, license="mit",
              gated=False)  # fmt: skip


@pytest.mark.parametrize("exit_code", [3, 4, 5])
def test_the_jobs_own_refusals_are_not_the_network(exit_code: int) -> None:
    """PM 2026-10-14: cord-v2 exit 3 (repo bigger than the catalog) read NET-001; the catalog is wrong, not the net."""
    part = _PART
    assert downloads._job_failure(exit_code, part) == {"code": "SYS-000", "params": {}}


@pytest.mark.parametrize("exit_code", [1, 2, 137, None])
def test_other_job_failures_stay_network(exit_code: int | None) -> None:
    part = _PART
    assert downloads._job_failure(exit_code, part) == {"code": "NET-001", "params": {"reason": "network"}}
