"""ADR-021 / P021 H5: add an inspected split, download only it, verify, convert, install; the node reads it."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlm_pipeline.app import downloads, user_datasets
from vlm_pipeline.app.downloads import DownloadService
from vlm_pipeline.app.user_assets import LicenseNotConfirmedError
from vlm_pipeline.common.catalog import set_user_dataset_lookup, set_user_model_lookup, user_dataset_status
from vlm_pipeline.common.jobrun import CONTAINER_ROOT
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.download import convert as cv
from vlm_pipeline.download import inspect as ins
from vlm_pipeline.download.resolver import user_dataset_id

SHA = "f" * 40
DS_ID = user_dataset_id("org/d", SHA, "default", "train")


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (200, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


_META = [{"file_name": f"{n}.png", "question": f"Q{n}?", "answer": f"A{n}"} for n in range(3)]
SPLIT = {
    "train/metadata.jsonl": "\n".join(json.dumps(m) for m in _META).encode(),
    **{f"train/{n}.png": _png() for n in range(3)},
}


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(downloads, "read_policy", lambda: None)
    monkeypatch.setattr(downloads, "read_disk", lambda _r: SimpleNamespace(free_gb=500.0, total_gb=1000.0))
    yield DownloadService(tmp_path, emit=lambda i, e: None, token=lambda: None)
    set_user_model_lookup(None)
    set_user_dataset_lookup(None)


def _host(arg: str, root: Path) -> str:
    return arg.replace(CONTAINER_ROOT, str(root))


def _fake_jobs(monkeypatch: pytest.MonkeyPatch, root: Path, seen: list[dict[str, Any]]) -> None:
    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        spec_file = run_dir / "download_spec.json"
        spec = json.loads(spec_file.read_text()) if spec_file.exists() else None
        seen.append({"module": job.argv[2], "spec": spec})
        if job.argv[2] == "vlm_pipeline.download.hub" and spec and spec.get("resolve"):  # inspect's resolve
            stage = run_dir / "stage"
            for name, data in SPLIT.items():
                (stage / "sample" / name).parent.mkdir(parents=True, exist_ok=True)
                (stage / "sample" / name).write_bytes(data)
            files = [{"name": n, "size": len(d), "sha256": None, "git_sha1": None} for n, d in SPLIT.items()]
            block = {"configs": ["default"], "has_script": False, "has_parquet_export": False, "config": "default",
                     "split": "train", "splits": ["train"], "format": "imagefolder",
                     "sample": ["train/metadata.jsonl", "train/0.png"], "split_bytes": 1}  # fmt: skip
            (stage / "resolve.json").write_text(json.dumps({
                "sha": SHA, "gated": False, "license": "cc-by-4.0", "all_files": sorted(SPLIT), "files": files,
                "total_bytes": sum(f["size"] for f in files), "unreadable": [], "dataset": block}))  # fmt: skip
        elif job.argv[2] == "vlm_pipeline.download.hub":  # the split download: only_files, plan + files
            stage = run_dir / "stage"
            plan = []
            for name in spec["only_files"]:
                data = SPLIT[name]
                (stage / "files" / name).parent.mkdir(parents=True, exist_ok=True)
                (stage / "files" / name).write_bytes(data)
                plan.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                             "git_sha1": None})  # fmt: skip
            (stage / "plan.json").write_text(json.dumps({"files": plan, "total_bytes": sum(p["size"] for p in plan)}))
        elif job.argv[2] == "vlm_pipeline.download.inspect":
            ins.main([_host(a, root) for a in job.argv[3:]])
        elif job.argv[2] == "vlm_pipeline.download.convert":
            cv.main([_host(a, root) for a in job.argv[3:]])
        return manifest.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})

    monkeypatch.setattr(downloads, "run_job", run_job)


def _client(service: Any) -> TestClient:
    from vlm_pipeline.app.user_assets_api import register_user_asset_routes

    app = FastAPI()
    app.state.downloads = service
    register_user_asset_routes(app)
    return TestClient(app)


def _wait(service: Any, job_id: str) -> Any:
    for _ in range(300):
        d = service.downloads[job_id]
        if d.status in ("completed", "failed"):
            return d
        threading.Event().wait(0.02)
    raise AssertionError("job did not finish")


def _added(service: Any, client: TestClient, columns: dict[str, str] | None = None) -> Any:
    inspect_id = client.post("/api/user-assets/datasets/inspect", json={"repo": "org/d"}).json()["inspect_id"]
    assert _wait(service, inspect_id).status == "completed"
    cols = columns or {"image": "file_name", "question": "question", "answer": "answer"}
    return client.post("/api/user-assets/datasets", json={"inspect_id": inspect_id, "columns": cols})


def test_add_confirm_download_convert_then_read_like_a_local_folder(
    service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    _fake_jobs(monkeypatch, service.root, seen)
    client = _client(service)
    added = _added(service, client)
    assert added.status_code == 200
    dataset = added.json()["dataset"]
    assert (dataset["id"], dataset["license_confirmed"], dataset["rows"]) == (DS_ID, False, None)
    with pytest.raises(LicenseNotConfirmedError) as info:
        service.start("dataset", DS_ID)
    assert info.value.params["kind"] == "dataset"
    assert client.post(f"/api/user-assets/datasets/{DS_ID}/confirm-license", json={"accept": True}).status_code == 200
    started = service.start("dataset", DS_ID, confirm=True)
    done = _wait(service, started["download_id"])
    assert (done.status, done.error) == ("completed", None)
    download = next(s for s in seen if s["spec"] and not s["spec"].get("resolve"))
    assert sorted(download["spec"]["only_files"]) == sorted(SPLIT)  # the split only
    folder = user_datasets.data_folder(service.root, DS_ID)
    labels = [json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines()]
    assert [lbl["target"] for lbl in labels] == ["A0", "A1", "A2"]
    record = user_datasets.load(service.root, DS_ID)
    assert record is not None
    assert record.rows == 3
    status = user_dataset_status(DS_ID)
    assert status is not None
    assert (status["rows"], status["download"], status["license"]) == (3, None, "cc-by-4.0")
    assert service.start("dataset", DS_ID)["status"] == "completed"  # converted: nothing to fetch again

    from vlm_pipeline.app.workers import RealWorker  # the node: the very checks of data.local_folder
    from vlm_pipeline.graph.worker import NodeJob
    from vlm_pipeline.nodes import REGISTRY

    spec = REGISTRY.get("data.hf_load")
    assert spec is not None
    params = spec.validate_params({"user_dataset": DS_ID, "user_repo": "org/d", "user_revision": SHA})
    job = NodeJob("d", spec, params, {}, service.root / "outputs" / "graph" / "runs" / "d", threading.Event(), None,
                  None)  # fmt: skip
    result = RealWorker(service.root)._hub_dataset(job)
    ref = json.loads(Path(result.outputs["dataset"].path).read_text())
    assert ref["records_sha256"]
    assert user_datasets.delete(service.root, DS_ID)
    assert not folder.exists()


def test_a_mapping_naming_an_unknown_column_is_data_010(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_jobs(monkeypatch, service.root, [])
    client = _client(service)
    reply = _added(service, client, {"image": "file_name", "answer": "nope"})
    assert reply.status_code == 400
    assert reply.json()["detail"]["code"] == "DATA-010"
    reply = _added(service, client, {"answer": "answer"})  # no image column at all
    assert reply.json()["detail"]["code"] == "DATA-010"


def test_the_node_says_not_added_or_not_downloaded(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.app.workers import NotCachedError, RealWorker, UserDatasetNotAddedError
    from vlm_pipeline.graph.worker import NodeJob
    from vlm_pipeline.nodes import REGISTRY

    spec = REGISTRY.get("data.hf_load")
    assert spec is not None
    params = spec.validate_params({"user_dataset": DS_ID, "user_repo": "org/d", "user_revision": SHA})
    job = NodeJob("d", spec, params, {}, service.root / "r", threading.Event(), None, None)
    with pytest.raises(UserDatasetNotAddedError) as info:
        RealWorker(service.root)._hub_dataset(job)
    assert info.value.params == {"node": "d", "repo": "org/d", "revision": SHA}
    _fake_jobs(monkeypatch, service.root, [])
    _added(service, _client(service))
    with pytest.raises(NotCachedError) as cached:
        RealWorker(service.root)._hub_dataset(job)
    assert cached.value.params["catalog_id"] == DS_ID
    assert user_dataset_status(DS_ID)["download"]["catalog_id"] == DS_ID  # type: ignore[index]


@pytest.mark.parametrize("stray", ["evil.sh", "images/0.svg", "images/x/0.png"])
def test_a_converted_folder_with_anything_else_is_refused(tmp_path: Path, stray: str) -> None:
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "labels.jsonl").write_text("")
    (out / "report.json").write_text(json.dumps({"rows": 0, "skipped": 0, "reasons": {}, "bytes": 0}))
    (out / stray).parent.mkdir(parents=True, exist_ok=True)
    (out / stray).write_text("x")
    with pytest.raises(user_datasets.DatasetUnreadableError):
        user_datasets._check_converted(out)
