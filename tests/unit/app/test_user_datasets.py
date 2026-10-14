"""ADR-021 / P021 H4: inspect a Hugging Face dataset: resolve job, network-none inspect job, validated result."""

from __future__ import annotations

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
from vlm_pipeline.common.catalog import set_user_model_lookup
from vlm_pipeline.common.jobrun import CONTAINER_ROOT
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.download import inspect as ins

SHA = "e" * 40


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 200), (0, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(downloads, "read_policy", lambda: None)
    monkeypatch.setattr(downloads, "read_disk", lambda _r: SimpleNamespace(free_gb=500.0, total_gb=1000.0))
    yield DownloadService(tmp_path, emit=lambda i, e: None, token=lambda: None)
    set_user_model_lookup(None)


def _fake_jobs(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]], *, block: dict[str, Any] | None = None,
               rows: list[dict[str, Any]] | None = None, inspect_ok: bool = True) -> None:  # fmt: skip
    def run_job(job: Any, manifest: Any, run_dir: Path, **kw: Any) -> Any:
        seen.append({"argv": list(job.argv), "egress": job.download_budget is not None, "gpu": job.gpu})
        if job.argv[2] == "vlm_pipeline.download.hub":  # the resolve job: resolve.json + one sample
            stage = run_dir / "stage"
            sample = stage / "sample" / "train"
            sample.mkdir(parents=True)
            (sample / "a.png").write_bytes(_png())
            data = rows if rows is not None else [{"file_name": "a.png", "question": "Q?", "answer": "A"}]
            (sample / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in data))
            files = [{"name": "train/metadata.jsonl", "size": 10, "sha256": None, "git_sha1": None}]
            dataset = block if block is not None else {
                "configs": ["default"], "has_script": False, "has_parquet_export": False, "config": "default",
                "split": "train", "splits": ["train"], "format": "imagefolder",
                "sample": ["train/metadata.jsonl", "train/a.png"], "split_bytes": 10}  # fmt: skip
            (stage / "resolve.json").write_text(json.dumps({
                "sha": SHA, "gated": False, "license": "mit", "all_files": ["train/metadata.jsonl"], "files": files,
                "total_bytes": 10, "unreadable": [], "dataset": dataset}))  # fmt: skip
        else:  # the data job: the real inspect module on the host paths the container paths name
            args = [a.replace(CONTAINER_ROOT, str(run_dir.parent.parent.parent)) for a in job.argv[3:]]
            if inspect_ok:
                ins.main(args)
        status = RunStatus.COMPLETED if inspect_ok or job.argv[2] == "vlm_pipeline.download.hub" else RunStatus.FAILED
        return manifest.model_copy(update={"status": status, "exit_code": 0})

    monkeypatch.setattr(downloads, "run_job", run_job)


def _client(service: Any) -> TestClient:
    from vlm_pipeline.app.user_assets_api import register_user_asset_routes

    app = FastAPI()
    app.state.downloads = service
    register_user_asset_routes(app)
    return TestClient(app)


def _done(client: TestClient, inspect_id: str) -> dict[str, Any]:
    for _ in range(200):
        reply = client.get(f"/api/user-assets/inspections/{inspect_id}").json()
        if reply["status"] in ("done", "failed"):
            return dict(reply)
        threading.Event().wait(0.02)
    raise AssertionError("inspection did not finish")


def test_an_inspection_maps_columns_and_serves_its_thumbnails(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    _fake_jobs(monkeypatch, seen)
    client = _client(service)
    reply = client.post("/api/user-assets/datasets/inspect", json={"repo": "org/d", "split": "train"})
    assert reply.status_code == 202
    done = _done(client, reply.json()["inspect_id"])
    assert done["status"] == "done"
    result = done["result"]
    got = (result["revision"], result["format"], result["config"], result["split"])
    assert got == (SHA, "imagefolder", "default", "train")
    assert result["suggested"] == {"image": "file_name", "question": "question", "answer": "answer"}
    [sample] = result["samples"]
    assert (sample["question"], sample["answer"]) == ("Q?", "A")
    thumb = client.get(sample["thumb_url"])
    assert (thumb.status_code, thumb.headers["content-type"], thumb.content[:4]) == (200, "image/png", b"\x89PNG")
    assert seen[0]["egress"] is True  # the resolve job: the only one with a network
    assert seen[1]["egress"] is False
    assert seen[1]["gpu"] is False
    assert seen[1]["argv"][:3] == ["python", "-m", "vlm_pipeline.download.inspect"]


def test_a_script_only_dataset_is_data_009(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    block = {"configs": [], "has_script": True, "has_parquet_export": True, "config": None, "split": None,
             "splits": [], "format": "unknown", "sample": []}  # fmt: skip
    _fake_jobs(monkeypatch, [], block=block)
    client = _client(service)
    done = _done(client, client.post("/api/user-assets/datasets/inspect", json={"repo": "org/d"}).json()["inspect_id"])
    assert done["error"] == {"code": "DATA-009", "params": {"repo": "org/d", "has_parquet_export": True}}


def test_a_failed_inspect_job_is_never_a_traceback(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_jobs(monkeypatch, [], inspect_ok=False)
    client = _client(service)
    done = _done(client, client.post("/api/user-assets/datasets/inspect", json={"repo": "org/d"}).json()["inspect_id"])
    assert done["status"] == "failed"
    assert done["error"]["code"] == "DATA-010"


@pytest.mark.parametrize("tamper", [
    lambda r: r.update(columns=[{"name": "<script>", "kind": "text"}]),
    lambda r: r.update(suggested={"question": "not_a_column"}),
    lambda r: r.update(samples=[{"question": "x" * 600}]),
    lambda r: r.update(samples=[{"thumb": 99}]),
    lambda r: r.update(extra=1),
])  # fmt: skip
def test_an_inspect_json_that_does_not_validate_is_refused(tmp_path: Path, tamper: Any) -> None:
    good = {"columns": [{"name": "q", "kind": "text"}], "suggested": {"question": "q"}, "samples": [], "rows_read": 0}
    tamper(good)
    (tmp_path / "inspect.json").write_text(json.dumps(good))
    with pytest.raises(user_datasets.RepoFilesInvalidError):
        user_datasets.read_inspect(tmp_path)


def test_only_a_small_png_thumbnail_is_served(tmp_path: Path) -> None:
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    (thumbs / "0.png").write_bytes(_png())
    (thumbs / "1.png").write_bytes(b"<svg/>")
    (thumbs / "2.png").symlink_to(thumbs / "0.png")
    assert user_datasets.thumb_path(tmp_path, 0) is not None
    assert user_datasets.thumb_path(tmp_path, 1) is None
    assert user_datasets.thumb_path(tmp_path, 2) is None
    assert user_datasets.thumb_path(tmp_path, 3) is None


@pytest.mark.parametrize("body", [{"repo": "org/d", "config": "../x"}, {"repo": "org/d", "split": "a b"},
                                  {"repo": "org/d", "revision": "refs/pr/2"}, {"repo": "o/d--x"}])  # fmt: skip
def test_bad_inspect_requests_never_start_a_job(service: Any, monkeypatch: pytest.MonkeyPatch,
                                                body: dict[str, Any]) -> None:  # fmt: skip
    seen: list[dict[str, Any]] = []
    _fake_jobs(monkeypatch, seen)
    assert _client(service).post("/api/user-assets/datasets/inspect", json=body).status_code in (400, 422)
    assert seen == []


def test_a_model_resolve_id_is_not_an_inspection(service: Any) -> None:
    client = _client(service)
    assert client.get("/api/user-assets/inspections/20261014-1200-rs-deadbeef").status_code == 404
    assert client.get("/api/user-assets/inspections/20261014-1200-rs-deadbeef/thumbs/0.png").status_code == 404
