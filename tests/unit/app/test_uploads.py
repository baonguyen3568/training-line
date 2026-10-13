"""B-7 uploads + trial result: bao-security's conditions (size, format, label, CSRF, ids) and the frontend contract."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app import uploads
from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.common.image_rules import MAX_IMAGE_BYTES
from vlm_pipeline.graph.testing import make_test_registry

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

TOKEN = "t" * 32


def _png(size: tuple[int, int] = (8, 6), fmt: str = "PNG", **save: object) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format=fmt, **save)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    services = DefaultServices(registry=make_test_registry(), root=tmp_path, worker="container")
    with TestClient(create_app(services, secure=False, static_dir=tmp_path)) as c:
        yield c


def _post(c: TestClient, body: bytes, ctype: str = "image/png", **headers: str) -> object:
    return c.post("/api/uploads?locale=en", content=body, headers={"Content-Type": ctype, **headers})


def _leftovers(root: Path) -> list[str]:
    d = root / "uploads"
    return sorted(p.name for p in d.iterdir() if p.name.startswith(".")) if d.exists() else []


def test_upload_reencodes_png_and_serves_it(client: TestClient, tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make: must not survive
    r = _post(
        client,
        _png(fmt="JPEG", exif=exif.tobytes()),
        "image/jpeg",
        **{"X-Upload-Label": "h%C3%B3a%20%C4%91%C6%A1n.jpg"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert (body["label"], body["width"], body["height"]) == ("hóa đơn.jpg", 8, 6)
    path = tmp_path / "uploads" / f"{body['upload_id']}.png"
    assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "uploads").stat().st_mode & 0o777 == 0o700
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG")
    assert b"SecretCam" not in data
    got = client.get(f"/api/uploads/{body['upload_id']}")
    assert got.headers["content-type"] == "image/png"
    assert got.headers["x-content-type-options"] == "nosniff"
    assert got.headers["content-disposition"].startswith("inline")
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("with_length", [True, False])
def test_over_20mb_is_413_and_leaves_nothing(client: TestClient, tmp_path: Path, with_length: bool) -> None:
    big = b"\0" * (MAX_IMAGE_BYTES + 1)
    if with_length:
        r = _post(client, big)
    else:  # chunked: counted while streaming
        r = client.post("/api/uploads", content=iter([big[:1024], big[1024:]]), headers={"Content-Type": "image/png"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "DATA-005"
    assert r.json()["error"]["params"]["reason"] == "too_large"
    assert _leftovers(tmp_path) == []
    assert not list((tmp_path / "uploads").glob("*.png")) if (tmp_path / "uploads").exists() else True


@pytest.mark.parametrize(
    "payload",
    [
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n",  # EPS renamed .png
        _png(fmt="GIF"),  # a real image, but not an allowed format
        b"\xff\xff" + b"\0" * 64,  # GD-like header
    ],
)
def test_disallowed_formats_are_rejected_whatever_the_name(client: TestClient, tmp_path: Path, payload: bytes) -> None:
    r = _post(client, payload, **{"X-Upload-Label": "photo.png"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert (err["code"], err["params"]["reason"]) == ("DATA-005", "unreadable")
    assert err["title"]  # rendered for the UI (locale en)
    assert err["what"]
    assert _leftovers(tmp_path) == []


def test_content_type_outside_the_allowlist_is_415(client: TestClient) -> None:
    assert _post(client, _png(), "application/octet-stream").status_code == 415


def test_label_is_display_only(client: TestClient, tmp_path: Path) -> None:
    r = _post(client, _png(), **{"X-Upload-Label": "..%2F..%2Fetc%2Fpasswd%E2%80%AE.png%00"})
    body = r.json()
    assert body["label"] == "../../etc/passwd.png"  # cleaned text, still just a label
    assert uploads.UPLOAD_ID.match(body["upload_id"])
    assert [p.name for p in (tmp_path / "uploads").glob("*.png")] == [f"{body['upload_id']}.png"]
    assert len(uploads.clean_label("x" * 500)) == uploads.LABEL_MAX


@pytest.mark.parametrize("bad", ["../../etc/passwd", "A" * 32, "0" * 31, "0" * 32 + ".png"])
def test_get_validates_the_id_before_any_path(client: TestClient, bad: str) -> None:
    assert client.get(f"/api/uploads/{bad}").status_code == 404


def test_upload_needs_csrf(tmp_path: Path) -> None:
    services = DefaultServices(registry=make_test_registry(), root=tmp_path, worker="container")
    c = TestClient(create_app(services, token=TOKEN, port=7870), base_url="http://127.0.0.1:7870")
    c.cookies = c.get(f"/?token={TOKEN}", follow_redirects=False).cookies
    r = c.post("/api/uploads", content=_png(), headers={"Content-Type": "image/png", "Origin": "http://127.0.0.1:7870"})
    assert r.status_code == 403


def test_trial_result_survives_restart_and_is_capped(client: TestClient, tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "graph" / "runs" / "try-abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"tags": ["graph:g-1", "node:try"]}))
    assert client.get("/api/runs/g-1/nodes/try/result").json() == {"error": {"code": "RESULT_NOT_READY"}}
    result = {"engine_used": "hf", "text": "x" * (uploads.RESULT_TEXT_MAX + 5), "truncated": False, "secret_path": "/x"}
    (run_dir / "result.json").write_text(json.dumps(result))
    body = client.get("/api/runs/g-1/nodes/try/result").json()
    assert (len(body["text"]), body["truncated"], "secret_path" in body) == (uploads.RESULT_TEXT_MAX, True, False)
    assert client.get("/api/runs/g-2/nodes/try/result").status_code == 404  # other graph run
    assert client.get("/api/runs/g-1/nodes/..%2F..%2Fx/result").status_code == 404


def test_oversized_or_linked_result_is_never_read(client: TestClient, tmp_path: Path) -> None:
    """result.json is written by the container: bounded, regular file only (bao-security)."""
    run_dir = tmp_path / "outputs" / "graph" / "runs" / "try-big"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"tags": ["graph:g-9"]}))
    with (run_dir / "result.json").open("wb") as fh:
        fh.truncate(uploads.RESULT_FILE_MAX + 1)  # sparse: big on paper, cheap on disk
    r = client.get("/api/runs/g-9/nodes/try/result")
    assert (r.status_code, r.json()) == (502, {"error": {"code": "RESULT_TOO_LARGE"}})
    (run_dir / "result.json").unlink()
    (tmp_path / "secret.json").write_text(json.dumps({"text": "outside"}))
    (run_dir / "result.json").symlink_to(tmp_path / "secret.json")
    assert client.get("/api/runs/g-9/nodes/try/result").status_code == 404
