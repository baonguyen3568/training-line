"""D-7 Settings API: bao-security's conditions (CSRF, body cap, closed schema, admin policy 403, no value out)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app import settings_api
from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.secret_store import AUDIT_REL
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.graph.testing import make_test_registry

TOKEN = "t" * 32
ORIGIN = "http://127.0.0.1:7870"
SECRET = "hf_" + "Ab3" * 12


def _app(tmp_path: Path, *, secure: bool) -> object:
    keydir = tmp_path / "cfg"
    keydir.mkdir(mode=0o700, exist_ok=True)
    services = DefaultServices(registry=make_test_registry(), root=tmp_path / "proj", worker="container")
    kwargs = {"token": TOKEN, "port": 7870} if secure else {"secure": False}
    return create_app(services, static_dir=tmp_path, secret_key_file=keydir / "secrets.key", **kwargs)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(_app(tmp_path, secure=False)) as c:  # type: ignore[arg-type]
        yield c


def test_put_then_get_shows_status_never_the_value(client: TestClient, tmp_path: Path) -> None:
    r = client.put("/api/settings/HF_TOKEN", json={"value": SECRET})
    assert r.status_code == 200
    assert r.json() == {**{k: r.json()[k] for k in ("updated_at",)}, "set": True, "last4": SECRET[-4:],
                        "readable": True, "scope": "unchecked"}  # fmt: skip
    got = client.get("/api/settings")
    assert SECRET not in got.text
    assert SECRET[:-4] not in got.text
    body = got.json()
    assert body["secrets"]["HF_TOKEN"]["set"] is True
    assert body["secrets"]["NGC_API_KEY"] == {"set": False, "updated_at": None, "last4": "", "readable": True,
                                              "scope": "unchecked"}  # fmt: skip
    assert body["key_ok"] is True
    assert client.delete("/api/settings/HF_TOKEN").json() == {"set": False}
    assert client.get("/api/settings").json()["secrets"]["HF_TOKEN"]["set"] is False
    audit = (tmp_path / "proj" / AUDIT_REL).read_text()
    assert SECRET not in audit
    assert [json.loads(line)["action"] for line in audit.splitlines()] == ["set", "delete"]


@pytest.mark.parametrize("key", settings_api.ADMIN_POLICY_KEYS)
def test_admin_policy_keys_are_forbidden(client: TestClient, key: str) -> None:
    assert client.put(f"/api/settings/{key}", json={"value": "personal"}).status_code == 403
    assert client.delete(f"/api/settings/{key}").status_code == 403


@pytest.mark.parametrize(
    ("key", "body", "status"),
    [
        ("VLM_PORT", {"value": "1"}, 422),  # not a setting of this API
        ("HF_TOKEN", {"value": SECRET, "extra": 1}, 422),  # closed schema
        ("HF_TOKEN", {"value": ""}, 422),
        ("HF_TOKEN", {"value": " padded-value "}, 422),
        ("HF_TOKEN", {"value": "x" * 4097}, 422),
    ],
)
def test_bad_requests_are_refused(client: TestClient, key: str, body: dict[str, object], status: int) -> None:
    assert client.put(f"/api/settings/{key}", json=body).status_code == status


def test_body_is_capped_while_reading(client: TestClient) -> None:
    big = b'{"value": "' + b"x" * (settings_api.MAX_BODY_BYTES + 10) + b'"}'
    r = client.put("/api/settings/HF_TOKEN", content=iter([big[:1000], big[1000:]]))  # no Content-Length to trust
    assert r.status_code == 413


def test_put_needs_csrf_and_get_needs_a_session(tmp_path: Path) -> None:
    c = TestClient(_app(tmp_path, secure=True), base_url=ORIGIN)  # type: ignore[arg-type]
    assert c.get("/api/settings").status_code == 401
    c.cookies = c.get(f"/?token={TOKEN}", follow_redirects=False).cookies
    assert c.put("/api/settings/HF_TOKEN", json={"value": SECRET}, headers={"Origin": ORIGIN}).status_code == 403
    csrf = c.get("/api/session").json()["csrf_token"]
    ok = c.put("/api/settings/HF_TOKEN", json={"value": SECRET}, headers={"Origin": ORIGIN, "x-vlm-csrf": csrf})
    assert ok.status_code == 200


def test_unsafe_key_file_is_reported_not_replaced(client: TestClient, tmp_path: Path) -> None:
    key = tmp_path / "cfg" / "secrets.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o644)
    assert client.get("/api/settings").json()["key_ok"] is False
    r = client.put("/api/settings/HF_TOKEN", json={"value": SECRET})
    assert (r.status_code, r.json()["error"]["code"]) == (409, "settings_key_unusable")
    assert key.read_bytes() == b"k" * 32


def test_the_routes_cannot_reach_a_plaintext_secret() -> None:
    source = Path(settings_api.__file__).read_text(encoding="utf-8")
    assert "reveal" not in source.replace("``SecretStore.reveal``", "")  # only the docstring names it


@pytest.mark.parametrize("body", [{"value": SECRET, "extra": 1}, {"value": SECRET + " "}, {"value": SECRET * 200}])
def test_a_refused_body_is_never_echoed(client: TestClient, body: dict[str, object]) -> None:
    r = client.put("/api/settings/HF_TOKEN", json=body)
    assert r.status_code == 422
    assert SECRET not in r.text  # pydantic's default 422 would print the input back
