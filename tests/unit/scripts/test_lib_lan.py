"""ADR-019: install.sh's LAN settings check (scripts/lib_lan.py), on a fake host: no ip / docker calls."""

from __future__ import annotations

import importlib.util
import ipaddress
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "lib_lan.py"


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("lib_lan", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    host = [ipaddress.IPv4Interface("127.0.0.1/8"), ipaddress.IPv4Interface("192.168.2.178/24"),
            ipaddress.IPv4Interface("172.30.0.1/16")]  # fmt: skip
    monkeypatch.setattr(module, "_host_networks", lambda: host)
    ours = [ipaddress.IPv4Network("10.231.47.0/29"), ipaddress.IPv4Network("10.231.48.0/28")]  # a previous install
    host += [ipaddress.IPv4Interface("10.231.47.4/29"), ipaddress.IPv4Interface("10.231.48.8/28")]  # their bridges
    monkeypatch.setattr(module, "_docker_subnets", lambda: ([ipaddress.IPv4Network("172.31.0.0/16")], ours))
    return module


def _out(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    return dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())


def test_lan_settings_give_the_names_and_origins(lib: Any, capsys: pytest.CaptureFixture[str]) -> None:
    lib.lan("192.168.2.178", "4000", "", "EdgeXpert-9786")
    out = _out(capsys)
    assert out["LAN_ALLOW"] == "192.168.2.0/24"
    assert out["SERVER_NAMES"] == "192.168.2.178 edgexpert-9786 edgexpert-9786.local"
    assert out["PUBLIC_ORIGINS"].split(",")[0] == "https://192.168.2.178:4000"


def test_the_origins_pass_the_apps_own_check(lib: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from vlm_pipeline.app.security import parse_public_origins

    lib.lan("192.168.2.178", "4000", "", "edgexpert-9786")
    raw = _out(capsys)["PUBLIC_ORIGINS"]
    assert parse_public_origins(raw) == tuple(raw.split(","))


def test_an_odd_hostname_leaves_the_ip_only(lib: Any, capsys: pytest.CaptureFixture[str]) -> None:
    lib.lan("192.168.2.178", "4000", "", "my_box!")
    assert _out(capsys)["SERVER_NAMES"] == "192.168.2.178"


@pytest.mark.parametrize(
    ("bind", "port", "allow"),
    [
        ("0.0.0.0", "4000", ""),  # noqa: S104 - every interface: refused
        ("127.0.0.1", "4000", ""),
        ("192.168.2.99", "4000", ""),  # not this machine
        ("192.168.2.178", "80", ""),
        ("192.168.2.178", "4000", "0.0.0.0/0"),
        ("192.168.2.178", "4000", "10.0.0.0/8"),  # does not contain the bind address
        ("192.168.2.178", "4000", "192.168.2.1/24"),  # not a network address
        ("192.168.2.178", "4000", "8.8.8.0/24"),
    ],
)
def test_bad_lan_settings_stop_the_install(lib: Any, bind: str, port: str, allow: str) -> None:
    with pytest.raises(SystemExit):
        lib.lan(bind, port, allow, "h")


def test_the_edge_network_gives_nginx_its_fixed_address(lib: Any, capsys: pytest.CaptureFixture[str]) -> None:
    lib.edge("10.231.47.0/29")
    assert _out(capsys) == {"EDGE_IP_RANGE": "10.231.47.4/30", "PROXY_PEER": "10.231.47.2"}


@pytest.mark.parametrize("subnet", ["172.30.0.0/29", "172.31.5.0/29", "192.168.2.0/29", "8.8.8.0/29",
                                    "10.231.47.0/30", "10.0.0.0/8", "10.231.47.1/29"])  # fmt: skip
def test_an_overlapping_or_odd_edge_subnet_is_refused(lib: Any, subnet: str) -> None:
    with pytest.raises(SystemExit):
        lib.edge(subnet)


def test_re_running_the_install_accepts_its_own_networks(lib: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """PM 2026-10-14: the second install.sh failed on the edge network the first one made (and its br- bridge)."""
    lib.edge("10.231.47.0/29")
    lib.edge("10.231.48.0/28", "10.231.47.0/29")
    assert "PROXY_PEER=10.231.48.2" in capsys.readouterr().out


def test_our_own_two_subnets_still_must_not_overlap(lib: Any) -> None:
    with pytest.raises(SystemExit):
        lib.edge("10.231.47.0/28", "10.231.47.0/29")
