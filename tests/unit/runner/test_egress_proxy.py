"""security download-egress §1: the download job's CONNECT proxy lets out only HF, on 443, within its byte budget."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from vlm_pipeline.runner.egress_proxy import EgressProxy, address_allowed, control_command, host_allowed

KEY = "ab" * 16


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("huggingface.co", True),
        ("cdn-lfs-us-1.hf.co", True),
        ("cas-bridge.xethub.hf.co", True),
        ("huggingface.co.evil.com", False),
        ("evilhuggingface.co", False),
        ("1.2.3.4", False),
        ("[::1]", False),
        ("localhost", False),
        ("HUGGINGFACE.CO", True),
    ],
)
def test_only_hf_names_are_allowed(host: str, ok: bool) -> None:
    assert host_allowed(host) is ok


@pytest.mark.parametrize(
    ("address", "ok"),
    [("18.239.50.1", True), ("127.0.0.1", False), ("10.1.2.3", False), ("169.254.169.254", False),
     ("100.64.0.1", False), ("::1", False), ("fe80::1", False), ("fc00::1", False), ("224.0.0.1", False),
     ("64:ff9b::7f00:1", False), ("::127.0.0.1", False), ("::ffff:127.0.0.1", False),  # NAT64 / v4 forms (E1)
     ("2606:4700::1111", False)],  # any IPv6: HF is served over IPv4
)  # fmt: skip
def test_only_public_addresses_are_dialled(address: str, ok: bool) -> None:
    assert address_allowed(address) is ok


async def _upstream() -> tuple[asyncio.Server, int, list[bytes]]:
    """A stand-in for HF: answers each chunk with 1000 bytes and records what it got."""
    seen: list[bytes] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(1024):
            seen.append(data)
            writer.write(b"x" * 1000)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], seen


async def _proxy(answers: list[str]) -> tuple[EgressProxy, asyncio.Server, int, list[str]]:
    _up, up_port, _seen = await _upstream()
    dialled: list[str] = []

    async def resolve(_host: str) -> list[str]:
        return answers

    async def connect(address: str, _port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        dialled.append(address)
        return await asyncio.open_connection("127.0.0.1", up_port)  # the test's stand-in for that address

    proxy = EgressProxy(resolve=resolve, connect=connect)
    server = await asyncio.start_server(proxy.handle, "127.0.0.1", 0)
    return proxy, server, server.sockets[0].getsockname()[1], dialled


async def _connect(port: int, request: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request.encode())
    await writer.drain()
    status = await reader.readline()
    return reader, writer, status


def _auth(job: str = "dl-1", key: str = KEY) -> str:
    return "Proxy-Authorization: Basic " + base64.b64encode(f"{job}:{key}".encode()).decode() + "\r\n"


@pytest.mark.parametrize(
    ("request_text", "status"),
    [
        ("GET http://huggingface.co/ HTTP/1.1\r\nHost: huggingface.co\r\n\r\n", b"405"),
        ("CONNECT huggingface.co:80 HTTP/1.1\r\n{auth}\r\n\r\n", b"403"),
        ("CONNECT 1.2.3.4:443 HTTP/1.1\r\n{auth}\r\n\r\n", b"403"),
        ("CONNECT example.com:443 HTTP/1.1\r\n{auth}\r\n\r\n", b"403"),
        ("CONNECT huggingface.co:443 HTTP/1.1\r\n\r\n", b"407"),  # no credential
        ("CONNECT huggingface.co:443 HTTP/1.1\r\n{bad}\r\n\r\n", b"407"),  # wrong key
    ],
)
def test_refusals(request_text: str, status: bytes) -> None:
    async def run() -> bytes:
        proxy, server, port, dialled = await _proxy(["18.239.50.1"])
        proxy.register("dl-1", KEY, 10_000)
        text = request_text.replace("{auth}", _auth().rstrip("\r\n")).replace("{bad}", _auth(key="cd" * 16).rstrip())
        _r, writer, line = await _connect(port, text)
        writer.close()
        server.close()
        assert dialled == []  # refused before any connection out
        return line

    assert status in asyncio.run(run())


def test_a_name_resolving_to_a_private_address_is_never_dialled() -> None:
    async def run() -> bytes:
        proxy, server, port, dialled = await _proxy(["18.239.50.1", "10.0.0.5"])  # one private answer is enough
        proxy.register("dl-1", KEY, 10_000)
        _r, writer, line = await _connect(port, "CONNECT huggingface.co:443 HTTP/1.1\r\n" + _auth() + "\r\n")
        writer.close()
        server.close()
        assert dialled == []
        return line

    assert b"403" in asyncio.run(run())


def test_a_tunnel_goes_to_the_checked_address_and_is_cut_over_budget() -> None:
    async def run() -> tuple[bytes, int, list[str]]:
        proxy, server, port, dialled = await _proxy(["18.239.50.1"])
        proxy.register("dl-1", KEY, 2500)  # upstream answers 1000 bytes per chunk sent
        reader, writer, line = await _connect(port, "CONNECT huggingface.co:443 HTTP/1.1\r\n" + _auth() + "\r\n")
        await reader.readline()  # end of the 200 head
        got = 0
        for _ in range(10):
            writer.write(b"tls")
            await writer.drain()
            chunk = await asyncio.wait_for(reader.read(4096), 2)
            if not chunk:
                break
            got += len(chunk)
        writer.close()
        server.close()
        return line, got, dialled

    line, got, dialled = asyncio.run(run())
    assert b"200" in line
    assert dialled == ["18.239.50.1"]
    assert got <= 2500  # the byte past the budget is never delivered


def test_the_control_language_is_closed() -> None:
    proxy = EgressProxy()
    ok = json.dumps({"command": "register", "job": "dl-1", "key": KEY, "max_bytes": 10}).encode()
    assert control_command(proxy, ok) == {"ok": True}
    for bad in (
        {"command": "register", "job": "dl-1", "key": KEY, "max_bytes": 10, "extra": 1},
        {"command": "register", "job": "../x", "key": KEY, "max_bytes": 10},
        {"command": "register", "job": "dl-2", "key": "short", "max_bytes": 10},
        {"command": "register", "job": "dl-2", "key": KEY, "max_bytes": 0},
        {"command": "shell", "job": "dl-1"},
    ):
        assert control_command(proxy, json.dumps(bad).encode()) == {"ok": False, "error": "bad_request"}
    assert control_command(proxy, b"not json") == {"ok": False, "error": "bad_request"}
    assert control_command(proxy, json.dumps({"command": "unregister", "job": "dl-1"}).encode()) == {"ok": True}


def test_a_client_that_never_finishes_its_request_is_dropped() -> None:
    """No slowloris: the request head must arrive within HEAD_TIMEOUT_S."""
    from vlm_pipeline.runner import egress_proxy

    async def run() -> bytes:
        egress_proxy.HEAD_TIMEOUT_S = 0.2
        _proxy_obj, server, port, _dialled = await _proxy(["18.239.50.1"])
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"CONNECT huggingface.co:443 HTTP/1.1\r\n")  # and nothing more
        await writer.drain()
        data = await asyncio.wait_for(reader.read(100), 2)
        writer.close()
        server.close()
        return data

    try:
        assert asyncio.run(run()) == b""  # closed by the proxy
    finally:
        egress_proxy.HEAD_TIMEOUT_S = 10.0
