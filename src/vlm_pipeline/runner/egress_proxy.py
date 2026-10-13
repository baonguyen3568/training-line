"""The download job's only way out: an HTTPS CONNECT proxy to Hugging Face (security download-egress §1). Stdlib only.

Runs as the compose service ``egress-proxy`` (runner image), the one container on both the internal ``egress``
network (where download jobs live) and the default network. Rules, each a refusal before any byte goes out:

* only ``CONNECT <host>:443``; plain HTTP, other methods and other ports are refused;
* ``host`` must be ``huggingface.co`` / ``hf.co`` or a sub-domain of them; an IP literal is refused;
* the proxy resolves the name ITSELF and refuses when any address is not global (private, loopback, link-local,
  CGNAT, multicast, reserved), then connects to the address it checked (no second resolution: no DNS rebinding);
* each tunnel needs the job's credential (``Proxy-Authorization: Basic job:key``) that vlm-runner registered over the
  local control socket with a byte budget; the job's tunnels are cut once it has received more than its budget;
* no TLS interception (the HF token stays end to end); the log line has job, host, port, bytes and seconds only.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("vlm.egress")

ALLOWED_SUFFIXES = ("huggingface.co", "hf.co")
PORT = 443
MAX_HEADER_BYTES = 8192
HEAD_TIMEOUT_S = 10.0  # the CONNECT head must arrive in time (no slowloris holding sockets)
_HOST = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_JOB = re.compile(r"^[a-z0-9][a-z0-9-]{0,120}$")
_KEY = re.compile(r"^[0-9a-f]{32,128}$")

Resolve = Callable[[str], Awaitable[list[str]]]
Connect = Callable[[str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]]


def host_allowed(host: str) -> bool:
    """A DNS name under an allowed suffix; never an IP literal or anything else."""
    host = host.lower()
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        return False
    if not _HOST.fullmatch(host):
        return False
    return any(host == s or host.endswith("." + s) for s in ALLOWED_SUFFIXES)


def address_allowed(address: str) -> bool:
    """A public unicast IPv4 address only (no private, loopback, link-local, CGNAT, multicast, reserved).

    IPv6 is refused outright: NAT64 (64:ff9b::/96) and IPv4-compatible (::a.b.c.d) forms of 127.0.0.1 report
    ``is_global`` (security E1); Hugging Face is fully served over IPv4.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_global and not ip.is_multicast


async def _system_resolve(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, PORT, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


async def _system_connect(address: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(address, port)


@dataclass
class _Budget:
    key: str
    max_bytes: int
    received: int = 0
    tunnels: set[asyncio.StreamWriter] = field(default_factory=set)


class EgressProxy:
    def __init__(
        self,
        *,
        resolve: Resolve = _system_resolve,
        connect: Connect = _system_connect,
        peers: ipaddress.IPv4Network | None = None,
    ) -> None:
        self._resolve = resolve
        self._connect = connect
        self._jobs: dict[str, _Budget] = {}
        self._peers = peers  # the egress network: a tunnel from anywhere else is closed unanswered

    def peer_allowed(self, peer: object) -> bool:
        if self._peers is None:
            return True
        host = peer[0] if isinstance(peer, tuple) and peer else None
        try:
            return ipaddress.ip_address(str(host)) in self._peers
        except ValueError:
            return False

    # ------------------------------------------------------------------------------------------ control (runner)
    def register(self, job: str, key: str, max_bytes: int) -> None:
        if not _JOB.fullmatch(job) or not _KEY.fullmatch(key) or not 0 < max_bytes < 2**50:
            raise ValueError("bad registration")
        self._jobs[job] = _Budget(key=key, max_bytes=max_bytes)

    def unregister(self, job: str) -> None:
        budget = self._jobs.pop(job, None)
        for writer in list(budget.tunnels if budget else ()):
            writer.close()

    def _auth(self, header: str | None) -> tuple[str, _Budget] | None:
        if not header or not header.lower().startswith("basic "):
            return None
        try:
            job, _, key = base64.b64decode(header[6:].strip(), validate=True).decode("ascii").partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return None
        budget = self._jobs.get(job)
        if budget is None or not hmac.compare_digest(budget.key.encode(), key.encode()):
            return None
        return job, budget

    # ------------------------------------------------------------------------------------------ tunnels (jobs)
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not self.peer_allowed(writer.get_extra_info("peername")):
                return  # not a download job (security nit): no reply at all
            await self._handle(reader, writer)
        except (OSError, TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            pass
        finally:
            writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
        if len(head) > MAX_HEADER_BYTES:
            return await _refuse(writer, 431)
        lines = head.decode("latin-1").split("\r\n")
        method, _, rest = lines[0].partition(" ")
        target = rest.rsplit(" ", 1)[0]
        if method != "CONNECT":
            return await _refuse(writer, 405)
        host, sep, port = target.rpartition(":")
        if not sep or port != str(PORT) or not host_allowed(host):
            return await _refuse(writer, 403)
        headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:] if line)}
        auth = self._auth(headers.get("proxy-authorization"))
        if auth is None:
            return await _refuse(writer, 407)
        job, budget = auth
        addresses = await self._resolve(host.lower())
        if not addresses or not all(address_allowed(a) for a in addresses):
            return await _refuse(writer, 403)  # any private answer: the name may be rebound to the LAN
        up_reader, up_writer = await self._connect(addresses[0], PORT)  # the address that was checked
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        budget.tunnels.add(writer)
        started, received = time.monotonic(), 0
        try:
            received = await _pipe_both(reader, writer, up_reader, up_writer, budget)
        finally:
            budget.tunnels.discard(writer)
            up_writer.close()
            logger.info("tunnel job=%s host=%s port=%d bytes_in=%d s=%.1f", job, host, PORT, received,
                        time.monotonic() - started)  # fmt: skip


async def _refuse(writer: asyncio.StreamWriter, status: int) -> None:
    writer.write(f"HTTP/1.1 {status} Refused\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()


async def _pipe_both(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    up_reader: asyncio.StreamReader,
    up_writer: asyncio.StreamWriter,
    budget: _Budget,
) -> int:
    """Copy both ways until one side closes; bytes from upstream count against the job's budget (hard cut)."""
    received = 0

    async def down() -> None:
        nonlocal received
        while chunk := await up_reader.read(65536):
            received += len(chunk)
            budget.received += len(chunk)
            if budget.received > budget.max_bytes:
                logger.warning("job over its byte budget (%d > %d): tunnel cut", budget.received, budget.max_bytes)
                return
            writer.write(chunk)
            await writer.drain()

    async def up() -> None:
        while chunk := await reader.read(65536):
            up_writer.write(chunk)
            await up_writer.drain()

    tasks = [asyncio.create_task(down()), asyncio.create_task(up())]
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    return received


# ------------------------------------------------------------------------------------------------ control socket
def _peer_uid(sock: socket.socket) -> int:
    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return int(struct.unpack("3i", creds)[1])


def control_command(proxy: EgressProxy, raw: bytes) -> dict[str, Any]:
    """``register {job, key, max_bytes}`` / ``unregister {job}``; anything else refused (typed, closed)."""
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("not an object")
        if body.get("command") == "register" and set(body) == {"command", "job", "key", "max_bytes"}:
            proxy.register(str(body["job"]), str(body["key"]), int(body["max_bytes"]))
        elif body.get("command") == "unregister" and set(body) == {"command", "job"}:
            proxy.unregister(str(body["job"]))
        else:
            return {"ok": False, "error": "bad_request"}
    except (ValueError, TypeError):
        return {"ok": False, "error": "bad_request"}
    return {"ok": True}


async def serve(
    control_path: Path,
    listen_port: int = 3128,
    allowed_uid: int | None = None,
    listen_host: str = "0.0.0.0",  # noqa: S104 - compose passes the proxy's own address on the egress network
    peers: ipaddress.IPv4Network | None = None,
) -> None:  # pragma: no cover
    proxy = EgressProxy(peers=peers)
    uid = os.getuid() if allowed_uid is None else allowed_uid

    async def on_control(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            sock = writer.get_extra_info("socket")
            if sock is None or _peer_uid(sock) != uid:
                return
            reply = control_command(proxy, await reader.read(4096))
            writer.write(json.dumps(reply).encode())
            await writer.drain()
        finally:
            writer.close()

    control_path.unlink(missing_ok=True)
    control = await asyncio.start_unix_server(on_control, path=str(control_path))
    control_path.chmod(0o660)
    tunnels = await asyncio.start_server(proxy.handle, host=listen_host, port=listen_port)
    async with control, tunnels:
        await asyncio.gather(control.serve_forever(), tunnels.serve_forever())


def main() -> None:  # pragma: no cover - the egress-proxy container entrypoint
    logging.basicConfig(level=logging.INFO)
    control = Path(os.environ.get("VLM_EGRESS_CONTROL", "/run/vlm-egress/control.sock"))
    subnet = os.environ.get("VLM_EGRESS_SUBNET", "")
    listen = os.environ.get("VLM_EGRESS_LISTEN", "")
    asyncio.run(
        serve(
            control,
            allowed_uid=int(os.environ["VLM_RUNNER_UID"]) if "VLM_RUNNER_UID" in os.environ else None,
            listen_host=str(ipaddress.IPv4Address(listen)) if listen else "0.0.0.0",  # noqa: S104 - dev only
            peers=ipaddress.IPv4Network(subnet) if subnet else None,  # a bad value stops the proxy, never widens
        )
    )


__all__ = ["ALLOWED_SUFFIXES", "EgressProxy", "address_allowed", "control_command", "host_allowed", "serve"]


if __name__ == "__main__":  # pragma: no cover
    main()
