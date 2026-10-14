"""LAN edge settings for scripts/install.sh (ADR-019): validate them and derive what compose and nginx need.

    python3 -I scripts/lib_lan.py edge <subnet> [subnet to avoid ...]   (the edge and the egress networks)
    python3 -I scripts/lib_lan.py lan <bind> <port> <allow> <hostname>

print KEY=value lines, or one ``ERROR: ...`` line on stderr and exit 1. ``edge`` always runs (the app joins the edge
network even without LAN access): the subnet must not overlap the host's or another docker network's. ``lan`` reads
the host's IPv4 addresses (``ip -4 -o addr``). Standard library only (the host's python3, before any image exists).
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys

PROJECT = "ntc-training"  # compose project name (compose.yaml)
OWN_NETWORKS = frozenset({"ntc-training_edge", "ntc-training_egress"})  # ours: a re-run must not refuse them
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _host_networks() -> list[tuple[str, ipaddress.IPv4Interface]]:
    """(interface name, address) of every IPv4 address of this host."""
    argv = ["ip", "-4", "-o", "addr", "show"]
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603 - host tool
    rows = [line.split() for line in out.splitlines()]
    return [(r[1], ipaddress.IPv4Interface(r[3])) for r in rows if len(r) > 3]


def _docker_subnets() -> tuple[list[ipaddress.IPv4Network], set[str]]:
    """(other docker networks' subnets, the host bridge names of OUR networks): ours by project label or name.

    Read as JSON: docker's text templates print a missing label as "" or "<no value>" depending on the call.
    """
    argv = ["docker", "network", "ls", "-q"]
    ids = subprocess.run(argv, capture_output=True, text=True, check=True).stdout.split()  # noqa: S603
    if not ids:
        return [], set()
    argv = ["docker", "network", "inspect", *ids]  # ids: docker's own hex network ids; JSON array out
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603
    others: list[ipaddress.IPv4Network] = []
    bridges: set[str] = set()
    for net in json.loads(out):
        labels = net.get("Labels") or {}
        if net.get("Name") in OWN_NETWORKS or labels.get("com.docker.compose.project") == PROJECT:
            bridges.add(f"br-{str(net.get('Id', ''))[:12]}")  # the host side of a compose bridge network
            continue
        for cfg in (net.get("IPAM") or {}).get("Config") or []:
            subnet = ipaddress.ip_network(str(cfg.get("Subnet", "")), strict=False) if cfg.get("Subnet") else None
            if isinstance(subnet, ipaddress.IPv4Network):
                others.append(subnet)
    return others, bridges


def _taken(avoid: tuple[str, ...]) -> list[ipaddress.IPv4Network]:
    """Networks a new subnet must not overlap: the host's and other docker networks', never our own.

    Re-running the install must pass (ADR-015 "safe to run again"): our networks, and exactly the host bridges docker
    made for them (``br-<network id[:12]>``), are not "already used by someone else". Any other interface in that
    range (a VPN, a hand-made bridge) still counts (security 2026-10-14).
    """
    others, bridges = _docker_subnets()
    host = [addr.network for ifname, addr in _host_networks() if ifname not in bridges]
    return [*host, *others, *(ipaddress.IPv4Network(a) for a in avoid)]


def edge(raw: str, *avoid: str) -> None:
    """The nginx container gets .2 (its fixed address: the proxy peer); the others get the upper half only."""
    try:
        net = ipaddress.IPv4Network(raw, strict=True)
    except ValueError:
        _fail(f"the subnet must be an IPv4 network like 10.231.47.0/29 (got: {raw})")
    if not net.is_private or not 16 <= net.prefixlen <= 29:
        _fail(f"the subnet must be a private /16../29 network (got: {raw})")
    if clash := next((n for n in _taken(avoid) if n.overlaps(net)), None):
        _fail(f"subnet {raw} overlaps {clash} already used on this machine: set another one in .env.local")
    print(f"EDGE_IP_RANGE={list(net.subnets(prefixlen_diff=1))[1]}")
    print(f"PROXY_PEER={net.network_address + 2}")  # .1 is the bridge gateway


def lan(bind: str, port: str, allow: str, hostname: str) -> None:
    try:
        address = ipaddress.IPv4Address(bind)
    except ValueError:
        _fail(f"VLM_LAN_BIND must be an IPv4 address of this machine (got: {bind})")
    if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
        _fail(f"VLM_LAN_BIND must be this machine's LAN address, not {bind}")
    if address not in {addr.ip for _name, addr in _host_networks()}:
        _fail(f"{bind} is not an address of this machine (ip -4 addr)")
    if not (port.isascii() and port.isdigit() and 1024 <= int(port) <= 65535):
        _fail(f"VLM_LAN_PORT must be 1024..65535 (got: {port})")
    allow = allow or str(ipaddress.IPv4Interface(f"{bind}/24").network)
    try:
        allowed = ipaddress.IPv4Network(allow, strict=True)
    except ValueError:
        _fail(f"VLM_LAN_ALLOW must be an IPv4 network like 192.168.2.0/24 (got: {allow})")
    if allowed.prefixlen < 16 or address not in allowed or not allowed.is_private:
        _fail(f"VLM_LAN_ALLOW must be a private /16../32 network that contains {bind} (got: {allow})")
    names = [str(address)]
    hostname = hostname.lower()
    if _HOST.fullmatch(hostname):
        names += [hostname, f"{hostname}.local"]
    print(f"LAN_ALLOW={allowed}")
    print(f"SERVER_NAMES={' '.join(names)}")
    print(f"PUBLIC_ORIGINS={','.join(f'https://{n}:{port}' for n in names)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["edge"] and len(args) >= 2:
        edge(*args[1:])
    elif args[:1] == ["lan"] and len(args) == 5:
        lan(*args[1:])
    else:
        _fail("usage: lib_lan.py edge <subnet> [avoid...] | lan <bind> <port> <allow> <hostname>")
