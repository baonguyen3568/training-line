"""LAN edge settings for scripts/install.sh (ADR-019): validate them and derive what compose and nginx need.

    python3 -I scripts/lib_lan.py edge <subnet> [subnet to avoid ...]   (the edge and the egress networks)
    python3 -I scripts/lib_lan.py lan <bind> <port> <allow> <hostname>

print KEY=value lines, or one ``ERROR: ...`` line on stderr and exit 1. ``edge`` always runs (the app joins the edge
network even without LAN access): the subnet must not overlap the host's or another docker network's. ``lan`` reads
the host's IPv4 addresses (``ip -4 -o addr``). Standard library only (the host's python3, before any image exists).
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
import sys

PROJECT = "ntc-training"  # compose project name (compose.yaml)
OWN_NETWORKS = frozenset({"ntc-training_edge", "ntc-training_egress"})  # ours: a re-run must not refuse them
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _host_networks() -> list[ipaddress.IPv4Interface]:
    argv = ["ip", "-4", "-o", "addr", "show"]
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603 - host tool
    return [ipaddress.IPv4Interface(line.split()[3]) for line in out.splitlines() if len(line.split()) > 3]


def _docker_subnets() -> tuple[list[ipaddress.IPv4Network], list[ipaddress.IPv4Network]]:
    """(other docker networks' subnets, OUR networks' subnets): ours by compose project label or name."""
    argv = ["docker", "network", "ls", "-q"]
    ids = subprocess.run(argv, capture_output=True, text=True, check=True).stdout.split()  # noqa: S603
    if not ids:
        return [], []
    fmt = '{{.Name}} {{index .Labels "com.docker.compose.project"}} {{range .IPAM.Config}}{{.Subnet}} {{end}}'
    argv = ["docker", "network", "inspect", "--format", fmt, *ids]  # ids: docker's own hex network ids
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603
    others: list[ipaddress.IPv4Network] = []
    ours: list[ipaddress.IPv4Network] = []
    for line in out.splitlines():
        parts = line.split(" ")  # "<name> <project label or empty> <subnet> <subnet> ..."
        name, project, subnets = parts[0], parts[1] if len(parts) > 1 else "", parts[2:]
        mine = name in OWN_NETWORKS or project == PROJECT
        for raw in (r for r in subnets if r):
            net = ipaddress.ip_network(raw, strict=False)
            if isinstance(net, ipaddress.IPv4Network):
                (ours if mine else others).append(net)
    return others, ours


def _taken(avoid: tuple[str, ...]) -> list[ipaddress.IPv4Network]:
    """Networks a new subnet must not overlap: the host's and other docker networks', never our own.

    Re-running the install must pass (ADR-015 "safe to run again"): our networks, and the host bridge interfaces
    docker made for them (``br-...`` inside their subnets), are not "already used by someone else".
    """
    others, ours = _docker_subnets()
    host = [i.network for i in _host_networks() if not any(i.ip in net for net in ours)]
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
    if address not in {i.ip for i in _host_networks()}:
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
