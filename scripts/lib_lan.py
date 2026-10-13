"""LAN edge settings for scripts/install.sh (ADR-019): validate them and derive what compose and nginx need.

    python3 -I scripts/lib_lan.py edge <edge_subnet>
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

EDGE_NETWORK = "ntc-training_edge"  # our own network: re-running the install must not refuse it
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _host_networks() -> list[ipaddress.IPv4Interface]:
    argv = ["ip", "-4", "-o", "addr", "show"]
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603 - host tool
    return [ipaddress.IPv4Interface(line.split()[3]) for line in out.splitlines() if len(line.split()) > 3]


def _docker_subnets() -> list[ipaddress.IPv4Network]:
    argv = ["docker", "network", "ls", "-q"]
    ids = subprocess.run(argv, capture_output=True, text=True, check=True).stdout.split()  # noqa: S603
    if not ids:
        return []
    fmt = "{{.Name}} {{range .IPAM.Config}}{{.Subnet}} {{end}}"
    argv = ["docker", "network", "inspect", "--format", fmt, *ids]  # ids: docker's own hex network ids
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603
    nets: list[ipaddress.IPv4Network] = []
    for line in out.splitlines():
        name, *subnets = line.split()
        for raw in subnets if name != EDGE_NETWORK else []:
            net = ipaddress.ip_network(raw, strict=False)
            if isinstance(net, ipaddress.IPv4Network):
                nets.append(net)
    return nets


def edge(raw: str) -> None:
    """The nginx container gets .2 (its fixed address: the proxy peer); the others get the upper half only."""
    try:
        net = ipaddress.IPv4Network(raw, strict=True)
    except ValueError:
        _fail(f"VLM_EDGE_SUBNET must be an IPv4 network like 10.231.47.0/29 (got: {raw})")
    if not net.is_private or not 16 <= net.prefixlen <= 29:
        _fail(f"VLM_EDGE_SUBNET must be a private /16../29 network (got: {raw})")
    taken = [*(i.network for i in _host_networks()), *_docker_subnets()]
    if clash := next((n for n in taken if n.overlaps(net)), None):
        _fail(f"VLM_EDGE_SUBNET {raw} overlaps {clash} already used on this machine: set another in .env.local")
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
    if args[:1] == ["edge"] and len(args) == 2:
        edge(args[1])
    elif args[:1] == ["lan"] and len(args) == 5:
        lan(*args[1:])
    else:
        _fail("usage: lib_lan.py edge <subnet> | lan <bind> <port> <allow> <hostname>")
