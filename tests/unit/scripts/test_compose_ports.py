"""PM 2026-10-14: the app image listened on the CLI default port while compose and nginx used 7870 (502)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _cmd_port() -> str:
    cmd = next(line for line in (ROOT / "docker" / "app.Dockerfile").read_text().splitlines() if line.startswith("CMD"))
    argv = json.loads(cmd.removeprefix("CMD").strip())
    return argv[argv.index("--port") + 1]


def test_the_app_image_listens_where_compose_and_nginx_send_traffic() -> None:
    port = _cmd_port()
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    [published] = compose["services"]["app"]["ports"]
    assert published.rsplit(":", 1)[1] == port  # host port -> container port
    nginx = (ROOT / "docker" / "nginx" / "nginx.conf.template").read_text()
    assert re.search(rf"server app:{port};", nginx)
