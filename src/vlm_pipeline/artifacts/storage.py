"""Project storage footprint for ``disk_report.json`` (ADR-005 D6) and the app storage panel.

Read-only. Only counts what belongs to this project: paths under the project root, and docker images that carry
the label ``project=vlm-training``, are local ``vlm-*`` builds, or are pinned in ``docker/IMAGES.lock``.
Thresholds/levels live in ``common.storage_report``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from vlm_pipeline.common.diskguard import read_disk
from vlm_pipeline.common.memguard import read_meminfo
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.storage_report import build_report

PROJECT_LABEL = "project=vlm-training"  # ADR-005 D5
# report key -> path under the project root (".cache" is reported without its "hf" part)
PROJECT_PATHS = {"cache_hf_gb": ".cache/hf", "cache_other_gb": ".cache", "outputs_gb": "outputs", "data_gb": "data"}

# (argv) -> stdout, or None when the command failed / docker is missing
Runner = Callable[[Sequence[str]], str | None]


def du_gb(path: Path) -> float:
    """Allocated size under ``path`` in GiB like ``du -sx``: no symlink following, same fs, hard links once."""
    return du_bytes(path) / GIB


def du_bytes(path: Path) -> int:
    """Allocated bytes under ``path`` (see ``du_gb``)."""
    if not path.exists():
        return 0
    dev = path.stat().st_dev
    seen: set[tuple[int, int]] = set()
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, onerror=lambda _err: None):
        for name in filenames:
            try:
                st = (Path(dirpath) / name).lstat()
            except OSError:
                continue
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_blocks * 512
        dirnames[:] = [d for d in dirnames if _same_dev(Path(dirpath) / d, dev)]
    return total


def _same_dev(path: Path, dev: int) -> bool:
    try:
        return path.lstat().st_dev == dev
    except OSError:  # removed while walking
        return False


def pinned_images(lock_file: Path) -> set[str]:
    """Image refs (column 2) from ``docker/IMAGES.lock``. TODO(S-2, bao-backend): use the backend loader."""
    if not lock_file.exists():
        return set()
    refs: set[str] = set()
    for line in lock_file.read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split("|")]
        if not line.startswith("#") and len(cols) >= 2 and cols[1]:
            refs.add(cols[1])
    return refs


def _is_read_only_docker(argv: Sequence[str]) -> bool:
    """Only ``docker images ...`` and ``docker image inspect ...``; deleting goes through the retention module."""
    return bool(argv) and (argv[0] == "images" or (argv[0] == "image" and len(argv) > 1 and argv[1] == "inspect"))


def _docker_runner(argv: Sequence[str]) -> str | None:
    if not _is_read_only_docker(argv):
        raise ValueError(f"storage only runs read-only docker queries, got: {list(argv)[:2]}")
    docker = shutil.which("docker")
    if docker is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv built in this module, no shell, read-only queries
            [docker, *argv], capture_output=True, text=True, timeout=60, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def project_image_ids(pinned: set[str], run: Runner = _docker_runner) -> set[str] | None:
    """IDs of project images; None when docker cannot be queried. Namespaced ``someone/vlm-x`` is not ours."""
    listing = run(["images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"])
    labelled = run(["images", "--quiet", "--filter", f"label={PROJECT_LABEL}"])
    if listing is None or labelled is None:
        return None
    ids = set(labelled.split())
    for line in listing.splitlines():
        ref, _, image_id = line.partition("\t")
        if ("/" not in ref and ref.startswith("vlm-")) or ref in pinned:
            ids.add(image_id)
    return ids


def dedup_layers_bytes(images: list[tuple[list[str], int]]) -> int:
    """Sum image sizes; an image whose layers extend another listed image only adds its own delta."""
    ordered = sorted(images, key=lambda item: len(item[0]))
    total = 0
    for i, (layers, size) in enumerate(ordered):
        parents = [s for lyr, s in ordered[:i] if lyr and layers[: len(lyr)] == lyr]
        total += size - max(parents) if parents else size
    return total


def docker_project_gb(pinned: set[str], run: Runner = _docker_runner) -> float | None:
    """Project image footprint in GiB, shared base layers counted once; None when docker is unavailable."""
    ids = project_image_ids(pinned, run)
    if ids is None:
        return None
    images: list[tuple[list[str], int]] = []
    for image_id in sorted(ids):
        raw = run(["image", "inspect", "--format", "{{json .RootFS.Layers}}\t{{.Size}}", image_id])
        layers_json, _, size = (raw or "").strip().partition("\t")
        if size.isdigit():
            images.append((json.loads(layers_json or "[]"), int(size)))
    return dedup_layers_bytes(images) / GIB


def measure(root: Path, *, run: Runner = _docker_runner, meminfo: Path = Path("/proc/meminfo")) -> dict[str, object]:
    """Measure disk, memory and project footprint and build the schema-v1 report."""
    disk = read_disk(root)
    mem = read_meminfo(meminfo)
    project = {key: du_gb(root / rel) for key, rel in PROJECT_PATHS.items()}
    project["cache_other_gb"] = max(project["cache_other_gb"] - project["cache_hf_gb"], 0.0)
    docker_gb = docker_project_gb(pinned_images(root / "docker" / "IMAGES.lock"), run)
    if docker_gb is None:
        print("disk_report: docker unavailable, docker_vlm_gb reported as 0", file=sys.stderr)
    project["docker_vlm_gb"] = docker_gb or 0.0
    return build_report(
        ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        disk={"total_gb": disk.total_gb, "used_gb": disk.total_gb - disk.free_gb, "free_gb": disk.free_gb},
        mem={
            "total_gb": mem.mem_total_gb,
            "available_gb": mem.mem_available_gb,
            "buff_cache_gb": mem.cached_gb,
            "swap_used_gb": max(mem.swap_total_gb - mem.swap_free_gb, 0.0),
        },
        project=project,
    )
