"""Disk / memory / project-footprint report (ADR-005 D6) -> ``outputs/system/disk_report.json`` + a short table.

Read-only except for writing the report file. Only measures what belongs to this project:
paths under the repo root, docker images named ``vlm-*`` and images pinned in ``docker/IMAGES.lock``.

    uv run python scripts/cleanup/disk_report.py            # write report + print table
    uv run python scripts/cleanup/disk_report.py --no-write # print only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vlm_pipeline.common.diskguard import GB, read_disk
from vlm_pipeline.common.memguard import read_meminfo
from vlm_pipeline.common.storage_report import build_report

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs" / "system" / "disk_report.json"
# component -> path under ROOT (measured like `du -sx`)
PROJECT_LABEL = "vlm-training"  # docker label project=vlm-training (ADR-005 D5)
PROJECT_PATHS = {"cache_hf_gb": ".cache/hf", "cache_other_gb": ".cache", "outputs_gb": "outputs", "data_gb": "data"}


def du_gb(path: Path) -> float:
    """Allocated bytes under ``path`` like ``du -sx``: no symlink following, same filesystem, hard links once."""
    if not path.exists():
        return 0.0
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
    return total / GB


def _same_dev(path: Path, dev: int) -> bool:
    try:
        return path.lstat().st_dev == dev
    except OSError:  # removed while walking
        return False


def pinned_images(lock_file: Path) -> set[str]:
    refs: set[str] = set()
    if not lock_file.exists():
        return refs
    for line in lock_file.read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split("|")]
        if line.startswith("#") or len(cols) < 2:
            continue
        refs.add(cols[1])  # "nvcr.io/nvidia/nemo-automodel:26.08"
    return refs


def docker_project_gb() -> float:
    """Sum project image sizes (label ``project=vlm-training``, local ``vlm-*`` repos, IMAGES.lock refs).

    Namespaced repos such as ``someone/vlm-x`` are not ours. 0 if docker is unavailable.
    """
    docker = shutil.which("docker")
    if docker is None:
        return 0.0
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, read-only docker query
            [docker, "images", "--format", '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Label "project"}}'],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    pinned = pinned_images(ROOT / "docker" / "IMAGES.lock")
    ids: set[str] = set()
    for line in out.splitlines():
        ref, image_id, label = (line.split("\t") + ["", ""])[:3]
        if label == PROJECT_LABEL or ("/" not in ref and ref.startswith("vlm-")) or ref in pinned:
            ids.add(image_id)
    images: list[tuple[list[str], int]] = []  # (layer digests, size bytes)
    for image_id in ids:  # dedupe: one image may carry several tags
        raw = subprocess.run(  # noqa: S603 - fixed argv, image id from docker itself
            [docker, "image", "inspect", "--format", "{{json .RootFS.Layers}}\t{{.Size}}", image_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
        layers_json, _, size = raw.partition("\t")
        if size.isdigit():
            images.append((json.loads(layers_json or "[]"), int(size)))
    return _dedup_layers_bytes(images) / GB


def _dedup_layers_bytes(images: list[tuple[list[str], int]]) -> int:
    """Image sizes include base layers; an image built FROM another project image only adds its own delta."""
    images = sorted(images, key=lambda item: len(item[0]))
    total = 0
    for i, (layers, size) in enumerate(images):
        parents = [s for lyr, s in images[:i] if lyr and layers[: len(lyr)] == lyr]
        total += size - max(parents) if parents else size
    return total


def measure() -> dict[str, object]:
    disk = read_disk(ROOT)
    mem = read_meminfo()
    project = {k: du_gb(ROOT / p) for k, p in PROJECT_PATHS.items()}
    project["cache_other_gb"] = max(project["cache_other_gb"] - project["cache_hf_gb"], 0.0)  # .cache minus hf
    project["docker_vlm_gb"] = docker_project_gb()
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


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    Path(fh.name).replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only, do not write the report file")
    args = parser.parse_args(argv)

    report = measure()
    if not args.no_write:
        write_atomic(REPORT, report)
    for section in ("disk", "mem", "project"):
        values = report[section]
        if isinstance(values, dict):
            print(f"{section:8} " + "  ".join(f"{k}={v}" for k, v in values.items()))
    print(f"level    {report['level']}")
    reasons = report["reasons"]
    for reason in reasons if isinstance(reasons, list) else []:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
