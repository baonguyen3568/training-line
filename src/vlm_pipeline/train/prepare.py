"""Prepare a model snapshot whose remote code pulls code from another repo (P003 D16/D17, ADR-001 amendment 1).

``auto_map`` entries of the form ``org/repo--module.Class`` make transformers download that repo's code from ``main``
at load time, ignoring the pinned revision. This step builds a self-contained, hash-verified model directory instead:

- every ``.py`` of the model and of each bridged dependency is checked against the sha256 list vetted by bao-security
  (``configs/models/pinned/<name>@<sha8>.sha256``) and *copied* (so the code that runs is the code that was hashed);
- weights/tokenizer files are relative symlinks into the HF hub cache (no second copy of GBs);
- ``config.json`` is rewritten so bridged references point at the local dependency directory, as seen *inside the
  container* (``container_root``). Paths must not contain ``--`` because remote code splits on it.

Host-side and stdlib only. Nothing is downloaded here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class PrepareError(RuntimeError):
    """The snapshot does not match what was vetted, or cannot be rewritten safely."""


@dataclass(frozen=True)
class PinnedCode:
    """A snapshot directory plus the vetted sha256 of every ``.py`` file it may contain."""

    repo_id: str
    revision: str
    snapshot_dir: Path
    sha256_file: Path

    def expected(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in self.sha256_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                out[name.strip()] = digest
        return out


@dataclass(frozen=True)
class Bridge:
    """A dependency referenced from the model config by ``<repo_id>--module.Class`` (e.g. a vision encoder)."""

    code: PinnedCode
    local_name: str  # directory name under the prepared dir; must not contain "--"
    config_paths: tuple[tuple[str, ...], ...] = field(default=(("encoder",),))  # sub-configs referencing it


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_code(code: PinnedCode) -> list[str]:
    """Return problems: a ``.py`` in the snapshot not in the vetted list, or a hash mismatch / missing file."""
    expected = code.expected()
    problems: list[str] = []
    present = {p.name for p in code.snapshot_dir.glob("*.py")}
    for name in sorted(present - expected.keys()):
        problems.append(f"{code.repo_id}: unvetted python file {name}")
    for name, digest in sorted(expected.items()):
        path = code.snapshot_dir / name
        if not path.exists():
            problems.append(f"{code.repo_id}: vetted file missing {name}")
        elif _sha256(path) != digest:
            problems.append(f"{code.repo_id}: sha256 mismatch {name}")
    return problems


def _verify_copies(code: PinnedCode, dest: Path) -> list[str]:
    """Hash the *copies* (what will actually run): the source may change between the check and the copy."""
    problems = []
    for name, digest in code.expected().items():
        copy = dest / name
        if copy.exists() and _sha256(copy) != digest:
            problems.append(f"{code.repo_id}: sha256 mismatch after copy {name}")
    return problems


def _copy_py(code: PinnedCode, dest: Path) -> None:
    for name in code.expected():
        src = code.snapshot_dir / name
        if src.exists():
            shutil.copyfile(src, dest / name)  # follows the hub symlink: a real, hashed copy


def _rewrite_bridge(sub: dict[str, object], repo_id: str, container_dir: str) -> None:
    auto_map = sub.get("auto_map")
    if isinstance(auto_map, dict):
        for key, ref in list(auto_map.items()):
            if isinstance(ref, str) and ref.startswith(f"{repo_id}--"):
                auto_map[key] = f"{container_dir}--{ref.split('--', 1)[1]}"
    if sub.get("_name_or_path") == repo_id:
        sub["_name_or_path"] = container_dir


def _remaining_bridges(node: object, path: str = "") -> list[str]:
    """Any ``org/repo--...`` reference left anywhere in the config (must be none after rewriting)."""
    found: list[str] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            found += _remaining_bridges(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _remaining_bridges(v, f"{path}[{i}]")
    elif isinstance(node, str) and "--" in node and "/" in node.split("--", 1)[0] and not node.startswith("/"):
        found.append(f"{path}={node}")
    return found


def _check_inputs(model: PinnedCode, bridges: list[Bridge], out_dir: Path, container_root: str) -> None:
    if "--" in str(out_dir) or "--" in container_root:
        raise PrepareError("prepared paths must not contain '--' (remote code splits auto_map on it)")
    problems = verify_code(model)
    for b in bridges:
        if "--" in b.local_name or "/" in b.local_name:
            raise PrepareError(f"invalid bridge directory name {b.local_name!r}")
        problems += verify_code(b.code)
    if problems:
        raise PrepareError("; ".join(problems))


def _install_bridge(b: Bridge, config: dict[str, object], out_dir: Path, container_out: str) -> None:
    bdir = out_dir / b.local_name
    bdir.mkdir()
    _copy_py(b.code, bdir)
    bconf = b.code.snapshot_dir / "config.json"
    if bconf.exists():
        shutil.copyfile(bconf, bdir / "config.json")
    for sub_path in b.config_paths:
        sub: object = config
        for key in sub_path:
            sub = sub.get(key) if isinstance(sub, dict) else None
        if not isinstance(sub, dict):
            raise PrepareError(f"config has no sub-config {'.'.join(sub_path)}")
        _rewrite_bridge(sub, b.code.repo_id, f"{container_out}/{b.local_name}")


def _populate_model_dir(model: PinnedCode, model_dir: Path) -> None:
    vetted_py = model.expected()
    for src in model.snapshot_dir.iterdir():
        if src.name == "config.json" or src.name.startswith("."):
            continue
        if src.suffix == ".py":
            if src.name in vetted_py:
                shutil.copyfile(src, model_dir / src.name)
            continue
        (model_dir / src.name).symlink_to(os.path.relpath(src.resolve(), model_dir))


def prepare_bridged_model(
    model: PinnedCode,
    bridges: list[Bridge],
    out_dir: Path,
    *,
    project_root: Path,
    container_root: str = "/workspace",
) -> Path:
    """Build ``out_dir`` (``model/`` + one dir per bridge + ``prepared.json``) and return the model directory."""
    _check_inputs(model, bridges, out_dir, container_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # out_dir is a derived cache owned by this step
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True)

    container_out = f"{container_root}/{out_dir.resolve().relative_to(project_root.resolve()).as_posix()}"
    config = json.loads((model.snapshot_dir / "config.json").read_text(encoding="utf-8"))
    for b in bridges:
        _install_bridge(b, config, out_dir, container_out)
    leftover = _remaining_bridges(config)
    if leftover:
        raise PrepareError(f"unpinned cross-repo references remain: {leftover}")

    _populate_model_dir(model, model_dir)
    problems = _verify_copies(model, model_dir)
    for b in bridges:
        problems += _verify_copies(b.code, out_dir / b.local_name)
    if problems:
        shutil.rmtree(out_dir)
        raise PrepareError("; ".join(problems))
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model": {"repo_id": model.repo_id, "revision": model.revision},
        "bridges": [{"repo_id": b.code.repo_id, "revision": b.code.revision, "dir": b.local_name} for b in bridges],
        "container_dir": f"{container_out}/model",
    }
    (out_dir / "prepared.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return model_dir


def main() -> int:  # pragma: no cover - thin CLI for the P-0 template prep
    import argparse

    from vlm_pipeline.common.paths import project_root

    parser = argparse.ArgumentParser(description="Prepare Nemotron-Parse-v1.1 with a pinned local C-RADIOv2-H")
    parser.parse_args()
    root = project_root()
    hub = root / ".cache" / "hf" / "hub"
    pinned = root / "configs" / "models" / "pinned"
    parse_rev, radio_rev = "fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4", "0d8f4c18c877166eda07ddae1386bcad256b7a6a"
    model = PinnedCode(
        "nvidia/NVIDIA-Nemotron-Parse-v1.1",
        parse_rev,
        hub / "models--nvidia--NVIDIA-Nemotron-Parse-v1.1" / "snapshots" / parse_rev,
        pinned / "nemotron-parse-v1.1@fd1e3c6b.sha256",
    )
    radio = PinnedCode(
        "nvidia/C-RADIOv2-H",
        radio_rev,
        hub / "models--nvidia--C-RADIOv2-H" / "snapshots" / radio_rev,
        pinned / "c-radiov2-h@0d8f4c18.sha256",
    )
    out = root / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b"
    model_dir = prepare_bridged_model(model, [Bridge(radio, "c-radiov2-h_0d8f4c18")], out, project_root=root)
    print(json.dumps({"model_dir": str(model_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
