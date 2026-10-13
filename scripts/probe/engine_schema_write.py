"""P016 §3: run the engine probe in the pinned train image and commit-ready write the ALLOWED options.

    uv run python scripts/probe/engine_schema_write.py            # image = common.images.DEFAULT_TRAIN_IMAGE

Runs ``scripts/probe/engine_schema.py`` in a throwaway ``vlm-probe-engine-schema`` container (offline, read-only
mounts, no GPU), drops every key ``nodes.advanced_denylist`` denies, and writes
``configs/engine/automodel-<tag>.json``. The test over that file re-applies the denylist (the file is reviewable and
hand-editable); an image bump regenerates it and the diff shows which knobs changed. Also prints the denied keys
grouped by reason, for the security review of the list.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.nodes.advanced_denylist import denial, value_range

ENGINE_DIR = Path("configs") / "engine"


def engine_file(image: str, root: Path | None = None) -> Path:
    return (root or project_root()) / ENGINE_DIR / f"automodel-{image.rsplit(':', 1)[-1]}.json"


def probe(image: str, root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as out:
        argv = [
            "docker", "run", "--rm", "--network", "none", "--name", "vlm-probe-engine-schema",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",  # noqa: S108 - in the container
            "--user", f"{subprocess.check_output(['id', '-u'], text=True).strip()}",  # noqa: S607
            "-v", f"{root / 'configs/train/templates'}:/t:ro", "-v", f"{root / 'scripts/probe'}:/p:ro",
            "-v", f"{out}:/out", image, "python", "/p/engine_schema.py", "/t", "/out/engine.json",
        ]  # fmt: skip
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
        data: dict[str, object] = json.loads((Path(out) / "engine.json").read_text(encoding="utf-8"))
        return data


def main() -> int:
    root = project_root()
    image = DEFAULT_TRAIN_IMAGE
    data = probe(image, root)
    keys = data["keys"]
    if not isinstance(keys, list):
        raise SystemExit("probe output has no key list")
    allowed, denied = [], defaultdict(list)
    for entry in keys:
        reason = denial(entry["key"], entry["type"])
        if reason is None:
            item = {k: entry[k] for k in ("key", "type", "default", "choices", "templates")}
            if entry["type"] in ("int", "float"):
                item["min"], item["max"] = value_range(entry["key"])  # type: ignore[misc]
            allowed.append(item)
        else:
            denied[reason].append(entry["key"])
    out = {"image": image, "engine": data["engine"], "engine_version": data["engine_version"], "keys": allowed}
    path = engine_file(image, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(root)}: {len(allowed)} allowed of {len(keys)}")
    for reason, names in sorted(denied.items()):
        print(f"denied {reason} ({len(names)}): {', '.join(sorted(names))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
