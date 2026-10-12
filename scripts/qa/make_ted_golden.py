"""Generate tests/fixtures/eval/ted_golden.json: tree pairs + Donut-cost tree edit distance computed by ``zss``.

Independent oracle for ``vlm_pipeline.eval.cord.tree_edit_distance`` (Zhang-Shasha, written in-house so the project
needs no extra dependency). zss is used only here, never at runtime.
Usage: uv run --no-project --with zss==1.2.0 --with nltk==3.9.1 python scripts/qa/make_ted_golden.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import zss  # type: ignore[import-untyped]
from nltk import edit_distance  # type: ignore[import-untyped]

OUT = Path("tests/fixtures/eval/ted_golden.json")
N_PAIRS = 40
SEED = 20261012
KEYS = ["nm", "cnt", "price", "unitprice", "sub", "total", "menu", "cash"]
LEAF = "<leaf>"


def _ins(node: zss.Node) -> int:
    return len(node.label.replace(LEAF, "")) if LEAF in node.label else 1


def _upd(a: zss.Node, b: zss.Node) -> int:
    la, lb = LEAF in a.label, LEAF in b.label
    if la and lb:
        return int(edit_distance(a.label.replace(LEAF, ""), b.label.replace(LEAF, "")))
    if la != lb:
        return 1 + len((a if la else b).label.replace(LEAF, ""))
    return int(a.label != b.label)


def _zss_tree(data: Any, label: str = "<root>") -> zss.Node:
    """Donut ``construct_tree_from_dict`` (input already normalised: dict values are lists)."""
    node = zss.Node(label)
    if isinstance(data, dict):
        for key, value in data.items():
            node.addkid(_zss_tree(value, key))
    elif all(isinstance(item, dict) for item in data):
        for item in data:
            node.addkid(_zss_tree(item, "<subtree>"))
    else:
        for item in data:
            node.addkid(zss.Node(f"{LEAF}{item}"))
    return node


def _rand_json(rng: random.Random, depth: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in rng.sample(KEYS, rng.randint(1, 4)):
        r = rng.random()
        if depth < 2 and r < 0.3:
            out[key] = [_rand_json(rng, depth + 1) for _ in range(rng.randint(1, 3))]
        elif depth < 2 and r < 0.45:
            out[key] = [_rand_json(rng, depth + 1)]
        elif r < 0.6:
            out[key] = [str(rng.randint(0, 999)) for _ in range(rng.randint(1, 3))]
        else:
            out[key] = ["".join(rng.choice("abc12") for _ in range(rng.randint(1, 6)))]
    return out


def main() -> None:
    rng = random.Random(SEED)  # noqa: S311 - reproducible fixtures, not security
    pairs = []
    for _ in range(N_PAIRS):
        a, b = _rand_json(rng), _rand_json(rng)
        dist = zss.distance(
            _zss_tree(a),
            _zss_tree(b),
            get_children=zss.Node.get_children,
            insert_cost=_ins,
            remove_cost=_ins,
            update_cost=_upd,
        )
        pairs.append({"a": a, "b": b, "distance": int(dist)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generator": "scripts/qa/make_ted_golden.py",
        "oracle": "zss==1.2.0 + nltk==3.9.1 edit_distance",
        "seed": SEED,
    }
    OUT.write_text(json.dumps({"meta": meta, "pairs": pairs}, indent=1) + "\n")
    print(f"wrote {len(pairs)} pairs to {OUT}")


if __name__ == "__main__":
    main()
