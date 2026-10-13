"""Which engine options may NEVER be Advanced params (P016 §3). Owner of the rules: bao-security; ANY edit needs review.

Imported by the generator (scripts/probe/engine_schema_write.py refuses to emit a denied key) and by a test over the
COMMITTED file (configs/engine/*.json can be hand-edited). Matching is per word, case-insensitive: a dotted key is
split on ``.`` and ``_`` (``model.pretrained_model_name_or_path`` -> pretrained, model, name, or, path).

Denied, in this order:
1. Hydra / instantiation keys (``_target_`` ...) and ``trust_remote_code``: they run code.
2. Words naming code, files, places or secrets (path, module, url, token...): a value could load code, read or write
   outside the run, reach the network or leak a secret (``${oc.env:HF_TOKEN}`` is refused separately at run time).
3. Keys the node itself sets (train/recipe.py build_recipe) or the app fixes (catalog pixel limits, decode overrides,
   FusedAdam master weights required by the template standards, one-GPU parallel sizes): one source of truth.
4. Values that are not a bool / int / float / enum / list of scalars: free strings and objects stay out of v1.
Exceptions: FULL key names per node-type pattern, scalar only, each with a reason and an approval line.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping

HYDRA_KEYS = frozenset({"_target_", "_partial_", "_args_", "_recursive_", "_convert_"})
DENIED_WORDS = frozenset(
    {
        *("trust", "path", "dir", "file", "url", "uri", "module", "class", "cls", "fn", "func", "callable"),
        *("import", "hook", "plugin", "script", "cmd", "command", "env", "token", "key", "secret", "password"),
        *("revision", "repo", "cache", "endpoint", "host", "port"),
    }
)
# build_recipe sets these from node params (an Advanced value would silently fight the node's own field).
NODE_CONTROLLED = (
    "step_scheduler.*",
    "optimizer.lr",
    "optimizer.weight_decay",
    "rng.seed",
    "checkpoint.*",
    "distributed.activation_checkpointing",
    "peft.*",
    "freeze_config.*",
    "dataset.*",
    "validation_dataset.*",
    "dataloader.dataset",
    "*dataloader.batch_size",  # the node's batch sizes (step_scheduler) drive the loaders
    "model.pretrained_model_name_or_path",
    "processor.pretrained_model_name_or_path",
    # fixed by the app, not by the node: catalog image policy (test_template_standards), our decode overrides,
    # FusedAdam fp32 master weights (test_template_standards), a single GB10 (no tensor/pipeline/expert/context split)
    "processor.min_pixels",
    "processor.max_pixels",
    "decode_overrides.*",
    "optimizer.master_weights",
    "distributed.*_size",
    "distributed.sequence_parallel",
    "model.backend.enable_deepep",
    "model.backend.dispatcher*",
    "model.backend.fake_*",
)
OFFERED_TYPES = frozenset({"bool", "int", "float", "enum", "list"})

# {node type pattern: {full key: "reason  # approved: bao-security <date>"}} — scalar keys only, never rule 1/2 words
# such as path/file/url/module/token/secret/password, never a Hydra key. Empty until security approves the first list.
_SECTION_ONLY = "scalar, denied only for its section name  # approved: bao-security 2026-10-13"
EXCEPTIONS: Mapping[str, Mapping[str, str]] = {
    "train.*": {
        "loss_fn.fp32_upcast": _SECTION_ONLY,
        "dataloader.collate_fn.drop_overlong": _SECTION_ONLY,
        "dataloader.collate_fn.max_length": _SECTION_ONLY,
        "model.text_config.use_cache": _SECTION_ONLY,
        "dist_env.timeout_minutes": _SECTION_ONLY,  # bounded 1..120 below: a hung job holds the shared GPU lock
    },
}

# Every offered int/float has a range (security: an unbounded number is a DoS on the shared UMA host, e.g. 512
# loader workers). Patterns match the full key; the first match wins; anything unlisted is refused by the test.
RANGES: tuple[tuple[str, float, float], ...] = (
    ("*dataloader.num_workers", 0, 8),
    ("*dataloader.prefetch_factor", 1, 8),
    ("*dataloader.snapshot_every_n_steps", 1, 100_000),
    ("*dataloader.timeout", 0, 600),
    ("dataloader.collate_fn.max_length", 128, 32_768),
    ("dist_env.timeout_minutes", 1, 120),
    ("optimizer.eps", 1e-12, 1e-3),
)


def value_range(key: str) -> tuple[float, float] | tuple[None, None]:
    """(min, max) an Advanced number must stay within, or (None, None) when the key has no range yet (refused)."""
    return next(((lo, hi) for pattern, lo, hi in RANGES if fnmatch.fnmatchcase(key, pattern)), (None, None))


_APPROVED = re.compile(r"# approved: bao-security \d{4}-\d{2}-\d{2}$")
_NEVER_EXCEPTED = frozenset({"trust", "path", "file", "url", "uri", "module", "token", "secret", "password"})


def words(key: str) -> list[str]:
    return [w for part in key.lower().split(".") for w in part.split("_") if w]


def _excepted(key: str, node_type: str) -> bool:
    return any(fnmatch.fnmatchcase(node_type, pattern) and key in keys for pattern, keys in EXCEPTIONS.items())


def denial(key: str, value_type: str, node_type: str = "train.*") -> str | None:
    """Why ``key`` may not be an Advanced param of ``node_type`` (None = allowed)."""
    segments = key.split(".")
    if any(s in HYDRA_KEYS or s.startswith("_") for s in segments) or "trust_remote_code" in segments:
        return "runs_code"
    if any(fnmatch.fnmatchcase(key, pattern) for pattern in NODE_CONTROLLED):
        return "node_controlled"
    if value_type not in OFFERED_TYPES:
        return "type"
    if set(words(key)) & DENIED_WORDS and not _excepted(key, node_type):
        return "denied_word"
    return None


def exception_problems() -> list[str]:
    """Problems in EXCEPTIONS itself (the test fails on any): approval line, never an always-denied word."""
    problems = []
    for pattern, keys in EXCEPTIONS.items():
        for key, reason in keys.items():
            if not _APPROVED.search(reason):
                problems.append(f"{pattern} {key}: no '# approved: bao-security <date>' line")
            if set(words(key)) & _NEVER_EXCEPTED or any(s.startswith("_") for s in key.split(".")):
                problems.append(f"{pattern} {key}: this key can never be an exception")
    return problems


__all__ = [
    "DENIED_WORDS",
    "EXCEPTIONS",
    "NODE_CONTROLLED",
    "RANGES",
    "denial",
    "exception_problems",
    "value_range",
    "words",
]
