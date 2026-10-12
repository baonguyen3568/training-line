"""Pre-run checks of a training recipe against the model checkpoint it targets (host-safe, stdlib only).

Called before taking the GPU lock (train.smoke / graph validate) so a wrong recipe fails fast without GPU time.
Findings carry an error-catalog code + params only (text is rendered from i18n at the app edge).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

PARSE_LOSS_TARGET_SUFFIX = "NemotronParseLoss"
# Default of NemotronParseLoss AND of NVIDIA's reference modeling code when config.json has no value (Parse v1.1).
PARSE_DEFAULT_CLASS_TOKEN_START_IDX = 50000
_SPECIAL_TOKEN = re.compile(r"^<[^<>\s]+>$")


@dataclass(frozen=True)
class RecipeIssue:
    code: str  # error catalog code
    params: Mapping[str, object] = field(default_factory=dict)


def resolve_parse_class_token_start_idx(model_dir: Path) -> int:
    """Loss-weighting threshold the checkpoint was trained with.

    NVIDIA's reference ``hf_nemotron_parse_modeling.py`` uses ``getattr(config, "class_token_start_idx", 50000)``
    and multiplies the loss of labels ``>=`` it by 10: v1.1 has no field (50000 = ``<output_plain>``, all special
    tokens weighted); 2.0 sets 52315 (``<class_Text>``: only class tokens and above). Recipes should take the value
    from here instead of hard-coding it.
    """
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return int(config.get("class_token_start_idx", PARSE_DEFAULT_CLASS_TOKEN_START_IDX))


def _vocab_size(config: Mapping[str, object]) -> int | None:
    decoder = config.get("decoder")
    size = decoder.get("vocab_size") if isinstance(decoder, Mapping) else None
    size = size if size is not None else config.get("vocab_size")
    return int(size) if isinstance(size, int) else None


def _token_at(model_dir: Path, token_id: int) -> str | None:
    """Token string for ``token_id`` from ``tokenizer.json`` (added tokens first), or None if unknown."""
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        return None
    tok = json.loads(path.read_text(encoding="utf-8"))
    for added in tok.get("added_tokens", ()):
        if added.get("id") == token_id:
            return str(added.get("content"))
    vocab = tok.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        return next((str(t) for t, i in vocab.items() if i == token_id), None)
    if isinstance(vocab, list) and 0 <= token_id < len(vocab):
        entry = vocab[token_id]
        return str(entry[0] if isinstance(entry, list) else entry)
    return None


def _issue(param: str, value: object, rule: str, node: str) -> RecipeIssue:
    return RecipeIssue("CFG-005", {"node": node, "param": param, "value": value, "rule": rule})


def check_nemotron_parse(model_dir: Path, recipe: Mapping[str, object], *, node: str = "train") -> list[RecipeIssue]:
    """Check the NemotronParseLoss threshold of ``recipe`` against the checkpoint in ``model_dir``.

    Returns CFG-005 issues (empty list = OK). A recipe without a NemotronParseLoss is not checked.
    """
    loss = recipe.get("loss_fn")
    if not isinstance(loss, Mapping) or not str(loss.get("_target_", "")).endswith(PARSE_LOSS_TARGET_SUFFIX):
        return []
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected = resolve_parse_class_token_start_idx(model_dir)
    param = "loss_fn.class_token_start_idx"
    issues: list[RecipeIssue] = []

    # A recipe that omits the value gets NemotronParseLoss's own default (50000), which is wrong for 2.0.
    given = loss.get("class_token_start_idx", PARSE_DEFAULT_CLASS_TOKEN_START_IDX)
    if given != expected:
        issues.append(_issue(param, given, f"= {expected} (model config.json)", node))

    vocab = _vocab_size(config)
    if vocab is not None and not 0 < expected < vocab:
        issues.append(_issue(param, expected, f"< vocab_size {vocab}", node))

    token = _token_at(model_dir, expected)
    if token is not None and not _SPECIAL_TOKEN.match(token):
        issues.append(_issue(param, expected, "must point at a special <...> token in tokenizer.json", node))
    return issues


__all__ = [
    "PARSE_DEFAULT_CLASS_TOKEN_START_IDX",
    "RecipeIssue",
    "check_nemotron_parse",
    "resolve_parse_class_token_start_idx",
]
