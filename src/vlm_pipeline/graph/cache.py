"""Node cache keys (P009 §5): H(type_id, version, validated params, input artifact hashes, image digest).

Computed on VALIDATED params (defaults filled in, values coerced), so ``{}`` and the explicit defaults - or ``lr=1``
and ``lr=1.0`` - give the same key. Canvas-only fields (``ui``) never reach this function.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel

from vlm_pipeline.nodes.spec import NodeSpec

CACHE_KEY_VERSION = 1  # bump if the key recipe changes (invalidates every cached artifact)


def compute_cache_key(
    spec: NodeSpec,
    params: Mapping[str, object] | BaseModel,
    input_hashes: Mapping[str, str],
    image_digest: str | None,
) -> str:
    """Deterministic, order-independent cache key; raises ``pydantic.ValidationError`` on invalid params."""
    validated = params if isinstance(params, BaseModel) else spec.validate_params(params)
    payload = {
        "v": CACHE_KEY_VERSION,
        "type_id": spec.type_id,
        "version": spec.version,
        "params": validated.model_dump(mode="json"),
        "inputs": dict(sorted(input_hashes.items())),
        "image": image_digest or "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "ck1-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = ["CACHE_KEY_VERSION", "compute_cache_key"]
