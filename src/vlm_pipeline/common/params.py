"""Base class and UI-hint helper for node/engine parameter models (P004 §8.2: one declaration per parameter)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Params(BaseModel):
    """Parameters of a node/engine: immutable and strict (unknown keys are errors, never silently dropped)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def ui(**hints: object) -> dict[str, Any]:
    """``json_schema_extra`` carrying editor hints: ``Field(1, json_schema_extra=ui(order=0, advanced=True))``.

    Keys (frontend ``UiHint``): widget, order, preset, advanced, option_keys, summary.
    """
    return {"x-ui": hints}
