"""Per-template decoding overrides (001a §10.7): ``decode_overrides`` in ``configs/train/templates/<id>.yaml``.

Decoding is a choice of the task/template, not of the weights: the export keeps the base ``generation_config.json``
and the Eval / "Try it" nodes apply these on top (order: base -> template -> node param). Lives in ``common`` because
both ``train`` (template loading) and ``eval`` (applying) use it. Host-safe (pydantic only).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DECODE_OVERRIDES_KEY = "decode_overrides"


class DecodeOverrides(BaseModel):
    """Generation-config keys a template may override. Unset = keep the base value.

    ``max_new_tokens`` must also stay <= the base value; that bound needs the base config, so it is checked where the
    overrides are applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    repetition_penalty: float | None = Field(None, ge=1.0, le=2.0)
    no_repeat_ngram_size: int | None = Field(None, ge=0)
    max_new_tokens: int | None = Field(None, ge=1)
    num_beams: int | None = Field(None, ge=1, le=4)
    do_sample: bool | None = None
    temperature: float | None = Field(None, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def _temperature_needs_sampling(self) -> Self:
        if self.temperature is not None and self.do_sample is not True:
            raise ValueError("temperature is only valid together with do_sample: true")
        return self

    def as_dict(self) -> dict[str, Any]:
        """Only the keys the template sets (what the eval record stores as ``applied_overrides``)."""
        return self.model_dump(exclude_none=True)


def decode_overrides_of(template: Mapping[str, Any]) -> DecodeOverrides:
    """The template's ``decode_overrides`` (empty when absent); raises ``pydantic.ValidationError`` when invalid."""
    raw = template.get(DECODE_OVERRIDES_KEY)
    return DecodeOverrides.model_validate({} if raw is None else raw)
