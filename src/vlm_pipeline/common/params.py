"""Base class and UI-hint helper for node/engine parameter models (P004 §8.2: one declaration per parameter)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class Params(BaseModel):
    """Parameters of a node/engine: immutable and strict (unknown keys are errors, never silently dropped)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def ui(**hints: object) -> dict[str, Any]:
    """``json_schema_extra`` carrying editor hints: ``Field(1, json_schema_extra=ui(order=0, advanced=True))``.

    Keys (frontend ``UiHint``): widget, order, preset, advanced, option_keys, summary, option_meta,
    disabled_options, recommended (the value the UI marks "khuyên dùng"; separate from the default on purpose).
    ``disabled_options={"fp8": "i18n.key.of.reason" | None}`` greys out values the worker cannot run yet (key = the
    value as a string: ``"fp8"``, ``"true"``, ``"512"``; None = generic "not supported yet").
    """
    return {"x-ui": hints}


LOCAL_PATH_FORMAT = "local-path"


def _relative_inside(value: str) -> str:
    # Syntactic check only (relative, no '..', no backslash). graph.validate re-checks every `local-path` param
    # against the workspace (percent-encoding, symlinks) and reports SEC-001; the worker checks again at run time.
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("path must be relative to the project and must not contain '..'")
    return value


# A path inside the project. The `format` marker tells graph.validate (SEC-001) and the editor (file picker) apart.
LocalPath = Annotated[str, AfterValidator(_relative_inside), Field(json_schema_extra={"format": LOCAL_PATH_FORMAT})]
