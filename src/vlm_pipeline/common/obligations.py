"""What each licence asks of the user: ``configs/licenses/obligations.yaml`` (owner: bao-security).

Read by the G-5 consent dialog (model.hf_pick option_meta) and export notices; never paraphrased here. A model's duties
are its licence's plus those of its separately licensed parts.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vlm_pipeline.common.licenses import LICENSES_DIR
from vlm_pipeline.common.paths import project_root

if TYPE_CHECKING:
    from vlm_pipeline.common.catalog import CatalogEntry

OBLIGATIONS_PATH: Path = LICENSES_DIR / "obligations.yaml"


LicenseClass = Literal["commercial_ok", "conditional", "non_commercial", "unknown"]
# Strictest last. unknown (mixed or unreviewed terms) beats non_commercial: the user must never think the terms are
# known when part of the model is unreviewed (bao-security).
_CLASS_ORDER: tuple[LicenseClass, ...] = ("commercial_ok", "conditional", "non_commercial", "unknown")


class Obligation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")  # i18n license.obligation.<id>.{title,body}
    when: Literal["always", "distribute", "cosmos"]
    clause: str = Field(min_length=1)
    text: str | None = None  # verbatim notice the user must keep
    url: str | None = None


class LicenceTerms(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    license_class: LicenseClass
    obligations: tuple[Obligation, ...]


class ObligationsTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    licences: dict[str, LicenceTerms]
    unknown_licence: LicenceTerms


@functools.cache
def load_obligations() -> ObligationsTable:
    path = project_root() / OBLIGATIONS_PATH
    return ObligationsTable.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def model_obligations(entry: CatalogEntry, table: ObligationsTable) -> tuple[LicenseClass, list[dict[str, object]]]:
    """Strictest licence class and the obligations of the model AND its separately licensed parts.

    Each obligation keeps its source ``licence`` and ``part`` and is de-duplicated on (licence, id): OML's
    include_license_copy and CC BY's are two different texts (bao-security). ``when: cosmos`` only for
    ``nvidia/Cosmos-*`` repos (case-sensitive; use a catalog ``family`` field if one is added).
    """
    sources = [(entry.license, "model"), *((p.license, p.name) for p in entry.model_parts)]
    cosmos = entry.repo.startswith("nvidia/Cosmos-")
    classes: list[LicenseClass] = []
    found: dict[tuple[str, str], Obligation] = {}
    parts: dict[tuple[str, str], list[str]] = {}
    for licence, part in sources:
        terms = table.licences.get(licence, table.unknown_licence)
        classes.append(terms.license_class)
        for ob in terms.obligations:
            if ob.when == "cosmos" and not cosmos:
                continue
            found.setdefault((licence, ob.id), ob)
            parts.setdefault((licence, ob.id), []).append(part)
    duties = [{**ob.model_dump(exclude_none=True), "licence": k[0], "parts": parts[k]} for k, ob in found.items()]
    return max(classes, key=_CLASS_ORDER.index), duties
