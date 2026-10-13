"""Licence, attribution and a minimal model card for an exported model (Proposal 012).

Pure host code: walks the run manifests behind a model artifact back to the catalog model and the datasets it was
trained on, then writes next to the weights:

- ``LICENSE``: the base model's licence, verbatim from ``configs/licenses/<id>.txt`` (locked by sha256);
- ``LICENSES/<id>.txt``: every other licence involved (training data, ...), verbatim;
- ``NOTICE``: one block per component (name, repo@revision, licence, required notice) + what was changed;
- ``README.md``: minimal model card (base model, data, licences, usage, known limitations from the catalog);
- ``ATTRIBUTION.json``: the same facts, machine-readable, with the sha256 of every file in the export.

Licence texts and notice wording are never written here: they come from the vendored texts and the catalog
(bao-research / bao-security). A licence without a locked text fails the export before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vlm_pipeline.common.catalog import Catalog, CatalogEntry, DatasetCatalog, DatasetEntry
from vlm_pipeline.common.licenses import license_text
from vlm_pipeline.common.runinfo import RunManifest, read_manifest, read_small_json

ATTRIBUTION_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,119}$")  # a run dir name, never a path (".", "..", "/")
LOCAL_DATA_LABEL = "local data (your own folder)"
_GRAPH_RUN = re.compile(r"^g-\d{8}-\d{6}-[0-9a-f]{6}$")  # executor / app graph run ids


class Component(BaseModel):
    """One thing the exported model is built on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["base_model", "model_part", "training_data"]  # model_part: e.g. Parse's CC-BY tokenizer
    name: str
    repo: str | None  # None: user data without provenance (TD-2)
    revision: str | None
    license: str | None  # catalog licence id; None: not recorded
    commercial: Literal["yes", "no", "conditional"] | None = None
    attribution: str = ""  # catalog label (UI); the legal notice comes from the vendored licence + catalog notice
    # Required notice sentences from the catalog (research-written, security-approved), placeholders filled by
    # fill_notices(): {repo} {revision} {date} {tool}. Empty until the catalog carries them.
    notices: tuple[str, ...] = ()


class Modification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    task: str  # lora | sft | ... ; fp8 | nvfp4 | ... for a quantization run
    dataset: str | None
    ended_at: float | None
    # A quantization run's parity check (ptq/report.json): "warn" = handed over with QNT-002, quality not measured.
    quant_check: Literal["pass", "warn"] | None = None


class Attribution(BaseModel):
    """``ATTRIBUTION.json`` (schema_version 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1  # ATTRIBUTION_SCHEMA_VERSION
    exported_at: float
    components: tuple[Component, ...]
    modifications: tuple[Modification, ...]
    gaps: tuple[str, ...] = ()  # lineage that could not be established; listed, never guessed
    files: dict[str, str] = Field(default_factory=dict)  # relative path -> sha256


class LicenseTextMissingError(RuntimeError):
    """A component's licence has no locked verbatim text: the export must not be written (fail closed)."""

    def __init__(self, licenses: Iterable[str]) -> None:
        self.licenses = sorted(set(licenses))
        super().__init__(f"no locked licence text for: {', '.join(self.licenses)}")


# ------------------------------------------------------------------------------------------------ lineage


def _run_dir_of(artifact: Path) -> Path | None:
    """The run dir (has run.json) that contains ``artifact``."""
    for parent in (artifact, *artifact.parents):
        if (parent / "run.json").is_file():
            return parent
    return None


def _repo_rev(ref: str) -> tuple[str, str | None]:
    """``org/name@rev#limit=N`` -> (``org/name``, ``rev``)."""
    ref = ref.split("#", 1)[0]
    repo, _, rev = ref.partition("@")
    return repo, rev or None


def _model_components(entry: CatalogEntry) -> list[Component]:
    """The base model + parts the catalog licenses separately (``model_parts``, e.g. a CC-BY tokenizer)."""
    base = Component(
        role="base_model", name=entry.display_name, repo=entry.repo, revision=entry.revision, license=entry.license,
        commercial=entry.commercial, attribution=entry.attribution, notices=entry.notices,
    )  # fmt: skip
    parts = [
        Component(
            role="model_part",
            name=str(part.name),
            repo=entry.repo,
            revision=entry.revision,
            license=str(part.license),
            notices=part.notices,
        )
        for part in entry.model_parts
    ]
    return [base, *parts]


def _is_local_path(ref: str) -> bool:
    """Not ``org/name[@rev][#...]``: an absolute/relative path of user data."""
    name = ref.split("#", 1)[0].split("@", 1)[0]
    return name.startswith(("/", "~", ".")) or name.count("/") != 1


def public_dataset_ref(ref: str | None) -> str | None:
    """What may appear in a shared export: hub refs as-is, local paths reduced to their last name (privacy)."""
    if ref is not None and ref.startswith("local:"):
        return LOCAL_DATA_LABEL
    if ref is None or not _is_local_path(ref):
        return ref
    return f"local data '{Path(ref.split('#', 1)[0]).name}' (path not recorded)"


def _dataset_component(ref: str, datasets: DatasetCatalog) -> Component:
    if ref.startswith("local:"):  # DatasetRef of a user folder: content hash only, no path (privacy)
        return Component(role="training_data", name=LOCAL_DATA_LABEL, repo=None, revision=None,
                         license=None)  # fmt: skip
    if _is_local_path(ref):
        return Component(role="training_data", name=public_dataset_ref(ref) or "local data", repo=None, revision=None,
                         license=None)  # fmt: skip
    repo, rev = _repo_rev(ref)
    entry: DatasetEntry | None = next((d for d in datasets.datasets if d.repo == repo), None)
    if entry is None:
        return Component(role="training_data", name=repo or ref, repo=repo or None, revision=rev, license=None)
    return Component(
        role="training_data", name=entry.display_name, repo=entry.repo, revision=rev or entry.revision,
        license=entry.license, commercial=entry.commercial, attribution=entry.attribution,
        notices=entry.notices,
    )  # fmt: skip


def lineage(
    artifact: Path, *, root: Path, catalog: Catalog, datasets: DatasetCatalog, base_ref: str | None = None
) -> tuple[list[Component], list[Modification], list[str]]:
    """(components, modifications, gaps) behind a model artifact.

    ``base_ref`` (``repo@revision``) covers an untrained catalog model, which has no run dir. Trained artifacts are
    followed through ``run.json`` -> ``parent_run_id`` (runs under ``outputs/runs``) until a run names its base model.
    """
    components: list[Component] = []
    modifications: list[Modification] = []
    gaps: list[str] = []
    seen: set[str] = set()
    run_dir = _run_dir_of(artifact)
    base = base_ref
    while run_dir is not None:
        manifest: RunManifest = read_manifest(run_dir)
        if manifest.run_id in seen:
            gaps.append(f"run {manifest.run_id}: parent chain loops")
            break
        seen.add(manifest.run_id)
        modifications.append(
            Modification(
                run_id=manifest.run_id,
                task=manifest.task,
                dataset=public_dataset_ref(manifest.dataset),
                ended_at=manifest.ended_at,
                quant_check=_quant_check(run_dir) if manifest.stage == "quantize" else None,
            )
        )
        if manifest.dataset:
            components.append(_dataset_component(manifest.dataset, datasets))
        if manifest.base_model:
            base = f"{manifest.base_model}@{manifest.base_model_revision or ''}"
        parent_id = manifest.parent_run_id
        if parent_id is None or _GRAPH_RUN.match(parent_id):  # a graph run groups nodes; it is not a training step
            break
        if not _RUN_ID.match(parent_id) or parent_id in {".", ".."}:  # run.json is data: never a path
            gaps.append(f"run {manifest.run_id}: invalid parent run id")
            break
        candidates = (run_dir.parent / parent_id, root / "outputs" / "runs" / parent_id)  # node run dirs: siblings
        parent = next((c for c in candidates if (c / "run.json").is_file()), None)
        if parent is None:
            gaps.append(f"run {manifest.run_id}: parent run {parent_id} not found")
            break
        run_dir = parent
    if base is None:
        gaps.append(f"{artifact}: no run manifest and no base model reference")
    else:
        repo, rev = _repo_rev(base)
        entry = next((m for m in catalog.models if m.repo == repo and m.revision == rev), None)
        if rev is None:
            gaps.append(f"base model {repo}: revision not recorded")
        elif entry is None:
            gaps.append(f"base model {base} is not a catalog model")
        else:
            components[:0] = _model_components(entry)
    unique = list(dict.fromkeys(components))  # same dataset twice (resumed / chained runs) -> listed once
    gaps += [f"training data {c.name}: licence not recorded" for c in unique if c.license is None]
    return unique, modifications, gaps


# ------------------------------------------------------------------------------------------------ licence texts


def missing_license_texts(root: Path, components: Iterable[Component]) -> list[str]:
    """Licence ids (of components that have one) without a locked verbatim text. Non-empty -> refuse the export."""
    return sorted({c.license for c in components if c.license and license_text(root, c.license) is None})


# ------------------------------------------------------------------------------------------------ writing


def fill_notices(components: list[Component], *, tool: str, date: str) -> list[Component]:
    """Fill ``{repo} {revision} {date} {tool}`` in catalog notices; any other brace text is left exactly as written."""

    def fill(c: Component, text: str) -> str:
        values = {"{repo}": c.repo or "", "{revision}": c.revision or "", "{date}": date, "{tool}": tool}
        for token, value in values.items():
            text = text.replace(token, value)
        return text

    return [c.model_copy(update={"notices": tuple(fill(c, n) for n in c.notices)}) for c in components]


def _quant_check(run_dir: Path) -> Literal["pass", "warn"] | None:
    """The parity verdict a quantization job wrote (job output: read no-follow, bounded); None if not readable."""
    try:
        report = read_small_json(run_dir / "ptq" / "report.json")
    except (OSError, ValueError):
        return None
    parity = report.get("parity") if isinstance(report, dict) else None
    verdict = parity.get("verdict") if isinstance(parity, dict) else None
    return verdict if verdict in ("pass", "warn") else None


QUANT_WARN_NOTE = (
    "quantization check: its outputs differ from the original model more than the warning limit, and accuracy "
    "after quantization was not measured; evaluate it on your own data before use"
)


def _notice(components: list[Component], modifications: list[Modification], gaps: list[str]) -> str:
    out = ["This model folder contains a model derived from the components below.", ""]
    for c in components:
        out += [
            f"* {c.name} ({c.role.replace('_', ' ')})",
            f"  Source: {c.repo or 'not recorded'}" + (f" @ {c.revision}" if c.revision else ""),
            f"  Licence: {c.license or 'not recorded - add the terms of this data before sharing'}"
            + (f" (see {'LICENSE' if c.role == 'base_model' else f'LICENSES/{c.license}.txt'})" if c.license else ""),
        ]
        out += [f"  Notice: {n}" for n in c.notices]
        if c.attribution and not c.notices:
            out.append(f"  Attribution: {c.attribution}")
        if c.commercial == "no":
            out.append("  Commercial use: NOT permitted by this licence.")
        out.append("")
    if modifications:
        out.append("Changes made (training and quantization runs, oldest last):")
        for m in modifications:
            when = time.strftime("%Y-%m-%d", time.gmtime(m.ended_at)) if m.ended_at else "unknown date"
            out.append(f"  - {m.task} run {m.run_id} on {m.dataset or 'unrecorded data'} ({when})")
            if m.quant_check == "warn":
                out.append(f"    Warning - {QUANT_WARN_NOTE}.")
        out.append("")
    if gaps:
        out.append("Provenance that could not be established:")
        out += [f"  - {g}" for g in gaps]
        out.append("")
    return "\n".join(out)


def _model_card(components: list[Component], modifications: list[Modification], base: CatalogEntry | None) -> str:
    model = next((c for c in components if c.role == "base_model"), None)
    data = [c for c in components if c.role == "training_data"]
    out = [f"# Fine-tuned {model.name if model else 'model'}", ""]
    out += [f"- Base model: {model.repo}@{model.revision}" if model else "- Base model: not recorded"]
    out += [f"- Training data: {c.name} ({c.repo or 'not recorded'})" for c in data] or ["- Training data: none"]
    out += [f"- Licences: {', '.join(sorted({c.license for c in components if c.license})) or 'not recorded'}"]
    out += ["- Licence texts and required notices: see LICENSE, LICENSES/ and NOTICE.", ""]
    out += ["## Usage", *_usage(base), "", "## Known limitations"]
    out += [f"- {line}" for line in _limitations(base, modifications)]
    return "\n".join(out) + "\n"


def _usage(base: CatalogEntry | None) -> list[str]:
    """Load snippet: ``trust_remote_code`` only for models that need it, with a warning (never a default to copy)."""
    if base is None:
        return ["Base model unknown: see the base model's own documentation."]
    if base.auto_class != "AutoModel":  # catalog: AutoModel <=> pinned remote code
        return [
            "```python",
            f"from transformers import {base.auto_class}, AutoProcessor",
            f'model = {base.auto_class}.from_pretrained(".")',
            'processor = AutoProcessor.from_pretrained(".")',
            "```",
        ]
    return [
        "This model runs Python code shipped in this folder. Only load this folder from a source you trust.",
        "```python",
        f"from transformers import {base.auto_class}, AutoProcessor",
        f'model = {base.auto_class}.from_pretrained(".", trust_remote_code=True)',
        'processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)',
        "```",
    ]


def _limitations(base: CatalogEntry | None, modifications: list[Modification]) -> list[str]:
    if base is None:
        return ["Base model unknown: limitations not recorded."]
    lines = []
    if base.commercial != "yes":
        lines.append(f"Commercial use: {base.commercial} under the base model licence.")
    if base.task_prompt_fixed:
        lines.append("Fixed task prompt: the model was trained for one task, not free-form questions.")
    if base.fixed_image_hw:
        lines.append(f"Input images are resized to {base.fixed_image_hw[0]}x{base.fixed_image_hw[1]}.")
    if not base.video:
        lines.append("Images only (no video).")
    if base.quant_fp8 != "official":
        lines.append(f"FP8 quantization: {base.quant_fp8}.")
    if base.remote_code != "none":
        lines.append("Loading runs model code from this folder (trust_remote_code).")
    if any(m.quant_check == "warn" for m in modifications):
        lines.append(f"Quantized model, {QUANT_WARN_NOTE}.")
    if modifications:
        lines.append("Quality depends on the fine-tuning data; evaluate on your own documents before use.")
    return lines


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


UPSTREAM_FILES = ("LICENSE", "NOTICE")  # the licensor's own copies in the base snapshot (Apache §4 d, OpenMDW)


def write_notices(
    dest: Path, *, root: Path, components: list[Component], modifications: list[Modification], gaps: list[str],
    catalog: Catalog, upstream_dir: Path | None = None,
) -> Attribution:  # fmt: skip
    """Write LICENSE, LICENSES/, NOTICE, README.md, ATTRIBUTION.json into ``dest`` (weights already there).

    ``upstream_dir`` = the base model snapshot: its own LICENSE / NOTICE are kept as ``<name>.upstream``.
    """
    if missing := missing_license_texts(root, components):
        raise LicenseTextMissingError(missing)
    base = next((c for c in components if c.role == "base_model"), None)
    base_entry = next((m for m in catalog.models if base and (m.repo, m.revision) == (base.repo, base.revision)), None)
    texts = {lic: license_text(root, lic) for lic in {c.license for c in components if c.license}}
    if base and base.license and (base_text := texts[base.license]) is not None:
        (dest / "LICENSE").write_bytes(base_text.read_bytes())
    others = sorted({c.license for c in components if c.license and (base is None or c.license != base.license)})
    if others:
        (dest / "LICENSES").mkdir(exist_ok=True)
        for lic in others:
            if (text := texts[lic]) is not None:
                (dest / "LICENSES" / f"{lic}.txt").write_bytes(text.read_bytes())
    for name in UPSTREAM_FILES:
        if upstream_dir is not None and (upstream := upstream_dir / name).is_file():  # only files the snapshot has
            (dest / f"{name}.upstream").write_bytes(upstream.read_bytes())
    (dest / "NOTICE").write_text(_notice(components, modifications, gaps), encoding="utf-8")
    (dest / "README.md").write_text(_model_card(components, modifications, base_entry), encoding="utf-8")
    files = {
        p.relative_to(dest).as_posix(): _sha256(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "ATTRIBUTION.json"
    }
    attribution = Attribution(
        exported_at=time.time(), components=tuple(components), modifications=tuple(modifications), gaps=tuple(gaps),
        files=files,
    )  # fmt: skip
    (dest / "ATTRIBUTION.json").write_text(
        json.dumps(attribution.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return attribution


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "Attribution",
    "Component",
    "LicenseTextMissingError",
    "Modification",
    "fill_notices",
    "lineage",
    "missing_license_texts",
    "public_dataset_ref",
    "write_notices",
]
