"""Schema of ``configs/reference_pipelines.yaml`` v2: the gallery organised by training METHOD (ADR-022, P023 §5).

One record per method (10 in 1.0). ``train_arch`` drives the card's architecture diagram (design spec I): each block's
state, the quantization apart from the state (QLoRA = ``lm: lora`` + ``quant: nf4``; QAT adds ``fake_quant``), side
blocks (teacher, reference, rollout, reward) and optional stages (#4 two-stage; the top-level blocks equal the last
stage). Every number in ``measured`` cites committed ``sources`` (the site gate checks them). Strict: an unknown key
or value fails loading, so a typo never renders as an empty card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REFERENCE_PIPELINES = Path("configs") / "reference_pipelines.yaml"

Method = Literal["full_sft", "lora", "qlora", "two_stage", "vision_unfrozen", "distill", "dpo", "grpo", "ptq", "qat"]
BlockState = Literal["frozen", "lora", "full", "full_low_lr", "n/a"]
Quant = Literal["nf4", "fp8", "nvfp4"]
Extra = Literal["teacher", "reference", "rollout", "reward"]
_KEY = r"^reference_pipeline\.[a-z0-9_-]+\.[a-z0-9_]+$"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Stage(_Strict):
    name_key: str = Field(pattern=_KEY)
    vision: BlockState
    projector: BlockState
    lm: BlockState
    quant: Quant | None = None
    fake_quant: bool = False  # QAT: trains with quantization simulated, ships the quantized model


class TrainArch(_Strict):
    vision: BlockState
    projector: BlockState
    lm: BlockState
    quant: Quant | None = None
    fake_quant: bool = False
    extras: tuple[Extra, ...] = ()
    stages: tuple[Stage, ...] = ()

    @model_validator(mode="after")
    def _single_view_is_the_last_stage(self) -> Self:
        """With stages, the one-block view of the card shows the final state: it must equal the last stage."""
        if self.stages:
            last = self.stages[-1]
            final = (last.vision, last.projector, last.lm, last.quant, last.fake_quant)
            if (self.vision, self.projector, self.lm, self.quant, self.fake_quant) != final:
                raise ValueError("train_arch blocks must equal its last stage")
        if self.fake_quant and self.quant is None:
            raise ValueError("fake_quant needs the quant format it simulates")
        return self


class Source(_Strict):
    url: str = Field(pattern=r"^https://")
    publisher: str = Field(min_length=1)
    paper_url: str | None = Field(default=None, pattern=r"^https://")


class MeasuredWith(_Strict):
    model: str = Field(min_length=1)
    data: str = Field(min_length=1)


class Score(_Strict):
    benchmark: str
    metric: str
    value: float
    ci95: tuple[float, float] | None = None
    n: int = Field(gt=0)
    compare_with: str | None = None


class Measured(_Strict):
    train_minutes: float | None = Field(default=None, gt=0)
    peak_ram_gb: float | None = Field(default=None, gt=0)
    score: Score | None = None
    config: str = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)  # committed repo paths (check_site_numbers)


class Step(_Strict):
    node: str | None  # an app node type, or null when the step has no node yet
    label: dict[Literal["vi", "en"], str]


class Pipeline(_Strict):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    method: Method
    title: dict[Literal["vi", "en"], str]
    summary_key: str = Field(pattern=_KEY)
    when_to_use_key: str = Field(pattern=_KEY)
    kind: Literal["app_template", "candidate", "external"]
    template_id: str | None
    status: Literal["available", "planned", "reference_only"]
    difficulty: Literal["basic", "advanced"] | None
    train_arch: TrainArch
    source: Source
    license: dict[str, str | None]
    measured: Measured | None = None
    measured_with: MeasuredWith | None = None
    steps: tuple[Step, ...] = Field(min_length=1)
    notice: str | None = None
    model_parts: tuple[dict[str, str], ...] = ()
    requires_hf_terms: bool = False

    @model_validator(mode="after")
    def _keys_name_this_pipeline(self) -> Self:
        for key in (self.summary_key, self.when_to_use_key):
            if key.split(".")[1] != self.id:
                raise ValueError(f"{key}: i18n keys are reference_pipeline.{self.id}.*")
        if self.measured is not None and self.measured_with is None:
            raise ValueError("measured numbers need measured_with (the card says which model and data)")
        return self


class ReferencePipelines(_Strict):
    schema_version: Literal[2]
    pipelines: tuple[Pipeline, ...]

    @model_validator(mode="after")
    def _one_per_method_and_comparable(self) -> Self:
        ids = [p.id for p in self.pipelines]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate pipeline id")
        methods = [p.method for p in self.pipelines]
        if dupes := sorted({m for m in methods if methods.count(m) > 1}):
            raise ValueError(f"one pipeline per method (ADR-022): {dupes}")
        by_id = {p.id: p for p in self.pipelines}
        for p in self.pipelines:
            score = p.measured.score if p.measured else None
            if score is None or score.compare_with is None:
                continue
            other = by_id.get(score.compare_with)
            other_score = other.measured.score if other and other.measured else None
            if other_score is None or (other_score.benchmark, other_score.metric) != (score.benchmark, score.metric):
                raise ValueError(
                    f"{p.id}: compare_with {score.compare_with!r} has no score on the same benchmark/metric"
                )
        return self


def load_reference_pipelines(path: Path) -> ReferencePipelines:
    return ReferencePipelines.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = ["REFERENCE_PIPELINES", "Method", "Pipeline", "ReferencePipelines", "TrainArch", "load_reference_pipelines"]
