"""train.dpo_vlm (ADR-022 #7): preference training spec, refused with its real reason until the VLM DPO patch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_pipeline.train.config import DpoParams


def test_defaults_are_plain_dpo() -> None:
    p = DpoParams()
    assert (p.beta, p.sft_weight, p.epochs) == (0.1, 0.0, 1)


def test_beta_is_bounded() -> None:
    with pytest.raises(ValidationError):
        DpoParams(beta=0.0)


def test_dpo_is_refused_with_its_reason_on_canvas_and_run() -> None:
    from types import SimpleNamespace

    from vlm_pipeline.app.workers import DPO_BLOCKED_REASON, NodeNotRunnableError, _support_checks

    query = SimpleNamespace(
        node_id="d", spec=SimpleNamespace(type_id="train.dpo_vlm"), params=None, model=None, inputs={}
    )
    with pytest.raises(NodeNotRunnableError) as refused:
        _support_checks()["train.dpo_vlm"](query)  # type: ignore[arg-type]
    assert refused.value.params["reason"] == DPO_BLOCKED_REASON
