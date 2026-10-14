"""train.grpo_vlm params (P020 v3): only the rewards NeMo-RL's VLMEnvironment ships; never user code."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_pipeline.train.config import GrpoParams


def test_defaults_follow_the_engine_example() -> None:
    p = GrpoParams()
    assert (p.reward, p.num_generations, p.format_weight, p.correctness_weight) == ("exact_match", 4, 0.1, 0.9)


@pytest.mark.parametrize("bad", [{"reward": "python"}, {"reward": "json_field_f1"}, {"reward_fn": "lambda a, b: 1"}])
def test_no_reward_outside_the_engine_can_be_chosen(bad: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        GrpoParams.model_validate(bad)
