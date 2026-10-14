"""DPO for VLMs on NeMo AutoModel's VLM recipe (ADR-022 #7) — runs INSIDE the training image.

``python -m vlm_pipeline.train.dpo_vlm --config <recipe.yaml> [dotted.key=value ...]``

Neither NeMo-RL v0.7.0 (DPO is text-only) nor AutoModel 26.08 has VLM preference training, so this subclasses
``FinetuneRecipeForVLM`` and replaces only the per-microbatch loss: everything else (model build, LoRA, freezing,
FSDP2, checkpoints, logging, validation cadence) is AutoModel's. Data = the community preference layout of RLAIF-V /
MMPR: one row per image with ``question``, ``chosen`` and ``rejected`` answers. The collate renders the chosen
conversations then the rejected ones in ONE padded batch (rows ``[:n]`` and ``[n:]``): both see the same image tokens.

Loss (Rafailov et al. 2023, arXiv 2305.18290), per pair, on summed answer-token log-probs:
``-logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`` + ``sft_weight * CE(chosen)``
(DPO+SFT; ``sft_weight`` 0 = plain DPO).
The reference is the SAME model with every LoRA delta switched off (``LinearLoRA.scale = 0``) under ``no_grad``: no
second model in memory. That needs LoRA (``peft:`` in the recipe); a full fine-tune would need a frozen copy and is
refused. Not MPO (InternVL): MPO adds a quality (BCO) term.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator, Sequence
from typing import Any

PAIR_KEY = "dpo_num_pairs"
LOGP_CHUNK = 256
DEFAULT_COLUMNS = {"image": "image", "question": "question", "chosen": "chosen", "rejected": "rejected"}
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ pure parts (tested)
def conversation(image: Any, question: str, answer: str) -> list[dict[str, Any]]:
    """The chat layout AutoModel's VLM collates expect (as make_cord_v2_dataset builds it)."""
    return [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def dpo_terms(
    policy_chosen: Any, policy_rejected: Any, ref_chosen: Any, ref_rejected: Any, beta: float
) -> tuple[Any, Any, Any]:
    """Per-pair DPO loss, implicit reward margin and "chosen wins" (tensors of shape [n])."""
    from torch.nn.functional import logsigmoid

    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -logsigmoid(margin), margin, (margin > 0).float()


def _chunk_token_logps(logits: Any, labels: Any) -> Any:
    part = logits.float()
    return part.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - part.logsumexp(dim=-1)


def sequence_logps(logits: Any, labels: Any, chunk: int = LOGP_CHUNK) -> tuple[Any, Any]:
    """Summed log-prob of the label tokens per row, and the label-token count per row (labels already aligned).

    Chunked over the sequence so only [2n, chunk, vocab] is upcast to fp32 at a time (Qwen3.5 vocab ~248k: a full fp32
    log_softmax is ~2 GB per 2k-token row on the unified memory). With grad, each chunk is checkpointed: backward
    recomputes the upcast from the bf16 logits instead of keeping every fp32 chunk alive (bao-backend review).
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    mask = labels != -100
    safe = labels.masked_fill(~mask, 0)
    total = torch.zeros(labels.shape[0], dtype=torch.float32, device=labels.device)
    for s in range(0, labels.shape[1], chunk):
        piece, ids = logits[:, s : s + chunk], safe[:, s : s + chunk]
        if torch.is_grad_enabled() and piece.requires_grad:
            tok = checkpoint(_chunk_token_logps, piece, ids, use_reentrant=False)
        else:
            tok = _chunk_token_logps(piece, ids)
        total = total + (tok * mask[:, s : s + chunk]).sum(-1)
    return total, mask.sum(-1)


# ------------------------------------------------------------------------------------------------ data
def make_preference_dataset(
    path_or_dataset: str,
    split: str = "train",
    limit_dataset_samples: int | None = None,
    columns: dict[str, str] | None = None,
    **load_kwargs: Any,
) -> list[dict[str, Any]]:
    """Rows ``{chosen: conversation, rejected: conversation}`` from ``<dir>/data/<split>-*.parquet`` (image column).

    Rows whose chosen and rejected answers are equal teach nothing and are dropped.
    """
    from pathlib import Path

    from datasets import load_dataset

    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    files = sorted(str(f) for f in Path(path_or_dataset).glob(f"data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data/{split}-*.parquet under {path_or_dataset}")
    sliced = f"{split}[:{limit_dataset_samples}]" if limit_dataset_samples is not None else split
    rows = []
    # the parquet builder, never the directory: a dataset folder's own *.py loading script must not run (ADR-021 §4)
    for ex in load_dataset("parquet", data_files={split: files}, split=sliced, **load_kwargs):
        chosen, rejected = str(ex[cols["chosen"]]), str(ex[cols["rejected"]])
        if chosen.strip() == rejected.strip():
            continue
        image, question = ex[cols["image"]], str(ex[cols["question"]])
        rows.append(
            {"chosen": conversation(image, question, chosen), "rejected": conversation(image, question, rejected)}
        )
    if not rows:
        raise ValueError("no preference pair left (chosen == rejected everywhere, or an empty split)")
    return rows


def dpo_collate_fn(
    examples: Sequence[dict[str, Any]], processor: Any, base: str = "default_collate_fn"
) -> dict[str, Any]:
    """AutoModel's collate on [chosen..., rejected...]; records the pair count for the loss."""
    from nemo_automodel.components.datasets.vlm import collate_fns

    fn = collate_fns.COLLATE_FNS.get(base) or getattr(collate_fns, base)
    rows = [{"conversation": e["chosen"]} for e in examples] + [{"conversation": e["rejected"]} for e in examples]
    batch: dict[str, Any] = fn(rows, processor)
    batch[PAIR_KEY] = len(examples)
    return batch


# ------------------------------------------------------------------------------------------------ recipe
@contextlib.contextmanager
def lora_disabled(model: Any) -> Iterator[None]:
    """The base model: every AutoModel LinearLoRA delta scaled by 0 for the duration (restored after)."""
    from nemo_automodel.components._peft.lora import LinearLoRA

    saved = [(m, m.scale) for m in model.modules() if isinstance(m, LinearLoRA)]
    if not saved:
        raise RuntimeError("DPO needs LoRA (peft:): the reference is the base model with the adapter switched off")
    try:
        for m, _ in saved:
            m.scale = 0.0
        yield
    finally:
        for m, scale in saved:
            m.scale = scale


def _recipe_class() -> type:
    from contextlib import nullcontext

    import torch
    from nemo_automodel.components.distributed.utils import get_sync_ctx
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM, _move_to_device

    class DPORecipeForVLM(FinetuneRecipeForVLM):  # type: ignore[misc]
        """FinetuneRecipeForVLM with the DPO loss (no pipeline / context parallel: one GB10)."""

        def setup(self) -> None:
            super().setup()
            dpo = self.cfg.get("dpo", None)
            self.dpo_beta = float(dpo.get("beta", 0.1)) if dpo is not None else 0.1
            self.dpo_sft_weight = float(dpo.get("sft_weight", 0.0)) if dpo is not None else 0.0
            if self.pp_enabled or self.mesh_context.cp_size > 1:
                raise NotImplementedError("DPO runs without pipeline or context parallelism")
            if self.peft_config is None:
                raise ValueError("DPO needs LoRA (peft:): the reference is the base model with the adapter off")

        def _forward_backward_step(self, idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True):  # type: ignore[no-untyped-def]
            n = int(batch.pop(PAIR_KEY))
            batch = {k: _move_to_device(v, self.dist_env.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            model = self.model_parts[0]
            sync = get_sync_ctx(model, idx == num_batches - 1) if is_train else nullcontext()
            with sync:
                logits = model(**batch).logits
                logps, counts = sequence_logps(logits, labels)
                was_training = model.training
                model.eval()  # reference pass: no dropout (Qwen has none, other families may)
                try:
                    with torch.no_grad(), lora_disabled(model):
                        ref_logps, _ = sequence_logps(model(**batch).logits, labels)
                finally:
                    model.train(was_training)
                loss, margin, wins = dpo_terms(logps[:n], logps[n:], ref_logps[:n], ref_logps[n:], self.dpo_beta)
                loss = loss.mean()
                if self.dpo_sft_weight > 0:
                    loss = loss - self.dpo_sft_weight * (logps[:n].sum() / counts[:n].sum().clamp(min=1))
                loss = loss / num_batches  # mean over the gradient-accumulation microbatches
                loss_buffer.append(loss.clone().detach())
                if idx == 0 and math.isfinite(float(margin.mean())):
                    logger.info(
                        "dpo microbatch: margin %.4f, chosen-wins %.2f, policy-ref chosen %.3f rejected %.3f",
                        float(margin.mean()), float(wins.mean()),
                        float((logps[:n] - ref_logps[:n]).mean()), float((logps[n:] - ref_logps[n:]).mean()),
                    )  # fmt: skip
                if is_train:
                    loss.backward()

    return DPORecipeForVLM


def main(argv: list[str] | None = None) -> int:
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()  # before any checkpoint tensor is on the CPU (GB10 context creation OOM)
    cfg = parse_args_and_load_config(None, argv=argv)
    trainer = _recipe_class()(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
