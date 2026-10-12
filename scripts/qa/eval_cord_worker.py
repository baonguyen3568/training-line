"""CORD-v2 generation for Nemotron-Parse (P-2), run INSIDE a vlm-automodel image; scoring happens on the host.

The decoder prefix is taken from the training collate itself (``nemotron_parse_collate_fn``): the decoder context
up to and including the position that predicts the first labelled token, so eval sees exactly what training saw
(task tokens included). generate()'s start token is pinned to that prefix, otherwise HF prepends the model's
``decoder_start_token_id`` (2 for Parse) and silently shifts the context; an assert checks the result.
Greedy decoding; a generation that hits ``max_new_tokens`` is flagged ``truncated`` and still scored.

Usage (repo mounted at /workspace):
  python /workspace/scripts/qa/eval_cord_worker.py --model <prepared dir> --data <cord-v2 snapshot dir> --split test \
      [--ids 0,1,2 | --limit N] --out <predictions.jsonl> [--device cuda] [--max-new-tokens 1536]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _conversation(example: dict[str, Any], json2token: Any) -> dict[str, Any]:
    gt = json.loads(example["ground_truth"])["gt_parse"]
    target = json2token(gt, sort_json_key=True)
    return {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": example["image"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
        "gt_parse": gt,
    }


def train_decoder_prefix(conversation: dict[str, Any], processor: Any) -> Any:
    """Decoder ids that precede the first labelled token, exactly as the training collate builds them."""
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn

    batch = nemotron_parse_collate_fn([{"conversation": conversation}], processor)
    labelled = (batch["labels"][0] != -100).nonzero()
    if labelled.numel() == 0:
        raise ValueError("sample has no labelled token (all labels are -100)")
    first = int(labelled[0])
    return batch["decoder_input_ids"][0, : first + 1], batch


def strip_trailing_pad(ids: list[int], pad_id: int | None) -> tuple[list[int], int]:
    """Drop trailing pad ids; returns the kept ids and how many pads were dropped.

    Parse-2.0's tokenizer.json pads to 9000 with pad_id 1: a pad tail must never turn a good answer into a parse
    failure.
    """
    end = len(ids)
    while pad_id is not None and end > 0 and ids[end - 1] == pad_id:
        end -= 1
    return ids[:end], len(ids) - end


def _local_dir(path: str, what: str) -> Path:
    """Only pinned local snapshots (prepared + hash-verified, ADR-003): never a hub id resolved to `main`."""
    p = Path(path)
    if not p.is_dir():
        raise SystemExit(f"--{what} must be a local snapshot directory (pinned), got {path!r}")
    return p.resolve()


def main() -> int:
    import torch
    from datasets import load_dataset
    from nemo_automodel.components.datasets.vlm.utils import json2token
    from transformers import AutoModel, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument(
        "--expect-prefix",
        default="",
        help="comma-separated decoder ids the training prefix must equal, e.g. 2,0,50004,50008,50001 (001a §10.5)",
    )
    args = ap.parse_args()
    expected_prefix = [int(x) for x in args.expect_prefix.split(",") if x]
    model_dir, data_dir = _local_dir(args.model, "model"), _local_dir(args.data, "data")

    ds = load_dataset(str(data_dir), split=args.split)
    idx = [int(i) for i in args.ids.split(",") if i] or list(range(args.limit or len(ds)))
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(args.device).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in idx:
            item = _conversation(ds[i], json2token)
            try:
                prefix, batch = train_decoder_prefix(item["conversation"], processor)
            except ValueError as exc:
                raise SystemExit(f"[{i}] {exc}") from exc
            if expected_prefix and prefix.tolist() != expected_prefix:
                raise SystemExit(f"[{i}] training prefix {prefix.tolist()} != expected {expected_prefix}")
            # HF generate() prepends generation_config.decoder_start_token_id (2 for Parse) unless the prefix already
            # starts with it; training's collate starts with <s> (0). Pin the start token to the training prefix so the
            # eval context is identical to training (verified by the assert below).
            model.generation_config.decoder_start_token_id = int(prefix[0])
            with torch.no_grad():
                gen = model.generate(
                    pixel_values=batch["pixel_values"].to(args.device),
                    decoder_input_ids=prefix.unsqueeze(0).to(args.device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            if gen[0, : prefix.shape[0]].tolist() != prefix.tolist():
                raise SystemExit(f"[{i}] eval prefix differs from training prefix: {gen[0, :12].tolist()}")
            new = gen[0, prefix.shape[0] :].tolist()
            new, n_pad = strip_trailing_pad(new, tokenizer.pad_token_id)
            text = tokenizer.decode(new, skip_special_tokens=False)
            for special in (tokenizer.eos_token, tokenizer.pad_token):
                if special:
                    text = text.replace(special, "")
            record = {
                "index": i,
                "sequence": text.strip(),
                "gt_parse": item["gt_parse"],
                "decoder_prefix_ids": prefix.tolist(),
                "model_dir": str(model_dir),
                "data_dir": str(data_dir),
                "split": args.split,
                "n_new_tokens": len(new),
                "n_trailing_pad_stripped": n_pad,
                "truncated": len(new) >= args.max_new_tokens,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}] new_tokens={len(new)} truncated={record['truncated']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
