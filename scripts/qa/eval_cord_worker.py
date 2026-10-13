"""CORD-v2 generation for Nemotron-Parse (P-2), run INSIDE a vlm-automodel image; scoring happens on the host.

The decoder prefix is taken from the training collate itself (``nemotron_parse_collate_fn``): the decoder context
up to and including the position that predicts the first labelled token, so eval sees exactly what training saw
(task tokens included). generate()'s start token is pinned to that prefix, otherwise HF prepends the model's
``decoder_start_token_id`` (2 for Parse) and silently shifts the context; an assert checks the result.
Decoding follows the model's own generation_config.json (Parse v1.1: greedy, repetition_penalty 1.1), read from
the required ``--generation-config`` (the pinned base snapshot): an export may carry a regenerated config without
those keys (AutoModel 26.08 before 23e6aa6); drift is recorded, and fatal with ``--require-no-drift``.
Only ``max_new_tokens``, the start token and an explicit ``--repetition-penalty`` are overridden. A generation
that hits ``max_new_tokens`` is flagged ``truncated`` and still scored. bf16 kernels depend on the batch size
(batch 3 diverged from batch 1 on 3/6 O-1b samples): official numbers use batch 1 (recorded).

Usage (repo mounted at /workspace):
  python /workspace/scripts/qa/eval_cord_worker.py --model <prepared dir> --data <cord-v2 snapshot dir> --split test \
      [--ids 0,1,2 | --limit N] --out <predictions.jsonl> [--device cuda] [--max-new-tokens 1024] \
      --generation-config <base snapshot dir> [--require-no-drift] [--batch-size 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

# Exact worker source per record: a matrix may span worker commits, and run.json's git_commit is the shared tree's.
WORKER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


def _record(i: int, item: dict[str, Any], new: list[int], *, prefix: Any, tokenizer: Any, args: Any) -> dict[str, Any]:
    """One prediction record: pad tail dropped at id level, eos/pad text removed, truncation flagged."""
    new, n_pad = strip_trailing_pad(new, tokenizer.pad_token_id)
    text = tokenizer.decode(new, skip_special_tokens=False)
    for special in (tokenizer.eos_token, tokenizer.pad_token):
        if special:
            text = text.replace(special, "")
    return {
        "index": i,
        "sequence": text.strip(),
        "gt_parse": item["gt_parse"],
        "decoder_prefix_ids": prefix.tolist(),
        "split": args.split,
        "batch_size": args.batch_size,
        "n_new_tokens": len(new),
        "n_trailing_pad_stripped": n_pad,
        "truncated": len(new) >= args.max_new_tokens,
    }


def _prepare(chunk: list[int], ds: Any, processor: Any, json2token: Any, expected: list[int]) -> tuple[Any, Any, Any]:
    """Conversations, training prefixes and pixel values of one chunk; stops on a sample the collate rejects."""
    items, prefixes, pixels = [], [], []
    for i in chunk:
        item = _conversation(ds[i], json2token)
        try:
            prefix, batch = train_decoder_prefix(item["conversation"], processor)
        except ValueError as exc:
            raise SystemExit(f"[{i}] {exc}") from exc
        if expected and prefix.tolist() != expected:
            raise SystemExit(f"[{i}] training prefix {prefix.tolist()} != expected {expected}")
        items.append(item)
        prefixes.append(prefix)
        pixels.append(batch["pixel_values"])
    return items, prefixes, pixels


# Keys that cannot change the generated tokens: cache use, bookkeeping, and extra outputs.
GENERATION_DRIFT_ALLOWED = frozenset(
    {"use_cache", "transformers_version", "_from_model_config", "output_attentions", "output_hidden_states"}
)


def generation_config_drift(checkpoint: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """Decoding keys where the checkpoint's generation config differs from the base's (allowlist excluded)."""
    keys = (checkpoint.keys() | base.keys()) - GENERATION_DRIFT_ALLOWED
    return sorted(k for k in keys if checkpoint.get(k) != base.get(k))


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _local_dir(path: str, what: str) -> Path:
    """Only pinned local snapshots (prepared + hash-verified, ADR-003): never a hub id resolved to `main`."""
    p = Path(path)
    if not p.is_dir():
        raise SystemExit(f"--{what} must be a local snapshot directory (pinned), got {path!r}")
    return p.resolve()


def _set_decoding(model: Any, model_dir: Path, args: Any) -> dict[str, Any]:
    """Load the base generation config into the model; returns the decoding parameters for the records."""
    from transformers import GenerationConfig

    gen_dir = _local_dir(args.generation_config, "generation-config")
    model.generation_config = GenerationConfig.from_pretrained(str(gen_dir))
    # A checkpoint completed by ckpt_complete (23e6aa6+) carries the base decoding keys. By default drift is only
    # recorded, so an older export (O-1b) is still evaluated with the base decoding; gate runs pass
    # --require-no-drift, where drift means the export is broken again.
    drift = generation_config_drift(
        GenerationConfig.from_pretrained(str(model_dir)).to_dict(), model.generation_config.to_dict()
    )
    if drift and args.require_no_drift:
        raise SystemExit(f"checkpoint generation_config differs from {gen_dir} on {drift}: re-export (ckpt_complete)")
    model.generation_config.max_new_tokens = args.max_new_tokens
    # Pinned, never inherited: the base config has no use_cache (HF default True) while exports carry False
    # (patch 0003). P-3 A/B (2026-10-12): no-cache degenerated like cache, so the cache was not the cause of the
    # loops; exact equivalence (golden ids through remote + cache) is still to be checked.
    model.generation_config.use_cache = args.use_cache == "true"
    overrides: dict[str, Any] = {"max_new_tokens": args.max_new_tokens, "use_cache": args.use_cache == "true"}
    if args.repetition_penalty is not None:
        # Protocol comparison (001a): the model's decoding vs train-time greedy; the override is recorded.
        model.generation_config.repetition_penalty = args.repetition_penalty
        overrides["repetition_penalty"] = args.repetition_penalty
    decoding = {
        k: getattr(model.generation_config, k, None)
        for k in (
            "do_sample",
            "num_beams",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "forced_eos_token_id",
            "use_cache",
        )
    }
    decoding["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
    decoding.update(
        {
            "max_new_tokens": args.max_new_tokens,
            "generation_config_dir": str(gen_dir),
            "checkpoint_drift": drift,
            "overrides": overrides,
        }
    )
    return decoding


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
    ap.add_argument("--max-new-tokens", type=int, default=1024)  # CORD test target max 829 tokens (+eos)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--expect-prefix",
        default="",
        help="comma-separated decoder ids the training prefix must equal, e.g. 2,0,50004,50008,50001 (001a §10.5)",
    )
    # Required: defaulting to the model dir would silently decode an old export without the base's
    # repetition_penalty (O-1b). For the base model itself, pass its own snapshot.
    ap.add_argument("--generation-config", required=True, help="pinned base snapshot dir (generation_config.json)")
    ap.add_argument("--use-cache", choices=("true", "false"), default="true", help="KV cache in generate (recorded)")
    ap.add_argument(
        "--repetition-penalty",
        type=_positive_float,  # fail at parse time, not in generate() after loading the model under the GPU lock
        default=None,
        help="override the base value (1.0 = off), recorded in each record's decoding.overrides",
    )
    ap.add_argument(
        "--require-no-drift",
        action="store_true",
        help="fail if the checkpoint's generation config differs from --generation-config (P-3, gates, gallery)",
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
    decoding = _set_decoding(model, model_dir, args)
    print("decoding", json.dumps(decoding), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk_start in range(0, len(idx), args.batch_size):
            chunk = idx[chunk_start : chunk_start + args.batch_size]
            items, prefixes, pixels = _prepare(chunk, ds, processor, json2token, expected_prefix)
            # Batching needs no decoder padding only because every sample shares the same training prefix.
            if any(p.tolist() != prefixes[0].tolist() for p in prefixes):
                raise SystemExit(f"{chunk}: samples have different prefixes, use --batch-size 1")
            prefix = prefixes[0]
            # HF generate() prepends generation_config.decoder_start_token_id unless the prefix already starts with it;
            # pin the start token to the training prefix so the eval context equals training (asserted below).
            model.generation_config.decoder_start_token_id = int(prefix[0])
            with torch.no_grad():
                gen = model.generate(
                    pixel_values=torch.cat(pixels).to(args.device),
                    decoder_input_ids=prefix.unsqueeze(0).repeat(len(chunk), 1).to(args.device),
                    generation_config=model.generation_config,
                )
            for row, (i, item) in enumerate(zip(chunk, items, strict=True)):
                if gen[row, : prefix.shape[0]].tolist() != prefix.tolist():
                    raise SystemExit(f"[{i}] eval prefix differs from training prefix: {gen[row, :12].tolist()}")
                record = _record(
                    i, item, gen[row, prefix.shape[0] :].tolist(), prefix=prefix, tokenizer=tokenizer, args=args
                )
                record.update(
                    {
                        "model_dir": str(model_dir),
                        "data_dir": str(data_dir),
                        "decoding": decoding,
                        "worker_sha256": WORKER_SHA256,
                    }
                )
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i}] new_tokens={record['n_new_tokens']} truncated={record['truncated']}", flush=True)
    return 0


if __name__ == "__main__":
    # GB10: create the CUDA context before the checkpoint is loaded on the CPU (fixcode 2026-10-13, common.cuda_init)
    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()
    raise SystemExit(main())
