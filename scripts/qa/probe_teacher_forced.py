"""Teacher-forced loss / token accuracy of a Parse checkpoint through the HF-remote eval path.

This is the path CORD eval uses; compared with the training val_loss it separates "eval path broken" from "model
loops when free-running" (P-3, 2026-10-12: 0.238 / 94% vs base 2.48 / 58%).

  python /workspace/scripts/qa/probe_teacher_forced.py --model <dir> --data <cord dir> --split validation \
      [--indices 0,1,2,3,4] [--dtype bfloat16] [--device cpu]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    import torch
    from datasets import load_dataset
    from parse_native_remote_parity import _training_batch  # same collate as training
    from transformers import AutoModel, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--indices", default="0,1,2,3,4")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for path, what in ((args.model, "model"), (args.data, "data")):
        if not Path(path).is_dir():
            raise SystemExit(f"--{what} must be a local snapshot directory (pinned), got {path!r}")
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(args.device).eval()
    ds = load_dataset(args.data, split=args.split)
    total_nll, total_tok, total_ok, rows = 0.0, 0, 0, []
    for i in [int(x) for x in args.indices.split(",")]:
        batch = _training_batch(ds[i], processor)
        with torch.no_grad():
            out = model(
                pixel_values=batch["pixel_values"].to(args.device, dtype),
                decoder_input_ids=batch["decoder_input_ids"].to(args.device),
                decoder_attention_mask=batch["decoder_attention_mask"].to(args.device),
            )
        labels = batch["labels"][0].to(args.device)
        keep = labels != -100
        logits = out.logits[0][keep].float()
        nll = torch.nn.functional.cross_entropy(logits, labels[keep], reduction="sum").item()
        ok = int((logits.argmax(-1) == labels[keep]).sum())
        n = int(keep.sum())
        total_nll, total_tok, total_ok = total_nll + nll, total_tok + n, total_ok + ok
        rows.append({"index": i, "loss": round(nll / n, 4), "token_acc": round(ok / n, 4), "n": n})
    result = {
        "model": args.model,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "split": args.split,
        "dtype": args.dtype,
        "per_sample": rows,
        "loss": round(total_nll / total_tok, 4),
        "token_acc": round(total_ok / total_tok, 4),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
