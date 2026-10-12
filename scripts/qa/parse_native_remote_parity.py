"""Native-vs-remote checkpoint parity for Nemotron-Parse, run before trusting CORD F1 from the HF-remote eval path.

For a checkpoint trained with the NATIVE AutoModel class (bao-research, 001a):
(1) the remote class must load the consolidated checkpoint STRICTLY: no missing / mismatched keys (from_pretrained
    would silently random-init them). Unexpected keys are allowed only if the pinned base checkpoint (``--base``) has
    exactly the same ones: Parse's own release carries ``decoder.embed_positions.weight``, which the remote class
    has no module for (BUG-002), so ``{}`` is impossible even for NVIDIA's checkpoint;
(2) native and remote forwards on the same batches (several images, training prefix + target tokens) must agree:
    float32, same attention implementation on both sides (so only weights/architecture can differ, not kernels),
    log-softmax within atol (default 1e-3) and identical argmax. bf16 would fail spuriously: its step is 0.125 for
    logits in [16, 32).

Runs INSIDE a vlm-automodel image (repo at /workspace); prints one JSON line, exit 1 if a check fails.
  python /workspace/scripts/qa/parse_native_remote_parity.py --model <consolidated dir> --data <cord-v2 dir> \
      [--base <pinned base model dir>] [--indices 0,1,2] [--device cuda] [--atol 1e-3] [--attn eager]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _base_unexpected_keys(base_dir: str) -> list[str]:
    """Unexpected keys the remote class reports for the pinned base checkpoint itself."""
    import torch
    from transformers import AutoModel

    if not Path(base_dir).is_dir():
        raise SystemExit("--base must be a local snapshot directory (pinned)")
    base, info = AutoModel.from_pretrained(
        base_dir, trust_remote_code=True, dtype=torch.float32, output_loading_info=True
    )
    del base
    return sorted(info.get("unexpected_keys") or [])


def blocking_load_problems(
    problems: dict[str, list[str]], base_unexpected: list[str]
) -> tuple[dict[str, list[str]], bool]:
    """Load problems that fail the gate, and whether the unexpected keys were tolerated.

    Missing / mismatched keys always fail (random init); unexpected keys only when they differ from the base's own.
    """
    tolerated = bool(base_unexpected) and problems.get("unexpected_keys") == base_unexpected
    return {k: v for k, v in problems.items() if not (k == "unexpected_keys" and tolerated)}, tolerated


def _training_batch(example: dict[str, Any], processor: Any) -> Any:
    """One sample collated exactly as training does (prefix + sorted-key target tokens)."""
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn
    from nemo_automodel.components.datasets.vlm.utils import json2token

    target = json2token(json.loads(example["ground_truth"])["gt_parse"], sort_json_key=True)
    conversation = [
        {"role": "user", "content": [{"type": "image", "image": example["image"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return nemotron_parse_collate_fn([{"conversation": conversation}], processor)


def main() -> int:
    import torch
    from datasets import load_dataset
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from transformers import AutoModel, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="", help="pinned base checkpoint: its unexpected keys are tolerated")
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()
    model_dir, data_dir = Path(args.model), Path(args.data)
    if not model_dir.is_dir() or not data_dir.is_dir():
        raise SystemExit("--model and --data must be local snapshot directories (pinned)")

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    dataset = load_dataset(str(data_dir), split="train")
    indices = [int(i) for i in args.indices.split(",") if i]
    batches = [_training_batch(dataset[index], processor) for index in indices]

    remote, info = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation=args.attn,
        output_loading_info=True,
    )
    load_problems = {k: sorted(v) for k, v in info.items() if v}
    base_unexpected = _base_unexpected_keys(args.base) if args.base else []
    blocking, tolerated = blocking_load_problems(load_problems, base_unexpected)
    native = NeMoAutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        attn_implementation=args.attn,
        use_liger_kernel=False,
        trust_remote_code=True,  # config class lives in the checkpoint (copied from the vetted prepared snapshot)
    )
    classes = {
        "remote": type(remote).__module__ + "." + type(remote).__qualname__,
        "native": type(native).__module__ + "." + type(native).__qualname__,
    }

    def _buffers(model: Any) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name, buf in model.named_buffers():
            if name.endswith(("input_conditioner.norm_mean", "input_conditioner.norm_std")):
                out[name.split("input_conditioner.")[-1]] = [round(float(v), 6) for v in buf.flatten()]
        return out

    conditioner = {"remote": _buffers(remote), "native": _buffers(native)}
    logprobs: dict[str, list[Any]] = {"remote": [], "native": []}
    for name, loaded in (("remote", remote), ("native", native)):
        net = loaded.to(args.device).eval()
        for batch in batches:
            inputs = {
                k: (batch[k].float() if k == "pixel_values" else batch[k]).to(args.device)
                for k in ("pixel_values", "decoder_input_ids", "decoder_attention_mask")
            }
            with torch.no_grad():
                out = net(**inputs)
            labelled = (batch["labels"][0] != -100).to(args.device)
            logprobs[name].append(torch.log_softmax(out.logits[0][labelled].float(), dim=-1).cpu())
        net.to("cpu")

    per_sample = []
    for r, n in zip(logprobs["remote"], logprobs["native"], strict=True):
        diff = (r - n).abs()
        per_sample.append(
            {
                "max_abs_logprob_diff": float(diff.max()),
                "argmax_agreement": float((r.argmax(-1) == n.argmax(-1)).float().mean()),
                "n_target_tokens": int(r.shape[0]),
            }
        )
    result = {
        "model_dir": str(model_dir),
        "indices": indices,
        "dtype": "float32",
        "attn_implementation": args.attn,
        "classes": classes,
        "distinct_implementations": classes["remote"] != classes["native"],
        "input_conditioner": conditioner,
        "strict_load_ok": not blocking,
        "load_problems": {k: v[:10] for k, v in load_problems.items()},
        "base_unexpected_keys": base_unexpected[:10],
        "unexpected_keys_same_as_base": tolerated,
        "per_sample": per_sample,
        "max_abs_logprob_diff": max(p["max_abs_logprob_diff"] for p in per_sample),
        "min_argmax_agreement": min(p["argmax_agreement"] for p in per_sample),
        "atol": args.atol,
    }
    # Parity between one class and itself proves nothing: require two distinct implementations.
    result["parity_ok"] = (
        result["distinct_implementations"]
        and result["max_abs_logprob_diff"] <= args.atol
        and result["min_argmax_agreement"] == 1.0
    )
    print(json.dumps(result))
    return 0 if result["strict_load_ok"] and result["parity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
