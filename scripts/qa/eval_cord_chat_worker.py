"""CORD-v2 generation for CHAT VLMs (Qwen3.5, Gemma…) trained to answer the ``gt_parse`` JSON.

Runs INSIDE a vlm-automodel image; scoring happens on the host (``cord.score_rows``: same metrics as Nemotron-Parse).

Parity with "Chạy thử" (serve.trial chat path) and with training:
- processor from the pinned BASE snapshot, with the training run's min/max pixels;
- model = the folder given (base snapshot, merged export) or the base + a PEFT adapter (``--adapter``);
- the training prompt, one user turn ``[image, text]``, ``apply_chat_template(add_generation_prompt=True,
  enable_thinking=False)`` (the answer is the JSON, no reasoning block);
- decoding = base generation_config.json (or the model's defaults when the base has none), then the template's
  overrides, then ``max_new_tokens``; greedy (``do_sample=False``), batch 1. Everything is recorded per row.
Records carry ``answer_format: "json"`` so the host scorers pick the JSON reader instead of Donut's token2json.

Usage (repo mounted at /workspace):
  python /workspace/scripts/qa/eval_cord_chat_worker.py --model <dir> [--adapter <dir>] --processor <base dir> \
      --generation-config <base dir> --data <cord-v2 snapshot> --split test --ids 0,1,2 --prompt "..." \
      [--min-pixels N --max-pixels N] [--max-new-tokens 1024] [--repetition-penalty 1.0] --out <pred.jsonl>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

WORKER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _local_dir(path: str, what: str) -> Path:
    """Only pinned local folders (ADR-003): never a hub id resolved to `main`."""
    p = Path(path)
    if not p.is_dir():
        raise SystemExit(f"--{what} must be a local directory (pinned), got {path!r}")
    return p.resolve()


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def folder_items(folder: Path) -> list[dict[str, Any]]:
    """Rows of an image folder (research's data/reference/cord-v2-*: labels.jsonl {image, target, prompt} + PNGs)."""
    lines = (folder / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def load_item(i: int, ds: Any, folder: Path | None, rows: list[dict[str, Any]]) -> tuple[Any, Any, str | None]:
    """(RGB image, gt_parse, image file sha256) of row ``i``.

    The HF dataset row, or the folder PNG decoded like vLLM does it (common.image_rules.open_image), so a parity run
    reads the same pixels on both engines.
    """
    if folder is None:
        item = ds[i]
        return item["image"].convert("RGB"), gt_parse_of(item), None
    from vlm_pipeline.common.image_rules import open_image
    from vlm_pipeline.common.local_records import image_name_ok

    if i >= len(rows):
        raise SystemExit(f"--ids {i}: the folder has {len(rows)} rows")
    name = rows[i]["image"]
    if not isinstance(name, str) or not image_name_ok(name):  # rule R1: a label never points outside the folder
        raise SystemExit(f"row {i}: image name not allowed")
    path = folder / name
    return open_image(path), rows[i]["target"], hashlib.sha256(path.read_bytes()).hexdigest()


def gt_parse_of(item: dict[str, Any]) -> Any:
    """CORD-v2 rows store the answer as a JSON string under ``ground_truth`` (``{"gt_parse": ...}``)."""
    return json.loads(item["ground_truth"])["gt_parse"]


def _generation_config(model: Any, args: Any) -> tuple[Any, dict[str, Any]]:
    from transformers import GenerationConfig

    gen_dir = _local_dir(args.generation_config, "generation-config")
    if (gen_dir / "generation_config.json").is_file():
        config, source = GenerationConfig.from_pretrained(str(gen_dir)), "base generation_config.json"
    else:  # e.g. Qwen3.5-0.8B ships none: the defaults generate() would use (same rule as serve.trial)
        config = copy.deepcopy(model.generation_config)
        source = "none in base -> transformers defaults (model config)"
    overrides: dict[str, Any] = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if args.repetition_penalty is not None:
        overrides["repetition_penalty"] = args.repetition_penalty
    for key, value in overrides.items():
        setattr(config, key, value)
    config.temperature, config.top_p, config.top_k = None, None, None  # greedy: sampling knobs are inert
    decoding = {k: getattr(config, k, None) for k in ("do_sample", "num_beams", "repetition_penalty", "max_new_tokens")}
    decoding.update({"_source": source, "overrides": overrides, "generation_config_dir": str(gen_dir)})
    return config, decoding


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (LoRA output) applied on --model")
    ap.add_argument("--processor", required=True, help="pinned base snapshot (processor + chat template)")
    ap.add_argument("--generation-config", required=True, help="pinned base snapshot (generation_config.json)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--prompt", required=True, help="the training prompt, verbatim")
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--repetition-penalty", type=_positive_float, default=None)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--images-dir", default=None, help="read images/ground truth from this folder (labels.jsonl)")
    ap.add_argument("--out", required=True)
    return ap


def main() -> int:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = _parser().parse_args()

    model_dir, data_dir = _local_dir(args.model, "model"), _local_dir(args.data, "data")
    processor_dir = _local_dir(args.processor, "processor")
    pixels = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels} if args.max_pixels else {}
    processor = AutoProcessor.from_pretrained(str(processor_dir), **pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation=args.attn
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(_local_dir(args.adapter, "adapter")))
    model = model.to("cuda").eval()
    config, decoding = _generation_config(model, args)
    decoding["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
    eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    prompt_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    print("decoding", json.dumps(decoding), "prompt_sha", prompt_sha, flush=True)

    folder = _local_dir(args.images_dir, "images-dir") if args.images_dir else None
    rows = folder_items(folder) if folder else []
    ds = None if folder else load_dataset(str(data_dir), split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in (int(x) for x in args.ids.split(",") if x):
            image, gt, image_sha = load_item(i, ds, folder, rows)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, generation_config=config)
            prompt_ids = [int(t) for t in inputs["input_ids"][0].tolist()]
            new = [int(t) for t in gen[0].tolist()][len(prompt_ids) :]
            finished = bool(new) and new[-1] in eos
            record = {
                "index": i,
                "answer_format": "json",
                "sequence": processor.tokenizer.decode(new, skip_special_tokens=True).strip(),
                "gt_parse": gt,
                "image_sha256": image_sha,
                "token_ids": new,
                "split": args.split,
                "batch_size": 1,
                "n_new_tokens": len(new),
                "truncated": not finished and len(new) >= args.max_new_tokens,
                "prompt": args.prompt,
                "prompt_sha": prompt_sha,
                # same formula as serve.vllm_batch (sha256 of json.dumps(ids), image tokens expanded): HF<->vLLM parity
                "prompt_ids_sha": hashlib.sha256(json.dumps(prompt_ids).encode("utf-8")).hexdigest(),
                "pixels": pixels,
                "model_dir": str(model_dir),
                "adapter_dir": args.adapter,
                "data_dir": str(folder or data_dir),
                "decoding": decoding,
                "worker_sha256": WORKER_SHA256,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}] new_tokens={record['n_new_tokens']} truncated={record['truncated']}", flush=True)
    return 0


if __name__ == "__main__":
    # GB10: create the CUDA context before the checkpoint is loaded on the CPU (fixcode 2026-10-13, common.cuda_init)
    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()
    raise SystemExit(main())
