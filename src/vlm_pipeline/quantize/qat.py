"""Quantization-aware training with ModelOpt (node ``quantize.qat``, ADR-022 #10, P023 §3). Host-safe at import.

Recipe (ModelOpt 0.44 ``examples/llm_qat``, probed in vlm-automodel:26.08-7): calibrate like PTQ (same scope: vision,
projector and lm_head stay BF16) -> a short fine-tune of the fake-quantized model at a low LR (LoRA by default) ->
export the same unified HF checkpoint as PTQ, so Eval / Benchmark / Export after it work unchanged.

Container entrypoint: ``python -m vlm_pipeline.quantize.qat --model DIR --data FILE --images DIR --out DIR [...]``.
``--data`` is a jsonl the host wrote in the run dir (one sample per line: ``image`` relative to ``--images`` under the
Record rule, ``prompt``, ``answer``), so the container never reads records.jsonl or decides splits. The loss covers
the answer tokens only. The model loads safetensors only, never remote code (bao-security, as vLLM S1).

LoRA: adapters on the language model's linear layers, the fake-quantized base frozen. After training the adapters are
merged; the merged weights moved, so the WEIGHT ranges are re-calibrated (``max``) before export, and the report
records how much the merged, re-quantized model drifts from the trained one (top-1 agreement on held samples).

``check_support`` still refuses every QAT node (``worker_pending``) until the host side and the Q2 smoke are done, so
the canvas never offers a run the app cannot start (support invariant).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.quantize.config import QatParams
from vlm_pipeline.quantize.scope import QuantizeNotSupportedError

NODE_TYPE = "quantize.qat"


def _refuse(node_id: str, reason: str, model: CatalogEntry | None = None) -> None:
    params = {"node": node_id, "node_type": NODE_TYPE, "reason": reason}
    if model is not None:
        params["model"] = model.display_name
    raise QuantizeNotSupportedError(f"qat not supported: {reason}", params=params)


def check_support(node_id: str, params: QatParams, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Validate-time twin of what the QAT worker refuses: raise ``QuantizeNotSupportedError`` (CFG-011)."""
    del inputs  # train_data / calib_data kinds: decided with the worker
    if params.format != "fp8":
        _refuse(node_id, "format", model)
    if model is not None:
        if model.auto_class != "AutoModelForImageTextToText":
            _refuse(node_id, "remote_code_model", model)
        if model.quant_fp8 != "official":
            _refuse(node_id, "fp8_not_official", model)
    _refuse(node_id, "worker_pending", model)


# LoRA on the language model only (vision / projector / lm_head stay BF16 and frozen, as the PTQ scope).
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
LORA_R, LORA_ALPHA = 16, 32
GRAD_ACCUM = 8  # batch 1 x 8 = the train templates' global batch
WARMUP_STEPS = 5
MAX_ANSWER_TOKENS = 1024
MAX_STEPS = 10_000
DRIFT_SAMPLES = 8
LOG_EVERY = 5


def read_samples(path: Path, images: Path) -> list[dict[str, str]]:
    """The host-written jsonl: every line ``{"image", "prompt", "answer"}`` with non-empty strings.

    ``image`` follows the Record rule (relative POSIX path, no ``..``, no control characters) and must resolve inside
    ``images``; the returned samples carry the resolved path. The narrow mount is not the only guard.
    """
    from vlm_pipeline.common.local_records import image_name_ok

    root = images.resolve()
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"image", "prompt", "answer"}:
            raise ValueError(f"{path.name}:{n}: expected keys image, prompt, answer")
        if not all(isinstance(row[k], str) and row[k] for k in row):
            raise ValueError(f"{path.name}:{n}: empty or non-string field")
        image = (
            (root / row["image"]).resolve() if image_name_ok(row["image"]) else None
        )  # rule first: NUL never resolves
        if image is None or not image.is_relative_to(root):
            raise ValueError(f"{path.name}:{n}: image must be a relative path inside the images folder")
        out.append({**row, "image": str(image)})
    if not out:
        raise ValueError(f"{path.name}: no samples")
    return out


def optimizer_steps(n_samples: int, epochs: int, max_steps: int) -> int:
    """Optimizer steps for ``epochs`` over ``n_samples`` at GRAD_ACCUM, capped by ``max_steps`` (0 = no cap)."""
    steps = epochs * math.ceil(n_samples / GRAD_ACCUM)
    return min(steps, max_steps) if max_steps else steps


def lr_at(step: int, total: int, peak: float) -> float:
    """Linear warmup over WARMUP_STEPS, then cosine to 10% of peak (step counts from 0)."""
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def _example(processor: Any, sample: dict[str, str]) -> dict[str, Any]:
    """Model inputs + labels (-100 on the prompt) for one sample, on CUDA.

    The prompt is encoded exactly as PTQ / eval encode it (chat template, generation prompt, thinking off); the
    answer is the rest of the rendered conversation, so the end-of-turn token is learned too.
    """
    import torch

    from vlm_pipeline.common.image_rules import open_image

    image = open_image(Path(sample["image"]))
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}]}]
    turn = [*user, {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}]
    prompt_text = processor.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)
    full_text = processor.apply_chat_template(turn, enable_thinking=False)
    tokenizer = processor.tokenizer
    if full_text.startswith(prompt_text):
        answer_text = full_text[len(prompt_text) :]
    else:  # a template that renders the reply differently: the answer + end of sequence
        answer_text = sample["answer"] + (tokenizer.eos_token or "")
    batch = processor(text=[prompt_text], images=[image], return_tensors="pt")
    answer = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0][:MAX_ANSWER_TOKENS]
    prompt_len = batch["input_ids"].shape[1]
    ids = torch.cat([batch["input_ids"][0], answer])[None]
    out = {**batch, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for key, value in batch.items():  # other per-token tensors (mm_token_type_ids): answer tokens are text = 0
        if key not in ("input_ids", "attention_mask") and value.dim() == 2 and value.shape[1] == prompt_len:
            out[key] = torch.cat([value, value.new_zeros((1, answer.shape[0]))], dim=1)
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    out["labels"] = labels
    return {k: v.to("cuda") for k, v in out.items()}


def _trainable(model: Any, adapter: str) -> Any:
    """LoRA-wrapped model (adapters trainable) or the model with only its language model trainable (full)."""
    if adapter == "lora":
        import peft

        cfg = peft.LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=LORA_TARGETS)
        return peft.get_peft_model(model, cfg)
    for name, param in model.named_parameters():
        param.requires_grad_("language_model" in name and "embed" not in name)
    return model


def _top1(model: Any, examples: list[dict[str, Any]]) -> list[Any]:
    import torch

    with torch.no_grad():
        return [model(**{k: v for k, v in ex.items() if k != "labels"}).logits[0].argmax(-1).cpu() for ex in examples]


def _train(
    model: Any, processor: Any, samples: list[dict[str, str]], total: int, lr: float
) -> tuple[list[float], float]:
    """``total`` optimizer steps (batch 1 x GRAD_ACCUM), epochs over ``samples`` in order; per-step losses."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    losses: list[float] = []
    step, micro, window, t0 = 0, 0, 0.0, time.time()
    while step < total:
        for sample in samples:
            loss = model(**_example(processor, sample)).loss / GRAD_ACCUM
            loss.backward()
            window += loss.item()
            micro += 1
            if micro % GRAD_ACCUM:
                continue
            for group in optim.param_groups:
                group["lr"] = lr_at(step, total, lr)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(round(window, 4))
            if step % LOG_EVERY == 0:
                print(json.dumps({"step": step, "of": total, "loss": round(window, 4)}), flush=True)
            step, window = step + 1, 0.0
            if step >= total:
                break
    model.eval()
    return losses, round(time.time() - t0, 1)


def _merge(model: Any, processor: Any, held: list[dict[str, str]], forward_loop: Any) -> tuple[Any, dict[str, Any]]:
    """Merge the LoRA adapters, re-calibrate the moved weight ranges, measure the drift from the trained model."""
    import modelopt.torch.quantization as mtq

    held_ex = [_example(processor, s) for s in held]
    before = _top1(model, held_ex)
    model = model.merge_and_unload()
    mtq.calibrate(model, algorithm="max", forward_loop=forward_loop)
    after = _top1(model, held_ex)
    agree = [(a == b).float().mean().item() for a, b in zip(before, after, strict=True)]
    drift = {"samples": len(agree), "top1_agreement_mean": round(sum(agree) / len(agree), 4),
             "top1_agreement_min": round(min(agree), 4)}  # fmt: skip
    return model, drift


def run(
    model_dir: Path, data: Path, images: Path, out: Path, *, fmt: str, adapter: str, epochs: int, lr: float,
    max_steps: int, calib_size: int,
) -> dict[str, Any]:  # fmt: skip
    import modelopt
    import modelopt.torch.quantization as mtq
    import torch
    import transformers
    from modelopt.torch.export import export_hf_checkpoint

    from vlm_pipeline.quantize.ptq import AUTO_CLASS, _encode, dir_gb, quant_cfg, skip_patterns

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    samples = read_samples(data, images)
    held = samples[-DRIFT_SAMPLES:]
    total = optimizer_steps(len(samples), epochs, max_steps)
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = (
        getattr(transformers, AUTO_CLASS)
        .from_pretrained(
            model_dir, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        .to("cuda")
    )
    model.config.use_cache = False
    calib = [s["image"] for s in samples[:calib_size]]

    def forward_loop(m: Any) -> None:
        from vlm_pipeline.common.image_rules import open_image

        with torch.no_grad():
            for path in calib:
                m(**_encode(processor, open_image(Path(path))))

    model.eval()
    t1 = time.time()
    model = mtq.quantize(model, quant_cfg(fmt), forward_loop)
    report: dict[str, Any] = {
        "format": fmt, "modelopt": modelopt.__version__, "transformers": transformers.__version__,
        "model_class": type(model).__name__, "calib_samples": len(calib), "calib_s": round(time.time() - t1, 1),
        "skip_patterns": list(skip_patterns("per_tensor")),
    }  # fmt: skip
    model = _trainable(model, adapter)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter == "lora":
        model.enable_input_require_grads()
    losses, train_s = _train(model, processor, samples, total, lr)
    report["qat"] = {
        "adapter": adapter, "epochs": epochs, "lr": lr, "steps": total, "grad_accum": GRAD_ACCUM,
        "samples": len(samples), "train_s": train_s, "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None, "losses": losses,
    }  # fmt: skip
    if adapter == "lora":
        model, report["merge_drift"] = _merge(model, processor, held, forward_loop)
    qdir = out / "quantized"
    export_hf_checkpoint(model, export_dir=str(qdir))
    processor.save_pretrained(qdir)
    report["size_gb"] = {"bf16": dir_gb(model_dir), fmt: dir_gb(qdir)}
    report["torch_alloc_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["wall_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    from vlm_pipeline.quantize.config import QAT_LR

    ap = argparse.ArgumentParser(prog="python -m vlm_pipeline.quantize.qat")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="host-written jsonl: image, prompt, answer")
    ap.add_argument("--images", required=True, type=Path, help="folder the jsonl image names are relative to")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--format", default="fp8", choices=("fp8",))
    ap.add_argument("--adapter", default="lora", choices=("lora", "full"))
    ap.add_argument("--epochs", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--lr", type=float, default=QAT_LR["low"] * 10)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--calib-size", type=int, default=128, choices=(128, 512))
    a = ap.parse_args(argv)
    if not 0 < a.lr <= 1e-3 or not 0 <= a.max_steps <= MAX_STEPS:
        ap.error(f"lr must be in (0, 1e-3] and max-steps in [0, {MAX_STEPS}]")
    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()  # before any checkpoint tensor is on the CPU: GB10 context creation OOM (fixcode)
    report = run(a.model, a.data, a.images, a.out, fmt=a.format, adapter=a.adapter, epochs=a.epochs, lr=a.lr,
                 max_steps=a.max_steps, calib_size=a.calib_size)  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k != "qat"}, indent=2))
    return 0


__all__ = ["NODE_TYPE", "check_support", "lr_at", "optimizer_steps", "read_samples"]


if __name__ == "__main__":
    sys.exit(main())
