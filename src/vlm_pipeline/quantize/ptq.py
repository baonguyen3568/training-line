"""FP8 post-training quantization of a native VLM with ModelOpt (node ``quantize.modelopt_ptq``, B-4).

Container entrypoint: ``python -m vlm_pipeline.quantize.ptq --model DIR --data DIR --out DIR [--calib-size N]``.
Measured in the B-4 probe (.team/proposals/001-stack-model-catalog.md): ModelOpt 0.44 in the 26.08 images quantizes
and exports Qwen3.5 (FP8, language model only) with transformers 5.12.

- Calibration uses the dataset's ``train`` split only; the parity check uses ``validation`` when it exists (never test).
- The vision tower and ``lm_head`` stay BF16 (ModelOpt hf_ptq default: only the language model is quantized).
- Writes ``OUT/quantized`` (HF checkpoint + ``hf_quant_config.json`` + processor files) and ``OUT/report.json``:
  parity vs BF16 (teacher-forced top-1 agreement, KL) and sizes. Greedy exact match is NOT a gate: one different token
  splits two greedy continuations.
Host-safe at import (heavy imports live inside functions).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

FORMATS = ("fp8",)
AUTO_CLASS = "AutoModelForImageTextToText"  # remote-code models (Nemotron Parse) are not supported yet (P-5)
# Modules kept BF16. Qwen3-VL / Qwen3.5 / Gemma vision towers live under these names; ModelOpt skips conv1d itself.
SKIP_PATTERNS = ("*lm_head*", "*visual*", "*vision_tower*", "*vision_model*", "*multi_modal_projector*")
CALIB_PROMPT = "Describe this image."  # same default prompt as training (make_cord_v2_dataset / data.local_folder)
PARITY_SAMPLES = 16
PARITY_MAX_NEW_TOKENS = 64
# Parity is a sanity check in two levels; the user-facing quality gate is the benchmark (eval quant_vs_parent).
# Typical FP8 level (A/B 2026-10-13, Qwen3.5-0.8B, 16 CORD val images, calib 32/128/512): top-1 0.928-0.935
# (per-image min 0.86-0.89), KL 0.021-0.023, independent of the calib size.
#   warn: clearly below that typical level; the model is still handed over (QNT-002).
#   fail: the quantization is broken (QNT-001). Not measured on a broken model yet: set far below the typical level.
# One model measured: re-check when a second model (Qwen3.5-2B) is quantized.
MIN_TOP1_AGREEMENT = 0.90
MAX_KL = 0.05
FAIL_TOP1_AGREEMENT = 0.80
FAIL_KL = 0.20
ParityVerdict = Literal["pass", "warn", "fail", "none"]  # contract with app/quantize_worker (QNT-001 / QNT-002)


def load_images(data_dir: Path, split: str, n: int) -> list[Any]:
    """First ``n`` images of ``split`` from a parquet HF snapshot (``data/<split>-*.parquet``, ``image`` column)."""
    import pyarrow.parquet as pq  # type: ignore[import-not-found]  # container-only

    from vlm_pipeline.common.image_rules import open_image_bytes  # one image-rule source for every dataset

    root = data_dir / "data" if (data_dir / "data").is_dir() else data_dir
    out: list[Any] = []
    for f in sorted(root.glob(f"{split}-*.parquet")):
        for row in pq.read_table(f, columns=["image"]).to_pylist():
            out.append(open_image_bytes(row["image"]["bytes"]))
            if len(out) == n:
                return out
    return out


def _encode(processor: Any, image: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CALIB_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    batch = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to("cuda") for k, v in batch.items()}


def _greedy(model: Any, batch: dict[str, Any]) -> Any:
    out = model.generate(**batch, max_new_tokens=PARITY_MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    return out[0, batch["input_ids"].shape[1] :].cpu()


def _answer_logits(model: Any, batch: dict[str, Any], answer: Any) -> Any:
    """Logits predicting each answer token, teacher-forced (float32, CPU)."""
    import torch

    prompt_len = batch["input_ids"].shape[1]
    ids = torch.cat([batch["input_ids"], answer[None].to("cuda")], dim=1)
    full = {**batch, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for key, value in batch.items():  # other per-token tensors (e.g. mm_token_type_ids): answer tokens are text = 0
        if key not in ("input_ids", "attention_mask") and value.dim() == 2 and value.shape[1] == prompt_len:
            full[key] = torch.cat([value, value.new_zeros((1, answer.shape[0]))], dim=1)
    logits = model(**full).logits[0]
    return logits[prompt_len - 1 : prompt_len - 1 + answer.shape[0]].float().cpu()


def parity_stats(agree: list[float], kls: list[float]) -> dict[str, Any]:
    """Summary of the per-sample parity numbers and the sanity verdict (pure, unit-tested)."""
    if not agree:
        return {"samples": 0, "verdict": "none"}
    top1, kl = sum(agree) / len(agree), sum(kls) / len(kls)
    verdict: ParityVerdict
    if top1 < FAIL_TOP1_AGREEMENT or kl > FAIL_KL:
        verdict = "fail"
    elif top1 < MIN_TOP1_AGREEMENT or kl > MAX_KL:
        verdict = "warn"
    else:
        verdict = "pass"
    return {
        "samples": len(agree),
        "top1_agreement_mean": round(top1, 4),
        "top1_agreement_min": round(min(agree), 4),
        "kl_mean": round(kl, 5),
        "kl_max": round(max(kls), 5),
        "verdict": verdict,
        "thresholds": {
            "warn_top1": MIN_TOP1_AGREEMENT,
            "warn_kl": MAX_KL,
            "fail_top1": FAIL_TOP1_AGREEMENT,
            "fail_kl": FAIL_KL,
        },
    }


def quant_cfg(fmt: str) -> dict[str, Any]:
    """ModelOpt config for ``fmt`` with SKIP_PATTERNS disabled (ModelOpt >= 0.44: ordered list, later entries win)."""
    import modelopt.torch.quantization as mtq

    if fmt not in FORMATS:
        raise ValueError(f"unsupported format {fmt!r}")
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pattern in SKIP_PATTERNS:
        if isinstance(cfg["quant_cfg"], list):
            cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
        else:
            cfg["quant_cfg"][pattern] = {"enable": False}
    return dict(cfg)


def dir_gb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9, 3)


def run(model_dir: Path, data_dir: Path, out: Path, *, fmt: str, calib_size: int) -> dict[str, Any]:
    import modelopt
    import modelopt.torch.quantization as mtq
    import torch
    import transformers
    from modelopt.torch.export import export_hf_checkpoint

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = getattr(transformers, AUTO_CLASS).from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
    model = model.to("cuda").eval()
    calib = load_images(data_dir, "train", calib_size)
    if not calib:
        raise ValueError("no calibration images in the train split")
    held = load_images(data_dir, "validation", PARITY_SAMPLES)
    report: dict[str, Any] = {
        "format": fmt, "modelopt": modelopt.__version__, "transformers": transformers.__version__,
        "model_class": type(model).__name__, "calib_split": "train", "calib_samples": len(calib),
        "parity_split": "validation" if held else None, "skip_patterns": list(SKIP_PATTERNS),
    }  # fmt: skip
    with torch.no_grad():
        refs = []
        for img in held:
            b = _encode(processor, img)
            ans = _greedy(model, b)
            refs.append((ans, _answer_logits(model, b, ans)))

        def forward_loop(m: Any) -> None:
            for img in calib:
                m(**_encode(processor, img))

        t1 = time.time()
        model = mtq.quantize(model, quant_cfg(fmt), forward_loop)
        report["calib_s"] = round(time.time() - t1, 1)
        summary = io.StringIO()
        with contextlib.redirect_stdout(summary):
            mtq.print_quant_summary(model)
        (out / "quant_summary.txt").write_text(summary.getvalue(), encoding="utf-8")
        agree, kls = [], []
        for img, (ans, ref) in zip(held, refs, strict=True):
            q = _answer_logits(model, _encode(processor, img), ans)
            agree.append((q.argmax(-1) == ref.argmax(-1)).float().mean().item())
            kl = torch.nn.functional.kl_div(
                q.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean"
            )
            kls.append(kl.item())
    report["parity"] = parity_stats(agree, kls)
    qdir = out / "quantized"
    export_hf_checkpoint(model, export_dir=str(qdir))
    processor.save_pretrained(qdir)
    report["size_gb"] = {"bf16": dir_gb(model_dir), fmt: dir_gb(qdir)}
    report["torch_alloc_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)  # real peak: run.json
    report["wall_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vlm_pipeline.quantize.ptq")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="HF dataset snapshot (parquet) with train/validation")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--format", default="fp8", choices=FORMATS)
    ap.add_argument("--calib-size", type=int, default=128)
    a = ap.parse_args(argv)
    report = run(a.model, a.data, a.out, fmt=a.format, calib_size=a.calib_size)
    print(json.dumps(report, indent=2))
    from vlm_pipeline.common.errors import ExitCode

    # warn: exit 0, the worker reads report.json (QNT-002); fail: QNT-001
    return ExitCode.QUANT_PARITY_FAIL if report["parity"]["verdict"] == "fail" else ExitCode.OK


if __name__ == "__main__":
    sys.exit(main())
