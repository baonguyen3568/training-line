"""B-4 spike: FP8 PTQ of a native VLM with the ModelOpt shipped in the image, plus an output-parity check and sizes.

Runs inside a project image via ``vlm_pipeline.common.jobrun_cli`` (offline, GPU lock, memguard):

    uv run python -m vlm_pipeline.common.jobrun_cli --image vlm-automodel:26.08-5 --tag b4-fp8-probe \
        --stage quantize --est-peak-gb 12 -- python /workspace/scripts/research/probe_modelopt_fp8.py \
        --model /workspace/.cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev> \
        --cord /workspace/.cache/hf/hub/datasets--naver-clova-ix--cord-v2/snapshots/<rev>/data --out RUN_DIR

Calibration = CORD train images (never validation/test); parity = CORD validation images. The vision tower and
lm_head stay BF16 (official ModelOpt VLM recipe quantizes the language model only).
Parity reported: teacher-forced top-1 agreement and mean KL(bf16 || fp8) over the BF16 greedy answers, plus greedy
exact-match rate. Writes RUN_DIR/report.json; the FP8 HF checkpoint goes to RUN_DIR/fp8.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path
from typing import Any

PROMPT = "Read this receipt and list every item with its price."
SKIP_PATTERNS = ("*visual*", "*vision*", "*lm_head*", "*mlp_AR*", "*merger*")


def load_images(data_dir: Path, split: str, n: int) -> list[Any]:
    import pyarrow.parquet as pq
    from PIL import Image

    out: list[Any] = []
    for f in sorted(data_dir.glob(f"{split}-*.parquet")):
        for row in pq.read_table(f, columns=["image"]).to_pylist():
            out.append(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
            if len(out) == n:
                return out
    return out


def encode(processor: Any, image: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    batch = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to("cuda") for k, v in batch.items()}


def greedy(model: Any, batch: dict[str, Any], max_new: int) -> Any:
    out = model.generate(**batch, max_new_tokens=max_new, do_sample=False, num_beams=1)
    return out[0, batch["input_ids"].shape[1] :].cpu()


def answer_logits(model: Any, batch: dict[str, Any], answer: Any) -> Any:
    """Logits that predict each answer token, teacher-forced on ``answer`` (float32, on CPU)."""
    import torch

    prompt_len = batch["input_ids"].shape[1]
    ids = torch.cat([batch["input_ids"], answer[None].to("cuda")], dim=1)
    full = {**batch, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for key, value in batch.items():  # other per-token tensors (e.g. mm_token_type_ids): answer tokens are text = 0
        if key not in ("input_ids", "attention_mask") and value.dim() == 2 and value.shape[1] == prompt_len:
            full[key] = torch.cat([value, value.new_zeros((1, answer.shape[0]))], dim=1)
    logits = model(**full).logits[0]
    start = prompt_len - 1
    return logits[start : start + answer.shape[0]].float().cpu()


def parity(model: Any, processor: Any, images: list[Any], refs: list[tuple[Any, Any]], max_new: int) -> dict[str, Any]:
    import torch

    agree, kls, exact = [], [], 0
    for img, (ans, ref) in zip(images, refs, strict=True):
        b = encode(processor, img)
        q = answer_logits(model, b, ans)
        agree.append((q.argmax(-1) == ref.argmax(-1)).float().mean().item())
        kl = torch.nn.functional.kl_div(q.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean")
        kls.append(kl.item())
        exact += int(torch.equal(greedy(model, b, max_new), ans))
    return {
        "top1_agreement_mean": round(sum(agree) / len(agree), 4),
        "top1_agreement_min": round(min(agree), 4),
        "kl_mean": round(sum(kls) / len(kls), 5),
        "kl_max": round(max(kls), 5),
        "greedy_exact_match": f"{exact}/{len(images)}",
    }


def quantize_fp8(model: Any, processor: Any, calib: list[Any], out_dir: Path) -> Any:
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pattern in SKIP_PATTERNS:  # ModelOpt >= 0.44: quant_cfg is an ordered list, later entries win
        if isinstance(cfg["quant_cfg"], list):
            cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
        else:
            cfg["quant_cfg"][pattern] = {"enable": False}

    def forward_loop(m: Any) -> None:
        for img in calib:
            m(**encode(processor, img))

    model = mtq.quantize(model, cfg, forward_loop)
    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        mtq.print_quant_summary(model)
    (out_dir / "quant_summary.txt").write_text(summary.getvalue(), encoding="utf-8")
    return model


def dir_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cord", required=True, help="CORD-v2 snapshot data/ dir (parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--parity", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    import modelopt
    import torch
    import transformers
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir, t0 = Path(a.out), time.time()
    report: dict[str, Any] = {"modelopt": modelopt.__version__, "transformers": transformers.__version__, **vars(a)}
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to("cuda").eval()
    report["model_class"] = type(model).__name__
    report["model_revision"] = Path(a.model).name  # HF snapshot dir name = commit sha
    report["skip_patterns"] = list(SKIP_PATTERNS)
    calib = load_images(Path(a.cord), "train", a.calib)
    held = load_images(Path(a.cord), "validation", a.parity)
    with torch.no_grad():
        refs = []
        for img in held:
            b = encode(processor, img)
            ans = greedy(model, b, a.max_new)
            refs.append((ans, answer_logits(model, b, ans)))
        t1 = time.time()
        model = quantize_fp8(model, processor, calib, out_dir)
        report["calib_s"] = round(time.time() - t1, 1)
        report["parity"] = parity(model, processor, held, refs, a.max_new)
    fp8_dir = out_dir / "fp8"
    t2 = time.time()
    export_hf_checkpoint(model, export_dir=str(fp8_dir))
    processor.save_pretrained(fp8_dir)
    report["export_s"] = round(time.time() - t2, 1)
    report["size_gb"] = {"bf16": round(dir_gb(Path(a.model)), 3), "fp8": round(dir_gb(fp8_dir), 3)}
    qcfg = fp8_dir / "hf_quant_config.json"
    report["hf_quant_config"] = json.loads(qcfg.read_text()) if qcfg.exists() else None
    # torch allocator peak only: on GB10 (UMA) the real peak is run.json peak_mem_gb (jobrun memguard)
    report["torch_alloc_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
