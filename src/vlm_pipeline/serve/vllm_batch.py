"""Offline vLLM batch (B7-3 step 1) — runs INSIDE vlm-vllm:0.31.0, one shot, under the GPU lock, ``--network none``.

``python3 -m vlm_pipeline.serve.vllm_batch <batch.json>``: loads one model with vLLM's offline ``LLM`` (no server, no
port: bao-security's default for eval/bench), answers every request greedily and writes ``out.jsonl`` (same order and
ids as the requests) + ``engine.json`` next to the request file. Host side: ``serve.vllm_runner.run_vllm_batch``.

Parity with the HF path ("Chạy thử", eval_cord_chat_worker): one user turn [image, text] through the model's chat
template (``enable_thinking`` False unless the request says otherwise), the training template's min/max pixels as
``mm_processor_kwargs``. Images are decoded by ``common.image_rules.open_image`` (bao-security rules) and handed to vLLM
as PIL: vLLM never opens a path itself. A bad image fails that request only (``error.reason``), never the batch.
Imports vllm lazily; stdlib + image_rules otherwise (the image has no pydantic guarantee for our config layer).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vlm_pipeline.common.image_rules import ImageRuleError, open_image

# Container paths only: the image has no USER, jobs run as --user uid:gid with HOME=/tmp (tmpfs in a narrowed job).
CONTAINER_ENV = {
    "VLLM_CACHE_ROOT": "/tmp/vllm",  # noqa: S108
    "XDG_CACHE_HOME": "/tmp/xdg",  # noqa: S108
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
SEED = 0
SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "repetition_penalty"})
OUT_NAME = "out.jsonl"
PARTIAL_NAME = "partial.jsonl"  # rows appended per chunk: a run stopped at 450/500 (memguard, cancel) keeps 449 answers
CHUNK = 64  # requests per llm.chat call; recorded, so two runs to compare use the same batch composition
ENGINE_NAME = "engine.json"


@dataclass(frozen=True)
class BatchItem:
    id: str
    image: str  # container path; "" = text only
    prompt: str
    max_tokens: int
    # Decoding beyond greedy (trial node): temperature / top_p / top_k / repetition_penalty, already resolved by the
    # host (base generation_config -> template overrides -> node params). Empty = greedy (eval gates).
    sampling: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchRequest:
    model_dir: str  # container path of the HF dir to load (base snapshot or a consolidated / quantized export)
    items: list[BatchItem]
    gpu_memory_utilization: float
    max_model_len: int = 8192
    # Pinned KV cache (GB): vLLM then skips its free-memory profiling, which asserts on a shared UMA host when another
    # service frees memory meanwhile ("Error in memory profiling", GB10 2026-10-13). 0 = let vLLM profile.
    kv_cache_gb: float = 0.0
    quantization: str | None = None  # None = what the checkpoint declares (hf_quant_config.json -> "modelopt")
    min_pixels: int = 0  # 0 = the processor's default
    max_pixels: int = 0
    enable_thinking: bool = False
    logprobs: int = 0  # >0: per generated token, the top-N (token id, logprob) for near-tie analysis (tester parity)
    enforce_eager: bool = True  # no torch.compile / CUDA graphs: faster start, no compile cache to persist
    extra: dict[str, Any] = field(default_factory=dict)  # recorded as is (e.g. who asked, gate id)

    @classmethod
    def from_json(cls, text: str) -> BatchRequest:
        raw = json.loads(text)
        raw["items"] = [BatchItem(**i) for i in raw["items"]]
        for item in raw["items"]:
            if unknown := set(item.sampling) - SAMPLING_KEYS:
                raise ValueError(f"unknown sampling keys {sorted(unknown)}")
        return cls(**raw)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _conversation(item: BatchItem, image: Any) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": item.prompt}]
    if image is not None:
        content.insert(0, {"type": "image_pil", "image_pil": image})
    return [{"role": "user", "content": content}]


def _top_logprobs(gen: Any) -> list[list[list[float]]] | None:
    """Per generated token: [[token_id, logprob], ...] by rank (vLLM may add the sampled token if it is not top-N)."""
    if not gen.logprobs:
        return None
    return [
        [[int(tid), round(float(lp.logprob), 5)] for tid, lp in sorted(step.items(), key=lambda kv: kv[1].rank or 0)]
        for step in gen.logprobs
    ]


def _row(item: BatchItem, out: Any) -> dict[str, Any]:
    gen = out.outputs[0]
    ids = list(out.prompt_token_ids or [])
    extra: dict[str, Any] = {}
    if (top := _top_logprobs(gen)) is not None:
        extra = {"token_ids": [int(t) for t in gen.token_ids], "top_logprobs": top}
    return {
        "id": item.id,
        "text": gen.text,
        "finish_reason": gen.finish_reason,  # "stop" | "length"
        "prompt_tokens": len(ids),
        "gen_tokens": len(gen.token_ids),
        "prompt_sha": _sha(out.prompt) if out.prompt else None,  # the chat-templated text (image placeholders as is)
        "prompt_ids_sha": _sha(json.dumps(ids)),
        "decoding": {"temperature": 0.0, **item.sampling, "max_tokens": item.max_tokens},
        **extra,
    }


def _scheduler(llm: Any) -> dict[str, Any]:
    """The real scheduler / KV-cache settings vLLM chose (not the defaults we think it has)."""
    vcfg = getattr(llm.llm_engine, "vllm_config", None)
    sched = getattr(vcfg, "scheduler_config", None)
    cache = getattr(vcfg, "cache_config", None)
    return {
        "max_num_seqs": getattr(sched, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(sched, "max_num_batched_tokens", None),
        "enable_chunked_prefill": getattr(
            sched, "enable_chunked_prefill", getattr(sched, "chunked_prefill_enabled", None)
        ),
        "kv_cache_dtype": getattr(cache, "cache_dtype", None),
    }


def _preemptions(llm: Any) -> int | None:
    """vllm:num_preemptions summed over engines; None when the metrics API is unavailable (never fails the batch)."""
    try:
        return int(sum(m.value for m in llm.get_metrics() if m.name == "vllm:num_preemptions"))
    except Exception:  # noqa: BLE001 - a missing metric must not cost the answers
        return None


def run(request_path: Path) -> dict[str, Any]:
    os.environ.update(CONTAINER_ENV)
    req = BatchRequest.from_json(request_path.read_text(encoding="utf-8"))
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.time()
    mm_kwargs = {"min_pixels": req.min_pixels, "max_pixels": req.max_pixels} if req.max_pixels else None
    llm = LLM(
        model=req.model_dir,
        quantization=req.quantization,
        gpu_memory_utilization=req.gpu_memory_utilization,
        kv_cache_memory_bytes=int(req.kv_cache_gb * 2**30) if req.kv_cache_gb > 0 else None,
        max_model_len=req.max_model_len,
        enforce_eager=req.enforce_eager,
        seed=SEED,
        disable_log_stats=False,  # keeps the metrics registry: num_preemptions for the gate (tester)
        load_format="safetensors",  # bao-security S1: never the .bin/.pt (pickle) branch, even if the dir has both
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=mm_kwargs,
    )
    config = llm.llm_engine.model_config
    load_s = round(time.time() - t0, 1)
    out_dir = request_path.parent
    partial = (out_dir / PARTIAL_NAME).open("w", encoding="utf-8")

    def emit(row: dict[str, Any]) -> None:
        rows[row["id"]] = row
        partial.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows: dict[str, dict[str, Any]] = {}
    # (max_tokens, sampling) -> requests: one SamplingParams per llm.chat call
    todo: dict[tuple[int, tuple[tuple[str, float], ...]], list[tuple[BatchItem, Any]]] = {}
    for item in req.items:
        image = None
        if item.image:
            try:
                image = open_image(Path(item.image))
            except ImageRuleError as exc:
                emit({"id": item.id, "error": {"code": "IMAGE", "reason": exc.reason}})
                continue
        key = (item.max_tokens, tuple(sorted(item.sampling.items())))
        todo.setdefault(key, []).append((item, image))
    t0 = time.time()
    for (max_tokens, sampling), group in sorted(todo.items()):
        decoding: dict[str, Any] = {"temperature": 0.0, **dict(sampling)}  # greedy unless resolved otherwise
        if "top_k" in decoding:
            decoding["top_k"] = int(decoding["top_k"])  # vLLM wants an int
        logprobs = req.logprobs or None
        params = SamplingParams(n=1, max_tokens=max_tokens, seed=SEED, logprobs=logprobs, **decoding)
        for start in range(0, len(group), CHUNK):
            chunk = group[start : start + CHUNK]
            outs = llm.chat(
                [_conversation(item, image) for item, image in chunk],
                params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": req.enable_thinking},
            )
            for (item, _), out in zip(chunk, outs, strict=True):
                emit(_row(item, out))
            partial.flush()
    generate_s = round(time.time() - t0, 1)
    partial.close()

    with (out_dir / OUT_NAME).open("w", encoding="utf-8") as f:
        for item in req.items:  # request order, ids unchanged (tester pairs bf16/FP8 by id)
            f.write(json.dumps(rows[item.id], ensure_ascii=False) + "\n")
    engine = {
        "engine": "vllm",
        "vllm": vllm.__version__,
        "model_dir": req.model_dir,
        "quantization": getattr(config, "quantization", None),
        "dtype": str(getattr(config, "dtype", "")),
        "max_model_len": req.max_model_len,
        "gpu_memory_utilization": req.gpu_memory_utilization,
        "kv_cache_gb": req.kv_cache_gb,
        "enforce_eager": req.enforce_eager,
        "seed": SEED,
        # Batch composition changes kernels/shapes (tester): a bf16-vs-FP8 comparison needs these equal on both runs.
        "chunk": CHUNK,
        "logprobs": req.logprobs,
        **_scheduler(llm),
        "sampling": {
            "default": {"temperature": 0.0, "n": 1},
            "per_request": sorted({k for i in req.items for k in i.sampling}),
        },
        "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
        "mm_processor_kwargs": mm_kwargs,
        "image_decoder": "common.image_rules.open_image",
        "requests": len(req.items),
        "errors": sum("error" in r for r in rows.values()),
        "num_preemptions": _preemptions(llm),
        "load_s": load_s,
        "generate_s": generate_s,
        "extra": req.extra,
    }
    (out_dir / ENGINE_NAME).write_text(json.dumps(engine, ensure_ascii=False, indent=1), encoding="utf-8")
    return engine


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python3 -m vlm_pipeline.serve.vllm_batch <batch.json>", file=sys.stderr)
        return 2
    run(Path(args[0]))
    return 0


__all__ = ["CHUNK", "CONTAINER_ENV", "ENGINE_NAME", "OUT_NAME", "PARTIAL_NAME", "BatchItem", "BatchRequest", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
