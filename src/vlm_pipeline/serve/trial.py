"""Trial inference ("Chạy thử", node ``serve.vllm``) — runs INSIDE the training image, one shot, under the GPU lock.

``python -m vlm_pipeline.serve.trial <request.json>``: loads the model the way it was trained, answers one image and
writes ``result.json`` next to the request. Parity with training (ADR-001 amendment 1):
- Nemotron-Parse: the decoder prefix is the task prompt encoded with ``add_special_tokens=False`` (the training collate,
  patch 0004) = ``[2, 0, 50004, 50008, 50001]``; generate() starts from it (asserted in the output). The model's own
  remote class + KV cache reproduces NVIDIA's golden ids (tests/integration/automodel/test_nemotron_parse_remote_cache).
- Chat VLMs (Qwen3-VL, Cosmos, Qwen3.5, Gemma): the processor's chat template with the catalog's min/max pixels.
- Decoding: the pinned BASE snapshot's generation_config.json (Parse: repetition_penalty 1.1); only max_new_tokens,
  temperature and top_p come from the node, and every override is recorded.
Images: ``common.image_rules.open_image`` (bao-security rules: format allowlist at open, decompression bomb is an error,
first frame only, EXIF orientation, nothing from the image is logged). Imports torch/transformers lazily.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_pipeline.common.image_rules import (
    IMAGE_FORMATS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    ImageRuleError,
    open_image,
)

PARSE_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
TRIAL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# The image rules live in one place for uploads, local datasets, eval and this job (common.image_rules); the old
# names stay importable for callers of this module.
TrialInputError = ImageRuleError


@dataclass(frozen=True)
class TrialRequest:
    model_dir: str  # model to answer with (base snapshot, consolidated export, or the base when adapter_dir is set)
    generation_config_dir: str  # pinned base snapshot: its generation_config.json sets the decoding
    processor_dir: str  # pinned base snapshot too: an export may not carry the processor/tokenizer files
    family: str  # "nemotron_parse" | "chat"
    image: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    adapter_dir: str | None = None  # LoRA output: PEFT adapter on top of model_dir
    min_pixels: int = 0
    max_pixels: int = 0
    attn: str = "sdpa"
    # The training template's decode_overrides (001a §10.7), applied between the base config and the node params.
    decode_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> TrialRequest:
        return cls(**json.loads(text))


def decoding_overrides(req: TrialRequest) -> dict[str, Any]:
    """Generation-config keys set from the node (greedy unless the user raised the temperature).

    Temperature 0 sets ``do_sample=False`` EXPLICITLY: a base generation_config.json may ship ``do_sample: true`` +
    temperature 0.7 (many chat models), and "temperature 0" must still mean greedy and repeatable (tester).
    """
    # use_cache pinned on: a fine-tuned export carries use_cache=false (training, patch 0003 / BUG-001) in its
    # generation config, and transformers 5 fills every key the base config leaves unset from the model's own generation
    # config, so the export decoded with no KV cache, ~3x slower per token (tester bench R0, 2026-10-13). The remote
    # code class run here has a working cache; eval_cord_worker pins use_cache the same way.
    out: dict[str, Any] = {"max_new_tokens": req.max_new_tokens, "use_cache": True}
    if req.temperature > 0:
        out.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        out["do_sample"] = False
    return out


def _generation_config(req: TrialRequest, model: Any) -> tuple[Any, dict[str, Any]]:
    """Base snapshot's generation_config.json, then the template's decode_overrides, then the node params.

    Some repos ship none (Qwen3.5-0.8B): then the defaults transformers derived from the model config, the same ones
    generate() would use. The source is recorded in the overrides (``_source``).
    """
    import copy

    from transformers import GenerationConfig

    if (Path(req.generation_config_dir) / "generation_config.json").is_file():
        config = GenerationConfig.from_pretrained(req.generation_config_dir)
        source = "base generation_config.json"
    elif req.family == "nemotron_parse":  # needs forced_eos/decoder_start/penalty from NVIDIA's file: never guess
        raise FileNotFoundError(f"{req.generation_config_dir}: Nemotron-Parse base has no generation_config.json")
    else:
        config = copy.deepcopy(model.generation_config)
        source = "model config (no generation_config.json in the base snapshot)"
    for key, value in req.decode_overrides.items():  # base -> template -> node (the same order as eval)
        setattr(config, key, value)
    overrides = decoding_overrides(req)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config, {**overrides, "_source": source, "_template": dict(req.decode_overrides)}


@dataclass
class ParseSession:
    """A loaded Nemotron-Parse ready to answer (shared by "Chạy thử" and bench.perf, so both time the same path)."""

    processor: Any
    model: Any
    prefix: list[int]  # training decoder prefix [2, 0, 50004, 50008, 50001]
    config: Any  # base generation config + node overrides, decoder_start pinned to prefix[0]
    overrides: dict[str, Any]


def load_parse(req: TrialRequest) -> ParseSession:
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(req.processor_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(req.model_dir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    prefix = processor.tokenizer(PARSE_TASK_PROMPT, add_special_tokens=False, padding=False)["input_ids"]
    config, overrides = _generation_config(req, model)
    config.decoder_start_token_id = prefix[0]  # otherwise generate() prepends it again (eval_cord_worker, 001a)
    return ParseSession(processor, model, list(prefix), config, overrides)


def prepare_parse(session: ParseSession, image: Any) -> dict[str, Any]:
    """Model inputs for one RGB image (processor on CPU, then copied to the GPU)."""
    import torch

    inputs = session.processor(images=[image], text=PARSE_TASK_PROMPT, return_tensors="pt", add_special_tokens=False)
    return {
        "pixel_values": inputs["pixel_values"].to("cuda", torch.bfloat16),
        "decoder_input_ids": torch.tensor([session.prefix], device="cuda"),
    }


def generate_parse(session: ParseSession, inputs: dict[str, Any]) -> list[int]:
    """New token ids after the prefix; raises if generate() did not keep the training prefix."""
    import torch

    with torch.no_grad():
        out = session.model.generate(**inputs, generation_config=session.config)
    ids = [int(t) for t in out[0].tolist()]
    if ids[: len(session.prefix)] != session.prefix:
        raise RuntimeError(f"decoder prefix differs from training: {ids[: len(session.prefix) + 3]}")
    return ids[len(session.prefix) :]


def _answer_parse(req: TrialRequest, image: Any) -> dict[str, Any]:
    session = load_parse(req)
    new = generate_parse(session, prepare_parse(session, image))
    return _record(
        req,
        session.processor.tokenizer,
        prefix=session.prefix,
        new=new,
        config=session.config,
        overrides=session.overrides,
        cls=type(session.model),
    )


def _answer_chat(req: TrialRequest, image: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = {"min_pixels": req.min_pixels, "max_pixels": req.max_pixels} if req.max_pixels else {}
    processor = AutoProcessor.from_pretrained(req.processor_dir, **kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        req.model_dir, dtype=torch.bfloat16, attn_implementation=req.attn
    )
    if req.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, req.adapter_dir)
    model = model.to("cuda").eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req.prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    config, overrides = _generation_config(req, model)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=config)
    prompt_ids = [int(t) for t in inputs["input_ids"][0].tolist()]
    new = [int(t) for t in out[0].tolist()][len(prompt_ids) :]
    return _record(
        req, processor.tokenizer, prefix=prompt_ids, new=new, config=config, overrides=overrides, cls=type(model)
    )


def _record(
    req: TrialRequest,
    tokenizer: Any,
    *,
    prefix: list[int],
    new: list[int],
    config: Any,
    overrides: dict[str, Any],
    cls: type,
) -> dict[str, Any]:
    eos = {t for t in (config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]) if t}
    finished = bool(new) and new[-1] in eos
    return {
        "engine_used": "hf",
        "model_class": f"{cls.__module__}.{cls.__qualname__}",
        "family": req.family,
        "prefix_ids": prefix if req.family == "nemotron_parse" else None,
        "prompt_token_count": len(prefix),
        "decoding": {
            k: getattr(config, k, None)
            for k in ("do_sample", "num_beams", "repetition_penalty", "temperature", "top_p", "max_new_tokens")
        },
        "overrides": overrides,
        "text": tokenizer.decode(new, skip_special_tokens=True).strip(),
        "n_tokens": len(new),
        "truncated": not finished and len(new) >= req.max_new_tokens,
        "trial_sha256": TRIAL_SHA256,
    }


def run(request_path: Path) -> dict[str, Any]:
    req = TrialRequest.from_json(request_path.read_text(encoding="utf-8"))
    try:
        image = open_image(Path(req.image))
    except TrialInputError as exc:
        result: dict[str, Any] = {"error": {"code": "TRIAL_IMAGE", "reason": exc.reason}}
    else:
        result = (_answer_parse if req.family == "nemotron_parse" else _answer_chat)(req, image)
    (request_path.parent / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m vlm_pipeline.serve.trial <request.json>", file=sys.stderr)
        return 2
    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()  # before the checkpoint is loaded on the CPU: GB10 context creation (KB fixcode §2.2b)
    result = run(Path(args[0]))
    return 3 if "error" in result else 0


__all__ = [
    "IMAGE_FORMATS",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_PIXELS",
    "ParseSession",
    "TrialInputError",
    "TrialRequest",
    "generate_parse",
    "load_parse",
    "open_image",
    "prepare_parse",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
