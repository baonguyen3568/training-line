"""G0 env probe of the vLLM image for B-7 (proposal 014): runs INSIDE vlm-vllm:0.31.0, --network none, under the lock.

``python scripts/probe/vllm_probe.py MODEL_DIR [IMAGE [PROMPT]]``: prints versions, CUDA capability, a bf16 matmul, a
short greedy text generation with vLLM's offline ``LLM`` at gpu_memory_utilization 0.15 (env VLLM_PROBE_MEM_UTIL)
and, with IMAGE (a local file, never a URL), one greedy chat answer about it. A ModelOpt FP8 export is detected from
its hf_quant_config.json.
Every cache goes to /tmp (the image has no USER and jobs run as --user uid:gid), usage stats are off (bao-security).
Prints one JSON line.
"""

import os

# Paths inside the throw-away container (not the host): the image has no writable home for --user uid:gid.
TMP = "/tmp"  # noqa: S108
CONTAINER_TMP_ENV = {
    "HOME": TMP,
    "XDG_CACHE_HOME": f"{TMP}/xdg",
    "VLLM_CACHE_ROOT": f"{TMP}/vllm",
    "TRITON_CACHE_DIR": f"{TMP}/triton",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
os.environ.update(CONTAINER_TMP_ENV)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    out: dict[str, object] = {
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out["capability"] = list(torch.cuda.get_device_capability())
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        out["bf16_matmul_ok"] = bool(torch.isfinite(a @ a).all())
    t0 = time.time()
    llm = LLM(
        model=sys.argv[1],
        gpu_memory_utilization=float(os.environ.get("VLLM_PROBE_MEM_UTIL", "0.15")),  # 0.15 ≈ 18 GB: enough for <=2B
        max_model_len=8192,
        enforce_eager=True,
        seed=0,
        limit_mm_per_prompt={"image": 1},
    )
    out["quantization"] = getattr(llm.llm_engine.model_config, "quantization", None)
    out["load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    res = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
    out["generate_s"] = round(time.time() - t0, 2)
    out["text"] = res[0].outputs[0].text
    if len(sys.argv) > 2:
        from PIL import Image

        image = Image.open(sys.argv[2], formats=["PNG", "JPEG"]).convert("RGB")
        prompt = sys.argv[3] if len(sys.argv) > 3 else "Read this receipt: list the items and the total."
        messages = [
            {"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": prompt}]}
        ]
        t0 = time.time()
        chat = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=256))
        out["image_generate_s"] = round(time.time() - t0, 2)
        out["image_prompt_tokens"] = len(chat[0].prompt_token_ids or [])
        out["image_text"] = chat[0].outputs[0].text
    out["max_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
