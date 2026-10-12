"""Wall-clock of Nemotron-Parse generation with the NATIVE AutoModel class (BUG-001 patch 0003 disables KV cache).

Runs inside vlm-automodel (GPU, network none): ``python parse_generate_timing.py MODEL_DIR [N ...]``.
Uses NVIDIA's deterministic test image from the repo's test_golden.py. Prints one JSON line per token budget.
"""

import json
import sys
import time

import torch
from transformers import AutoProcessor

sys.path.insert(0, sys.argv[1])
from nemo_automodel import NeMoAutoModelForImageTextToText
from test_golden import TASK_PROMPT, make_test_image

model_dir = sys.argv[1]
force = "--force" in sys.argv  # min_new_tokens = budget: measure the length curve even if EOS comes early
budgets = [int(x) for x in sys.argv[2:] if x != "--force"] or [50, 1024]
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = NeMoAutoModelForImageTextToText.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to("cuda").eval()
inputs = processor(images=[make_test_image()], text=TASK_PROMPT, return_tensors="pt", add_special_tokens=False).to(
    "cuda"
)
for n in budgets:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        extra = {"min_new_tokens": n} if force else {}
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False, **extra)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    produced = int(out.shape[-1])
    print(
        json.dumps(
            {
                "max_new_tokens": n,
                "generated_len": produced,
                "seconds": round(dt, 2),
                "tok_per_s": round(produced / dt, 2),
                "max_mem_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
        )
    )
