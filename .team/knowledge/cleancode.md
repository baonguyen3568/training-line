# Knowledge — Clean Code / Architecture (bao-cleancode)

Cập nhật: 2026-10-12. Phiên bản tool đã xác minh trực tiếp từ PyPI JSON API cùng ngày.

## 1. Bộ công cụ (phiên bản hiện tại, 2026-10)

| Tool | Bản mới nhất (ngày) | Vai trò trong dự án | Ghi chú |
|---|---|---|---|
| uv | 0.11.26 (đã cài trên máy, aarch64) | env, lock, run | `uv sync --locked` trong CI local; `[dependency-groups]` (PEP 735) cho dev/test/lint |
| ruff | 0.17.0 (2026-10-09) | lint + format (thay black/isort/flake8/pyupgrade) | 0.15 đổi "2026 style guide" của formatter → **pin chính xác** version, nâng có chủ đích |
| mypy | 2.4.0 (2026-10-01) | type check chính | 2.0: `--local-partial-types`, `--strict-bytes` mặc định on; `-n N` parallel (experimental) |
| ty (Astral) | 0.0.86 | *không dùng làm gate* | beta, 0.0.x breaking, không có plugin (pydantic plugin của mypy là lý do giữ mypy) |
| pre-commit / prek | 4.6.2 / 0.5.5 | git hooks | prek: drop-in, Rust, không cần Python runtime riêng; cùng file `.pre-commit-config.yaml` |
| import-linter | 2.15 (2026-09-04) | ép module boundary (layers/forbidden contracts) | `ignore_imports` để nhận nợ kỹ thuật có đếm được |
| pytest | 9.1.1 | test | `--strict-markers`, `--strict-config`, `--import-mode=importlib` |
| pytest-cov / pytest-xdist | 7.1.0 / 3.8.0 | coverage / song song | xdist **không** dùng cho test GPU (1 GPU, unified memory) |
| pydantic | 2.14.1 | config schema | `extra="forbid"`, `frozen=True` cho config |
| typer | 0.27.3 | CLI | 1 app, sub-command theo stage |
| jaxtyping | 0.3.11 | annotation shape tensor (tuỳ chọn) | chỉ chạy check runtime trong test (beartype), không bật ở hot path |
| deptry 0.25.1 / vulture 2.16 | | dep thừa/thiếu, dead code | chạy định kỳ, không gate |

Nguồn:
- PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`) — phiên bản/ngày ở bảng trên.
- Ruff 0.15 "2026 style guide": https://newreleases.io/project/pypi/ruff/release/0.15.19
- mypy changelog (2.0 parallel, default flags): https://mypy.readthedocs.io/en/stable/changelog.html ; https://pydevtools.com/blog/mypy-2-0-parallel-type-checking/
- ty beta / so sánh type checker: https://pydevtools.com/blog/ty-beta/ ; https://pydevtools.com/handbook/explanation/how-do-mypy-pyright-and-ty-compare.md
- prek: https://prek.j178.dev/index.md ; https://pypi.org/project/prek
- import-linter layers: https://import-linter.readthedocs.io/en/stable/contract_types/layers/ ; Kraken case study: https://blog.europython.eu/kraken-technologies-how-we-organize-our-very-large-pythonmonolith/
- uv dependency groups: https://pydevtools.com/handbook/explanation/understanding-dependency-groups-in-uv.md ; cooldown `exclude-newer`: https://pydevtools.com/handbook/how-to/how-to-protect-against-python-supply-chain-attacks-with-uv/
- NeMo AutoModel coding standards (mirror, cần đối chiếu repo gốc): ruff line-length 120, Google docstring, type hints cho public API, Conventional Commits — https://tomevault.io/tome/NVIDIA-NeMo/Automodel
- jaxtyping: https://docs.kidger.site/jaxtyping/
- pytest GPU markers pattern: https://docs.vllm.ai/projects/vllm-omni/en/stable/contributing/ci/tests_markers/ ; https://apache.googlesource.com/mahout/+/HEAD/testing/conftest.py

## 2. Nguyên tắc kiến trúc cho dự án này

1. **Thin wrapper quanh lib NVIDIA**. NeMo AutoModel / Megatron-Bridge / NeMo-RL / ModelOpt / TRT-LLM đã có recipe + config riêng.
   Code của ta chỉ: (a) validate config của ta bằng pydantic, (b) dịch sang config/API của lib, (c) gọi, (d) ghi artifact + metrics.
   Không viết `BaseTrainer`/`AbstractQuantizer` khi chỉ có 1 implementation. Rule of three trước khi trừu tượng hoá.
2. **Layer & boundary** (ép bằng import-linter):
   ```
   cli (scripts / vlm_pipeline.cli)        ← cao nhất, chỉ wiring
   data | train | quantize | eval | serve   ← stage độc lập, KHÔNG import chéo nhau
   common                                  ← config base, logging, paths, gpu_lock, seed, io
   ```
   Stage giao tiếp qua **artifact trên disk** (đường dẫn checkpoint/dataset + manifest JSON), không qua import Python.
   Lợi ích: mỗi stage chạy độc lập bằng 1 lệnh, chạy được trong container khác nhau (vd TRT-LLM container ≠ NeMo container).
3. **Import nặng phải lazy**. `torch`, `transformers`, `nemo_automodel`, `modelopt`, `tensorrt_llm` chỉ import trong hàm thực thi,
   không ở top-level của module CLI/config → `--help`, test config, lint chạy trên máy không có GPU/container và nhanh.
   Gói dependency nặng thành optional extras / chỉ có trong container NGC; pyproject core giữ nhẹ.
4. **Config**: pydantic v2 model per stage, `model_config = ConfigDict(extra="forbid", frozen=True)`;
   YAML → model; dump resolved config + git SHA + version lib vào output dir (reproducibility).
   Không truyền `dict` trôi nổi giữa các hàm.
5. **Side effects ở rìa**: hàm thuần (build config, tính metric, format data) tách khỏi hàm I/O / GPU → test được không cần GPU.
6. **Không global state ẩn**: seed, device, paths đi qua config/tham số. Logging qua `logging` chuẩn (+ JSONL metrics schema thống nhất với frontend), không `print`.
7. **Errors**: exception riêng tối thiểu (`VlmPipelineError` + vài subclass khi có caller cần phân biệt). Fail fast khi config sai; không nuốt exception.

## 3. Pitfalls riêng cho NVIDIA stack + GB10 aarch64 unified memory

- **Type stubs thiếu**: torch có type, nhưng nemo/megatron/modelopt/tensorrt_llm hầu như không → `ignore_missing_imports` *per-module* trong mypy overrides, KHÔNG global. Bọc lib trong 1 module adapter có type rõ để phần còn lại strict được.
- **mypy/CI chạy ngoài container**: không cài được TRT-LLM/ModelOpt trên host dễ dàng (aarch64 wheel/CUDA 13). Do đó lazy import (mục 2.3) là bắt buộc để lint/typecheck/test unit chạy host.
- **Version lệch giữa host venv và NGC container**: lockfile host chỉ cho tooling + code thuần; runtime lib pin theo tag container NGC (ghi trong config/ADR). Không pip install đè torch trong container NGC.
- **Torch aarch64 + CUDA 13**: wheel PyPI cho aarch64 có thể không khớp CUDA → không đưa torch vào dependency core của host trừ khi bao-research/backend xác nhận index phù hợp.
- **Unified memory 128GB dùng chung (~70GB đã bị chiếm)**: test không được tải model thật mặc định; marker `gpu`/`slow` + skip tự động; không chạy test GPU song song (xdist), luôn qua GPU lock.
- **Ruff formatter đổi style theo năm** → pin exact version trong pre-commit và dev group, cập nhật cùng nhau trong 1 commit riêng.
- **Supply chain** (chốt với bao-security): `[tool.uv] exclude-newer = "7 days"` → ruff pin 0.16.10 (0.17.0 quá mới); gitleaks v8.30.1; ruff `S` đầy đủ; cấm `pickle.load`/`torch.load` trực tiếp.
- Code copy từ recipe NVIDIA (vendored) → để trong thư mục riêng, ghi nguồn + license header, exclude khỏi lint strict, không refactor (để diff được với upstream).

## 4. Checklist review PR (dùng hằng ngày)

**Bắt buộc pass máy**: `ruff format --check`, `ruff check`, `mypy`, `lint-imports`, `pytest -m "not gpu and not slow"`.

**Người review kiểm:**
- [ ] Đúng layer? Có import chéo stage? Có import nặng ở top-level?
- [ ] Abstraction đúng mức: có class/interface nào chỉ 1 implementation, chỉ để "sau này"? → gỡ.
- [ ] Trùng lặp: logic đã có trong `common` hoặc trong lib NVIDIA chưa?
- [ ] Config: pydantic, `extra="forbid"`, default hợp lý, không hard-code path/hyperparam/magic number.
- [ ] Reproducibility: seed, version, resolved config được ghi ra.
- [ ] Error handling: không `except Exception: pass`; message có ngữ cảnh.
- [ ] Typing public API đủ; `Any` có lý do; `# type: ignore[code]` luôn có error code.
- [ ] Test: logic thuần có unit test; GPU path có smoke test đánh marker.
- [ ] Docstring: Google style, vừa đủ (vì sao, không lặp lại cái gì); không comment hiển nhiên.
- [ ] An toàn: không secret, không path ngoài project, GPU lock khi chạy job nặng.
- [ ] Diff nhỏ, 1 mục đích; refactor tách commit khỏi behavior change.

## 5. Mức độ nghiêm khắc theo giai đoạn (pragmatic)
- Phase 2 (build đầu): ruff rule set vừa phải + mypy strict cho `common` & config; các stage `check_untyped_defs`. Không chặn tiến độ backend.
- Sau smoke test E2E: nâng mypy strict toàn package, coverage floor cho `common`.
