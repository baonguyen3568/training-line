# ADR-004 — Code standards, repo layout, local CI

- Trạng thái: **ACCEPTED** — 2026-10-12
- Nguồn: `.team/proposals/004-code-standards-layout-ci.md` (tác giả bao-cleancode)
- Approve: bao-tester, bao-cleanup, bao-backend, bao-fixcode, bao-security
- Chốt bởi: bao-pm

## Quyết định
Chấp nhận D1–D10 của Proposal 004:
- **2 môi trường**: host venv (uv) chỉ chứa phần điều phối + tooling; lib nặng chỉ có trong container NGC (pin ở ADR-001).
- **Layering**: `cli` → stage độc lập `data|train|quantize|eval|serve` → `common`, ép bằng import-linter. Các stage nói chuyện với nhau qua artifact + manifest trên disk.
- Lib nặng chỉ được import trong `<stage>/worker.py` / `_nvidia_compat.py` (ruff TID253).
- ruff 0.16.10, mypy 2.4.0 (strict cho `common` + `*/config.py`, nâng dần), pytest 9 `tests/{unit,integration,gpu}`.
- Integration/GPU test chạy trong container NGC (thống nhất tester ⇄ cleancode).
- Dependency cooldown `exclude-newer = "7 days"`, gitleaks v8.30.1, ruff `S` bật đầy đủ, cấm gọi trực tiếp `torch.load`/`pickle.load`.
- Large-file 1MB, có allowlist cho `design/`, `docs/`.
- `GPU_LOCK` luôn do tiến trình host giữ và ghi PID của host.
- CI local = `scripts/check.sh`; Conventional Commits.

## Hệ quả
- P2-1 làm chung: bao-cleancode lo tooling, bao-backend lo runtime/Docker. Bắt đầu sau ADR-001.
- Đổi version tool phải qua cooldown + security review.

## Amendments
- A1 (2026-10-12, sửa lần 1): test viết trước làm spec phải đánh dấu `xfail(strict=True, raises=ImportError, reason="spec: <owner> implements in P2-x")`. Có `raises` để chỉ xfail khi thiếu module/hàm; fail do bug thật vẫn hiện FAIL. Tester, fixcode, cleancode đồng ý. Spec test luôn dùng `from x import f`; KHÔNG thêm AttributeError vào raises (sẽ che bug thật).
- A2 (2026-10-12, ACCEPTED — backend đồng ý, xem P008 §2.2): GPU_LOCK dùng token. Host sinh token, ghi vào `.team/GPU_LOCK`, truyền vào container qua `VLM_GPU_LOCK_TOKEN`. Fixture `gpu` so token, thiếu hoặc lệch thì fail nhanh. Bỏ cờ env `_HELD`.
