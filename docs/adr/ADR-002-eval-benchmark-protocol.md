# ADR-002 — Eval & benchmark protocol

- Trạng thái: **ACCEPTED** — 2026-10-12
- Nguồn: `.team/proposals/002-eval-benchmark-protocol.md` (Phần A bao-research, Phần B bao-tester); cleancode approve spec test
- Chốt bởi: bao-pm

## Quyết định
- Harness chính: endpoint OpenAI-compatible (vLLM). Đối chiếu chéo bằng lmms-eval in-process. Judge (nếu cần) là model local được pin; không gửi dữ liệu eval ra API ngoài.
- Gate G0–G5 do **app thực thi và hiển thị** (PASS / CẢNH BÁO / FAIL + giải thích). Node phía sau bị chặn khi gate fail.
- **G1 reproduce base**: rule-scored |Δ| ≤ 1.5pt trên cả 2 đường harness, chạy **full split**.
  - Trạng thái **INCONCLUSIVE** chỉ dùng khi model card không công bố đủ cấu hình. Phải thoả CẢ 4 điều kiện: |Δ|≤5pt, HF vs vLLM ≤1pt, unparseable ≤2%, spot-check ≥30 output. Thiếu 1 là FAIL.
  - Ngân sách máy chung: ưu tiên cột non-thinking. Thinking-on chỉ chạy MMMU + OCRBench.
- **G1-harness**: được tải 1 model nhỏ có số trên OpenVLM leaderboard để đối chiếu harness (<20GB, qua security-model-vetting).
- Perf (G4): đo theo `tester-perf-bench` (/proc/meminfo, không dùng nvidia-smi memory), preset cố định, đo cả FP8 và NVFP4.

## Amendment A1 (2026-10-12) — Run tái lập được
- Mọi run dùng làm **gate hoặc số công bố** (G1, G3, N-4, P-2/P-3, so sánh template) phải chạy từ **git worktree sạch** ở 1 commit cụ thể.
  Tạo bằng `git worktree add .worktrees/<run_id> <commit>`; `.worktrees/` nằm trong project và được gitignore. Không động tới working tree chung.
- Bắt buộc `reproducibility_gaps() == []` (git_commit, git_dirty=false, container_digest đều có). Thiếu → kết quả bị đánh dấu NON-REPRODUCIBLE, không được dùng làm gate.
- Run từ working tree chung chỉ dùng cho smoke/debug.
- Dọn worktree sau khi run xong (cleanup quản lý, dry-run trước).
