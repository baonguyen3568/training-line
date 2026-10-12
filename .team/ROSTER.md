# Đội ngũ dự án VLM Training

| tmux session      | Tên gọi (SendMessage) | Vai trò | Trách nhiệm chính |
|-------------------|-----------------------|---------|-------------------|
| bao-trainning-1   | trainning-model-6a (gọi là bao-pm) | Product Manager | Điều phối, tổng hợp, giải quyết mâu thuẫn, chốt ADR cùng user |
| bao-trainning-2   | bao-research  | Research        | Nghiên cứu stack NVIDIA mới nhất, chọn base model, benchmark chuẩn, viết proposal kiến trúc |
| bao-trainning-3   | bao-backend   | Backend/ML Eng  | Code pipeline: data → train → quantize → eval → serve, Docker arm64, CLI |
| bao-trainning-4   | bao-tester    | Tester/QA       | Test strategy, pytest, smoke test pipeline nhỏ, eval harness đúng chuẩn, regression |
| bao-trainning-5   | bao-security  | Security        | Supply chain (container/wheels/model weights), secrets, license dataset/model, an toàn GPU/host chung |
| bao-trainning-6   | bao-fixcode   | Fix code        | Nhận bug từ tester/security/review, debug và sửa, root-cause |
| bao-trainning-7   | bao-cleanup   | Dọn dẹp         | Quản lý disk/cache/checkpoint/container `vlm-*`, retention policy, giữ repo gọn (CHỈ trong phạm vi dự án) |
| bao-trainning-8   | bao-cleancode | Clean code      | Chuẩn code: ruff/mypy/pre-commit, refactor, kiến trúc module, review PR |
| bao-trainning-9   | bao-frontend  | Frontend        | Dashboard theo dõi train/eval/benchmark, so sánh model, demo inference |
| bao-trainning-10  | bao-design    | Design          | UX/UI, design system cho dashboard, trực quan hoá benchmark, docs/diagram kiến trúc |

## Ai cần nói chuyện với ai nhiều nhất
- research ⇄ backend (chọn stack, recipe), research ⇄ tester (benchmark/eval protocol)
- backend ⇄ tester ⇄ fixcode (vòng lặp bug)
- security ⇄ backend (deps, container, secrets), security ⇄ cleanup (xoá an toàn)
- cleancode ⇄ backend/fixcode (review), cleanup ⇄ backend (checkpoint retention)
- frontend ⇄ design (UI), frontend ⇄ backend (API/metrics format)
- Mọi người → bao-pm khi: cần quyết định, có mâu thuẫn, bị block, cần user cấp quyền/token.

> **Địa chỉ SendMessage của PM là `trainning-model-6a`** (khi tài liệu ghi "bao-pm" thì gửi tới địa chỉ này).
> Bỏ qua các session không thuộc dự án (vd `comfy-ui-ff`, `Card ComfyUI...`) — không nhắn, không đụng vào.
