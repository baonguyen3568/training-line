# Nhật ký quyết định (tóm tắt; chi tiết ở docs/adr/)
- 2026-10-12 TẠM: Chỉ dùng image GPU từ nvcr.io, pin digest, không --privileged, cho tới khi host vá CVE-2026-24260 (user quyết).
- 2026-10-12 TẠM (chờ user xác nhận): quota dự án warn 500GB / critical 800GB. Team không chạy drop_caches/sudo — chỉ cảnh báo + pre-flight check RAM trước GPU_LOCK.
- 2026-10-12 Gợi ý PM cho P004: large-file 1MB + allowlist design/ docs/ (cleancode+cleanup tự chốt).
- 2026-10-12 ADR-005 Storage & Retention ACCEPTED (có điều kiện: save_consolidated, quota recheck sau ADR-001).
- 2026-10-12 ADR-004 Code standards ACCEPTED.
- 2026-10-12 Ngưỡng bộ nhớ chung P005/P007: preflight margin max(10GB,20%×peak), SwapFree≥2GB, memguard abort MemAvailable<8GB (fixcode được phản biện).
- 2026-10-12 RESET: sản phẩm là app no-code node-graph (ComfyUI-style) cho no-tech user, hướng NVIDIA. Giai đoạn hiện tại chỉ setup — dừng spike GPU/tải lớn. P006 đổi phạm vi thành node editor; Gradio-only (phương án A) bị loại. ADR-004/005 giữ nguyên.
- 2026-10-12 Driver 580.142: mặc định pin image NGC CUDA ≤13.1; nâng driver ≥595 là quyết định của user.
