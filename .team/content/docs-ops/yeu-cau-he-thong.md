---
title: Yêu cầu hệ thống
description: Phần cứng, phần mềm và dung lượng cần có để chạy ứng dụng train VLM trên máy NVIDIA.
sources: [CLAUDE.md#phan-cung, docker/IMAGES.lock, docs/adr/ADR-005-storage-retention.md, .team/proposals/005-storage-retention-policy.md#7]
---

# Yêu cầu hệ thống

Ứng dụng hiện được phát triển và kiểm thử trên **một máy duy nhất: NVIDIA DGX Spark (chip GB10)**. Các cấu hình khác
chưa được kiểm thử; bảng dưới chỉ ghi những gì đã chạy thật.

## Phần cứng đã kiểm thử

| Hạng mục | Cấu hình |
|----------|----------|
| Máy | NVIDIA DGX Spark, 1 GPU GB10 |
| Kiến trúc CPU | aarch64 (ARM 64-bit) |
| Bộ nhớ | 128 GB bộ nhớ hợp nhất (CPU và GPU dùng chung) |
| Driver / CUDA | Driver 580 (đã thử 580.142), CUDA 13 |
| Ổ đĩa | Nên còn trống ít nhất 500 GB cho dự án (xem bên dưới) |

> [!note]
> Trên DGX Spark, CPU và GPU dùng chung một vùng 128 GB. Vì vậy "bộ nhớ còn trống" mà app kiểm tra là bộ nhớ
> chung của cả máy, không phải bộ nhớ GPU riêng. Công cụ `nvidia-smi` không hiển thị con số này; app đọc trực tiếp
> từ hệ điều hành.

<!-- popover: "Bộ nhớ hợp nhất là gì?" -->
Bộ nhớ hợp nhất (unified memory) nghĩa là GPU không có bộ nhớ riêng: model, dữ liệu và các ứng dụng khác trên máy
cùng chia 128 GB. Nếu máy đang chạy dịch vụ khác, phần còn lại cho việc train sẽ ít hơn.

## Phần mềm

| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| Hệ điều hành | Linux aarch64 (DGX OS) | |
| Docker + NVIDIA Container Toolkit | bản đi kèm DGX OS | việc train chạy trong container |
| Python | 3.12 | quản lý bằng `uv` |
| Node.js | cần cho bước build giao diện | chỉ khi tự build giao diện từ mã nguồn |
| Image train | `vlm-automodel:26.08-5` (dựng từ `nvcr.io/nvidia/nemo-automodel:26.08`, bản arm64) | ~29,6 GB mỗi image |

Mọi image và thư viện dùng trong dự án đều phải có bản **arm64**. Image gốc được ghim theo mã băm (digest) trong
`docker/IMAGES.lock`, không dùng tag trôi.

## Bộ nhớ (RAM)

Trước mỗi tác vụ nặng (train, quantize, benchmark), app kiểm tra bộ nhớ còn trống theo công thức:

- Còn trống ≥ **mức dự kiến của tác vụ + max(10 GB, 20% mức dự kiến)**
- và còn trống tối thiểu **20 GB** trong mọi trường hợp
- và swap còn trống ≥ **2 GB**.

Nếu không đạt, app **không chạy** và báo lỗi "Không đủ bộ nhớ cho bước này" kèm con số cụ thể. Tác vụ không khai báo
mức dự kiến được tính là 40 GB.

<!-- popover: "Ví dụ" -->
Tác vụ dự kiến dùng 30 GB cần còn trống ít nhất 30 + max(10; 6) = 40 GB. Tác vụ dự kiến 80 GB cần 80 + 16 = 96 GB.

## Dung lượng ổ đĩa

Ước lượng cho giai đoạn hiện tại (các model Qwen3.5 0,8B–9B, chưa gồm model 27B trở lên):

| Hạng mục | Ước lượng |
|----------|-----------|
| Model tải về (Qwen3.5 0,8B / 2B / 4B / 9B) | ~36 GB |
| Image container dự án | ~30 GB mỗi image |
| Dữ liệu đánh giá | ~10–30 GB |
| Checkpoint LoRA | vài trăm MB mỗi lần chạy |
| Checkpoint full fine-tune Nemotron-Parse v1.1 (đo thực tế) | ~8,1 GB mỗi checkpoint (5,4 GB model + 2,7 GB trạng thái optimizer) |
| Model đã gộp + bản nén FP8/NVFP4 + engine | ~60 GB |
| Bộ nhớ đệm kết quả node (giới hạn trên) | ≤ 200 GB |
| **Tổng trường hợp xấu nhất** | **~430 GB** |

Mức cảnh báo mặc định là **500 GB** và mức nguy hiểm **800 GB** cho toàn bộ dữ liệu của dự án. Mỗi model 27–35B thêm
khoảng 55–72 GB.

Trước khi train, app còn kiểm tra ổ đĩa có đủ chỗ cho các checkpoint sắp ghi:
**dung lượng 1 checkpoint × (số checkpoint giữ lại + 1) + 50 GB dự phòng**.

## Máy dùng chung

App được thiết kế cho máy có thể đang chạy dịch vụ khác:

- Mỗi lúc chỉ **một tác vụ nặng** dùng GPU (xem [Xử lý sự cố](./xu-ly-su-co-bo-nho)).
- App chỉ ghi và xoá dữ liệu **trong thư mục dự án**; không động vào container, image hay cache của ứng dụng khác.
- App không dùng `sudo` và không xoá bộ nhớ đệm hệ điều hành để "lấy thêm RAM".
