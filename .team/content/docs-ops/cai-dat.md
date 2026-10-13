---
title: Cài đặt và chạy ứng dụng
description: Các bước cài đặt từ mã nguồn, dựng image train và mở giao diện trên máy của bạn.
sources: [pyproject.toml, scripts/dev/app.sh, src/vlm_pipeline/app/server.py, frontend/README.md, docker/automodel-2.Dockerfile, docker/IMAGES.lock]
---

# Cài đặt và chạy ứng dụng

Hiện chỉ có cách cài **từ mã nguồn**. Hãy đọc [Yêu cầu hệ thống](./yeu-cau-he-thong) trước.

## 1. Lấy mã nguồn và cài môi trường Python

```bash
git clone <địa chỉ kho mã> Trainning-model
cd Trainning-model
uv sync
```

`uv` tạo môi trường Python 3.12 riêng cho dự án, không ảnh hưởng Python của hệ thống.

## 2. Build giao diện

```bash
cd frontend
npm ci --ignore-scripts
npm run build
cd ..
```

Giao diện được build thành file tĩnh trong `frontend/dist/` và do chính app phục vụ. App không tải script, font hay
thư viện nào từ CDN bên ngoài.

## 3. Chuẩn bị image train

Việc train chạy trong container dựng từ image chính thức của NVIDIA `nemo-automodel:26.08` (bản arm64, ghim theo mã băm
trong `docker/IMAGES.lock`). Image của dự án thêm một số thư viện và bản vá đã được review:

```bash
docker buildx build --builder vlm-builder --platform linux/arm64 \
  -f docker/automodel-2.Dockerfile --build-arg VLM_TAG=26.08-5 \
  -t vlm-automodel:26.08-5 --load .
```

> [!warning]
> Image gốc nặng khoảng 29 GB. Trên máy dùng chung, hãy báo quản trị viên trước khi tải.

<!-- popover: "Vì sao phải dựng image riêng?" -->
Image gốc chưa có vài thư viện xử lý ảnh mà model Nemotron-Parse cần, và có một số lỗi đã được dự án vá. Mỗi bản
vá được kiểm tra mã băm khi build; bản vá lạ sẽ làm build thất bại.

## 4. Chạy app

```bash
uv run vlm-app --worker container
```

| Tuỳ chọn | Ý nghĩa |
|----------|---------|
| `--port 7870` | cổng của app (mặc định 7870) |
| `--worker container` | chạy tác vụ thật trong container; `--worker fake` (mặc định) chỉ giả lập để thử giao diện |

Khi khởi động, app in ra **một lần** đường dẫn đăng nhập dạng `http://127.0.0.1:7870/?token=…`. Mở đường dẫn này
trong trình duyệt trên cùng máy. App chỉ lắng nghe ở `127.0.0.1`, không mở ra mạng ngoài.

> [!note]
> Trên máy dùng chung, dùng `scripts/dev/app.sh` thay cho `uv run vlm-app`. Script này bật hồ sơ `shared_dev`:
> chặt hơn về plugin và yêu cầu quản trị viên duyệt khi xoá hơn 20 GB dữ liệu một lần (xem
> [Quản lý dung lượng](./quan-ly-dung-luong)).

## 5. Kiểm tra trước khi chạy tác vụ nặng

App tự kiểm tra trước mỗi tác vụ nặng. Bạn cũng có thể tự kiểm tra từ dòng lệnh:

```bash
uv run python scripts/cleanup/preflight.py --est-peak-gb 40 --ckpt-gb 9
```

Lệnh chỉ đọc, in kết quả dạng JSON và kết thúc với mã: `0` đạt, `2` thiếu bộ nhớ hoặc swap, `3` thiếu ổ đĩa,
`4` GPU đang bận.

## Dữ liệu được lưu ở đâu

Mọi thứ app tạo ra nằm trong thư mục dự án:

| Thư mục | Nội dung |
|---------|----------|
| `.cache/` | model và dữ liệu tải về của dự án |
| `outputs/runs/` | mỗi lần train/đánh giá một thư mục: cấu hình, nhật ký, số đo, checkpoint |
| `outputs/graph/` | kết quả từng node của workflow và chỉ mục bộ nhớ đệm |
| `outputs/system/` | báo cáo dung lượng, nhật ký dọn dẹp |

Thư mục cache Hugging Face chung của máy (`~/.cache/huggingface`) chỉ được đọc, không bao giờ bị app ghi hay xoá.
