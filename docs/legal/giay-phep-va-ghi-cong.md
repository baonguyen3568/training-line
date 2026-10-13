---
title: Giấy phép & ghi công
description: "Giấy phép của {{PRODUCT_NAME}} và các dự án mã nguồn mở, font, icon, model, dataset được dùng hoặc nhắc tới."
---

# Giấy phép & ghi công

## {{PRODUCT_NAME}}
Mã nguồn: **Apache License 2.0**. Văn bản đầy đủ có trong file `LICENSE` của mã nguồn.
{{PRODUCT_NAME}} là dự án độc lập, không liên kết với và không được NVIDIA bảo trợ.

## Phần mềm bên thứ ba trong ứng dụng
Danh sách đầy đủ, kèm phiên bản, giấy phép và thông báo bản quyền, được sinh tự động từ
`configs/security/licenses.yaml` khi build (file `THIRD_PARTY_LICENSES`). Các thành phần chính:

| Thành phần | Dùng cho | Giấy phép |
|---|---|---|
| React, React DOM | Giao diện | MIT |
| React Flow (`@xyflow/react`) | Canvas node-graph | MIT, © webkid GmbH |
| dagre | Tự sắp xếp node | MIT |
| Zustand | Quản lý trạng thái giao diện | MIT |
| uPlot | Biểu đồ loss/metric | MIT |
| d3 (color, drag, zoom…) | Tương tác canvas (qua React Flow) | ISC / BSD-3-Clause |
| FastAPI, Starlette, Uvicorn, Pydantic | Server local | MIT / BSD-3-Clause |

**Engine chạy trong container** (người dùng tự tải từ NVIDIA NGC, theo điều khoản của NGC):
NVIDIA NeMo AutoModel (Apache-2.0), PyTorch (BSD-3-Clause), Hugging Face Transformers (Apache-2.0). Phần bổ sung do dự
án cài thêm: albumentations (MIT), OpenCV (Apache-2.0).

Ý tưởng giao diện tham khảo ComfyUI và LLaMA-Factory. Chúng tôi **không** sao chép mã nguồn của ComfyUI (GPL-3.0).

## Website
| Thành phần | Giấy phép |
|---|---|
| Astro, Starlight | MIT |
| Pagefind (tìm kiếm tài liệu, nếu dùng) | MIT |
| Lucide 1.51.0 (icon) | ISC; một phần từ Feather, MIT |
| Be Vietnam Pro, JetBrains Mono (font) | SIL Open Font License 1.1 |
| Hình minh hoạ | Xem danh sách `site/ASSETS.md` (nguồn, tác giả, giấy phép từng file) |

Danh sách đầy đủ phần mềm bên thứ ba được gửi kèm website, cùng toàn văn giấy phép, có ở trang
[Phần mềm bên thứ ba](/chinh-sach/ben-thu-ba/).

## Model và dataset trong template
{{PRODUCT_NAME}} **không phân phối** model hay dataset. Bạn tải chúng từ nguồn chính chủ, và giấy phép của từng
cái được áp dụng trực tiếp giữa bạn và chủ sở hữu.

| Tên | Chủ sở hữu | Giấy phép | Ghi chú |
|---|---|---|---|
| NVIDIA Nemotron Parse v1.1 | NVIDIA | NVIDIA Open Model License; tokenizer CC-BY-4.0 | Có điều khoản chấm dứt nếu vô hiệu hoá guardrail. Encoder C-RADIOv2-H cũng theo NVIDIA Open Model License |
| Qwen3-VL-2B-Instruct | Qwen (Alibaba) | Apache-2.0 | |
| CORD v2 | NAVER Clova | CC BY 4.0 | *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing* (Park et al.), https://github.com/clovaai/cord. Dữ liệu mẫu trong ảnh chụp màn hình được lấy từ đây |

Nếu một template dùng model **NVIDIA Cosmos** (vd Cosmos-Reason2), trang template đó sẽ ghi
**"Built on NVIDIA Cosmos"** theo yêu cầu của NVIDIA Open Model License, kèm câu "Licensed by NVIDIA Corporation
under the NVIDIA Open Model License" khi phân phối lại model.

## Nhãn hiệu
NVIDIA, NeMo, Nemotron, Cosmos, DGX, CUDA và TensorRT là nhãn hiệu hoặc nhãn hiệu đã đăng ký của NVIDIA Corporation tại
Hoa Kỳ và các quốc gia khác. Hugging Face là nhãn hiệu của Hugging Face, Inc. Qwen là nhãn hiệu của Alibaba Cloud.
Các tên được nhắc tới chỉ để mô tả phần mềm mà {{PRODUCT_NAME}} chạy trên đó. Việc nhắc tới không ngụ ý được bảo trợ.
