# R-1 — Khảo sát app node-graph / no-code ML: fork, extend hay tự xây?
Tác giả: bao-research · 2026-10-12 · Trạng thái: **REVIEW** (gửi PM, backend, frontend, design, security, cleancode)
Đầu vào: 2 khảo sát web (GitHub API, docs chính chủ), đọc trực tiếp ComfyUI v0.33.2 có trên máy (`/home/ntcai/comfy-ui/ComfyUI`, chỉ đọc),
license do bao-cleancode kiểm qua GitHub API. ❓ = chưa xác minh.

## 1. Kết luận (TL;DR)
**Tự xây app, nhưng không tự phát minh**: ghép các khối MIT/Apache, chép ý tưởng (không chép code) từ ComfyUI.
- **Editor**: React Flow / xyflow (MIT); Vue Flow (MIT) nếu frontend chọn Vue.
- **Mô hình node & giao thức**: theo ComfyUI. Gồm schema node có kiểu kiểu V3, queue `/prompt` + WebSocket events, và 2 định dạng file (template/layout ↔ plan thực thi). Chỉ **viết lại**, không copy code GPL.
- **Executor**: scheduler nhẹ của mình (FastAPI + SQLite + docker SDK). Mỗi node nặng biên dịch thành **job container `vlm-*`**. Đầu ra là **artifact handle** (đường dẫn + metadata), không phải object Python. Prefect 3 là phương án nâng cấp nếu cần.
- **Engine sau node**: dạng plugin, nhiều backend.
  - NeMo AutoModel (chính, hướng NVIDIA).
  - **LLaMA-Factory** (Apache-2.0) là backend train thứ 2 để phủ rộng model/method (**sau v1**).
  - ModelOpt (quantize), vLLM (chạy thử/serve), lmms-eval (benchmark).
- **Không fork ComfyUI.** Lý do kỹ thuật đủ mạnh, kể cả khi bỏ qua license (mục 3).

## 2. Bảng so sánh
Fit = mức khớp nhu cầu của ta (1–5): no-code, job GPU dài hàng giờ, aarch64 GB10 dùng chung, validate trước khi chạy, template.

### 2.1 App / nền tảng có thể fork hoặc extend
| Ứng viên | License | Độ trưởng thành (10/2026) | Fit | Lấy gì |
|---|---|---|---|---|
| **ComfyUI** backend + `ComfyUI_frontend` | **GPL-3.0** (cả 2) | ~137k★, release hàng tuần (v0.35–0.39 trong 9–10/2026), frontend v1.55 | **2** | **Ý tưởng**: V3 typed schema (`io.ComfyNode.define_schema`), WS events (`executing/progress/executed/execution_error`), cache theo fingerprint input, lazy input, subgraph, `blueprints/` (template JSON), registry |
| ComfyUI training nodes (FluxTrainer, Lora-Training-in-Comfy…) | Apache / không license | Thử nghiệm hoặc bỏ dở | 1 | Bằng chứng ngược: train **in-process** chặn queue, `subprocess(shell=True)` không progress. Chưa có pack nào cho LLM/VLM/NeMo |
| **LLaMA-Factory + LlamaBoard** | **Apache-2.0** | v0.9.5 (30/05/2026): Qwen3.5/3.6, Gemma 4, Transformers v5; có **playbook DGX Spark chính chủ** (07/2026, pip venv cu130) | **4** (backend) / 2 (UI) | Làm backend train thứ 2 (SFT/LoRA/QLoRA/DPO/KTO/ORPO, nhiều VLM); tham khảo schema YAML, `dataset_info`, template multimodal, mẫu runner (UI → YAML → `Popen` → poll log). Không có GRPO; UI Gradio dạng form |
| NeMo microservices → **NeMo Helix** (Customizer/Evaluator/Studio) | Apache-2.0 (Helix, v0.7.0 06/10/2026); container Customizer cũ cần NVAIE | Mới, đang chuyển đổi | **1.5** | GPU workload chỉ có tài liệu cho x86_64 ❓arm64; DPO/GRPO cần K8s/Ray; không thấy hỗ trợ VLM. Lấy **thiết kế API job**. Data Designer/Evaluator dùng như thư viện |
| NVIDIA AI Workbench | Proprietary (free) | 2026.06.5; forum báo hỏng trên Spark arm64 | 1.5 | Có thể phát hành "project spec" về sau |
| Unsloth Studio | **AGPL-3.0** (UI), core Apache-2.0 | Beta 03/2026 | 3 (tham khảo UX) | UX no-code train → chat → export; Data Recipes là node graph trên NeMo Data Designer. Không lấy code |
| InvokeAI workflow editor | Apache-2.0 | 28.6k★, active | 3 | Ý tưởng **"linear view"**: chỉ phơi vài input của graph cho người không biết code. Typed field templates |
| ChaiNNer | GPL-3.0 | 6.1k★ | 2 | Ý tưởng: **type expressions + validate trước khi chạy** trên React Flow |
| Langflow | MIT | ~v1.9, active | 2 | Mẫu API component Python có kiểu, trên React Flow. Engine chạy đồng bộ in-process, không hợp job dài |
| Flowise | Apache-2.0 **trừ thư mục enterprise** (thương mại); thuộc Workday | Active | 1.5 | — |
| n8n | Sustainable Use License (không OSI) | Active | 1.5 | — |
| Dify | Apache-2.0 có điều khoản bổ sung (cấm SaaS multi-tenant, logo) | Active | 1 | — |
| KNIME / Orange | GPL-3 (+ ngoại lệ node API ❓) / GPL-3 | Trưởng thành | 2 / 1 | Ý tưởng: trạng thái node (configured→executed), reset downstream, cache trên đĩa, preview dữ liệu trên dây |
| Unreal Blueprint | Proprietary | — | Khái niệm | Pin có kiểu và màu, collapsed graph, "compile" báo lỗi trước khi chạy |

### 2.2 Thư viện editor (frontend)
| Thư viện | License | Bảo trì | Typed ports | Subgraph | Minimap / Undo | Fit |
|---|---|---|---|---|---|---|
| **React Flow (@xyflow/react 12.12.0, 2026-10-11)** | MIT | Rất active; Langflow, InvokeAI, ChaiNNer dùng | qua `isValidConnection` | sub-flow | có / tự viết | **5** |
| Vue Flow | MIT | Active | custom | có | có / tự viết | 4 (nếu chọn Vue) |
| Rete.js v2 | MIT | Active | sockets có kiểu sẵn | scopes plugin | plugin | 4 |
| Baklava.js | MIT ❓ | Vừa phải | có | có | không / có | 3 |
| LiteGraph.js (gốc / bản Comfy đã archive) | MIT | Ngừng từ 08/2024 | có | có | — | 2 |
| Frontend ComfyUI (litegraph fork) | GPL-3.0 | Rất active | có | có | có | 2 (GPL) |

### 2.3 Executor / orchestrator
| Công cụ | License | 1 máy? | Fit | Ghi chú |
|---|---|---|---|---|
| **Tự viết: SQLite job table + docker SDK** | — | ✓ | **4** | GPU lock vốn đã serial hoá hàng đợi, nên đây là phương án đơn giản nhất. Cần tự viết resume/cancel |
| Prefect 3 | Apache-2.0 | ✓ | 4 | Retry, cancel, state, log UI. Đang mua lại Dagster (07/2026), tương lai chưa rõ |
| Dagster | Apache-2.0 | ✓ | 3.5 | Mô hình asset/lineage hợp với dataset → ckpt → quantized |
| ZenML | Apache-2.0 | ✓ | 3 | Một số tính năng dashboard là bản Pro |
| Kubeflow Pipelines | Apache-2.0 | ✗ (K8s) | 1 | Quá nặng |

## 3. Vì sao không fork ComfyUI (kể cả khi bỏ qua GPL)
Bằng chứng từ code v0.33.2 trên máy và docs:
1. **Executor in-process**: các node truyền object Python (tensor, model) cho nhau trong 1 process. Train 6 giờ mà crash/OOM là sập luôn server. Không lưu trạng thái, không resume.
2. **Mỗi server chạy 1 prompt tại một thời điểm**: train dài sẽ chặn mọi workflow khác (chạy thử, benchmark).
3. **Cancel kiểu hợp tác** (`/interrupt`): chỉ dừng được ở chỗ node tự kiểm tra cờ ngắt. Ta cần `docker stop` cứng.
4. **`comfy.model_management` giả định VRAM rời** và tự load/offload model. Giả định này sai trên UMA 128GB dùng chung.
5. **Validate chỉ ở mức kiểu và giá trị**: không có hook ước lượng RAM/disk/thời gian, trong khi tầm nhìn sản phẩm bắt buộc có.
6. Hệ sinh thái custom node, trình quản lý model và UI preview ảnh/video tối ưu cho diffusion. Gần như không có gì cho train VLM (mục 2.1).
7. API V3 vẫn ghi "may change"; nhịp release hàng tuần nên **chi phí rebase fork cao**. bao-cleancode cũng lưu ý điểm này trong P004 §8.3.
Phần đáng giá nhất của ComfyUI là **giao thức và mô hình dữ liệu**, và phần này viết lại được bằng ~vài trăm dòng.

## 4. Kịch bản license theo hướng phát hành (phối hợp bao-security cho phần pháp lý)
| | (A) Chỉ dùng nội bộ | (B) Thương mại, đóng nguồn | (C) Thương mại, mở nguồn GPL |
|---|---|---|---|
| Fork ComfyUI / frontend | Hợp lệ (GPL chỉ ràng buộc khi phân phối), nhưng vẫn **không nên** vì lý do kỹ thuật ở mục 3 | ❌ Không được | Hợp lệ; vẫn vướng lý do kỹ thuật |
| Chép code ChaiNNer / Unsloth Studio UI (AGPL) / ComfyUI | Hợp lệ | ❌ | GPL được; AGPL thì phải mở cả phần dịch vụ mạng |
| React Flow / Vue Flow / Rete / LiteGraph gốc (MIT) | ✓ | ✓ (giữ notice) | ✓ |
| LLaMA-Factory, Langflow, Prefect, NeMo AutoModel, ModelOpt (Apache/MIT) | ✓ | ✓ (giữ NOTICE) | ✓ |
| Flowise phần enterprise, Dify, n8n | Đọc kỹ điều khoản | ❌ / hạn chế | ❌ / hạn chế |
| Container NGC (NVIDIA license) | ✓ (user tự pull) | **Không đóng gói lại image**; app hướng dẫn user pull từ nvcr.io ❓ (security xác nhận) | như (B) |
| Model weights | Theo license từng model (catalog P001) | Như cột (A); cẩn thận license non-commercial và NVIDIA Open Model | như (B) |

**User đã chốt (qua PM, 2026-10-12): mục đích là nghiên cứu/phát triển và phát hành miễn phí cho người khác dùng, không bán.**
Như vậy là phi thương mại nhưng **có phân phối**, nên nghĩa vụ GPL/AGPL vẫn áp dụng (công khai source khi phân phối) dù không bán.
Fork ComfyUI là hợp lệ nếu mở nguồn toàn bộ app theo GPL-3.0. Khuyến nghị của research **không đổi** vì các lý do kỹ thuật ở mục 3.
Ngoài ra, giữ lõi MIT/Apache giúp không khoá cửa các lựa chọn license sau này. Pháp lý: chờ bao-security.

**Khuyến nghị kiến trúc giữ được cả 3 phương án**: lõi app chỉ dùng MIT/Apache. Nếu muốn tương thích ComfyUI (ví dụ cho user dùng model đã fine-tune trong ComfyUI), làm thành **gói custom node riêng, repo riêng, license GPL**, giao tiếp với app qua HTTP. Không import chéo.

## 5. Kiến trúc đề xuất (đầu vào cho Proposal 009 của backend + frontend)
```
[Web UI: React Flow canvas · node library · template gallery · "linear view" cho người mới]
        │  workflow.json (layout + giá trị widget + input được phơi ra)    ← file chia sẻ/template
        ▼
[API server (FastAPI)]
   validate(graph): kiểu kết nối · ràng buộc giá trị · estimate() RAM/disk/thời gian từng node
   compile(graph) → plan.json (DAG, image@digest, model@SHA, seed)          ← tái lập được
        ▼
[Scheduler: SQLite job table · GPU_LOCK · memguard (MemAvailable) · cache theo fingerprint input]
        ▼  mỗi node nặng = 1 job container vlm-<run>-<node>
[Engine plugins] train: nemo-automodel | llamafactory · rl: nemo-rl | trl · quantize: modelopt
                 serve/try: vllm | trtllm · eval: lmms-eval | vlmevalkit · data: hf-datasets | curator
        ▼  artifact handle: {uri, kind: dataset|adapter|checkpoint|quantized|report, meta}
[Events: WebSocket (queued/executing/progress/log/executed/error kèm thông báo dễ hiểu)]
```
Hệ kiểu pin: theo P009/P010 (Dataset / Model{Adapter, Checkpoint, QuantizedModel, ExportArtifact} / EvalReport / BenchReport / Endpoint). Research đồng ý Checkpoint là subtype của Model, BaseModel chỉ là nhãn hiển thị.
Mỗi node khai báo: `inputs/outputs có kiểu`, `params (schema pydantic → widget)`, `estimate(params, inputs) → {ram_gb, disk_gb, minutes}`, `engine`, `image`.
Ràng buộc ban đầu: **chỉ DAG** (không loop). Subgraph và blueprint để giai đoạn sau.

Template v1 (gợi ý, research sẽ soạn recipe):
1. "Fine-tune VLM bằng LoRA (Qwen3.5-2B, dataset mẫu)": chạy được trong ≤1h trên Spark.
2. "Chạy thử model + so sánh base vs fine-tuned".
3. "Quantize FP8 + benchmark accuracy/latency".
4. "Benchmark nhanh 1 model trên bộ chuẩn (smoke)".
5. (sau) "GRPO sau SFT", "Nemotron Omni LoRA" (khi máy đủ RAM).

## 6. Rủi ro và việc cần làm tiếp
- GPL lây nhiễm nếu ai đó copy code ComfyUI/ChaiNNer vào repo. **bao-cleancode/security** thêm kiểm tra license vào CI.
- Phạm vi: tự xây editor + scheduler là khối lượng đáng kể. Cắt v1 thành: canvas + 6–8 node + 3 template + 1 engine train (AutoModel).
- LLaMA-Factory trên arm64 chỉ có hướng dẫn pip venv, không có container. Nếu dùng làm backend thứ 2 thì cần tự build image `vlm-llamafactory` trên base `pytorch:25.11` (CUDA 13.0, đã chứng minh chạy native).
- Ràng buộc driver 580 (CUDA ≤13.1) áp cho mọi engine image (xem `.team/knowledge/research.md`).
- ❓ cần xác minh thêm: NeMo Helix trên arm64; điều khoản phân phối lại container NGC.

## Nguồn chính
ComfyUI: https://github.com/Comfy-Org/ComfyUI · https://github.com/Comfy-Org/ComfyUI_frontend · https://docs.comfy.org/custom-nodes/v3_migration · https://docs.comfy.org/development/comfyui-server/comms_messages · https://docs.comfy.org/development/comfyui-server/execution_model_inversion_guide · https://github.com/kijai/ComfyUI-FluxTrainer · https://github.com/LarryJane491/Lora-Training-in-Comfy · https://build.nvidia.com/spark/comfyui
LLaMA-Factory: https://github.com/hiyouga/LlamaFactory/releases · https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/webui/runner.py · https://build.nvidia.com/spark/llama-factory/overview
NVIDIA: https://docs.nvidia.com/nim-operator/latest/customizer.html · https://docs.nvidia.com/nemo-helix/documentation/reference/release-notes/current-release/ · https://github.com/NVIDIA-NeMo/DataDesigner/releases · https://docs.nvidia.com/ai-workbench/user-guide/latest/release-notes/current.html · https://forums.developer.nvidia.com/t/workbench-is-dead-on-dgx-spark/362011 · https://github.com/NVIDIA/dgx-spark-playbooks · https://unsloth.ai/docs/get-started/unsloth-studio
Flow builders / editor: https://reactflow.dev/ · https://github.com/invoke-ai/InvokeAI · https://github.com/chaiNNer-org/chaiNNer · https://docs.datastax.com/en/langflow/changelog.html · https://raw.githubusercontent.com/FlowiseAI/Flowise/main/LICENSE.md · https://docs.dify.ai/policies/open-source · https://github.com/jagenjo/litegraph.js · https://github.com/Comfy-Org/litegraph.js
Orchestrator: https://dagster.io/blog/prefect-is-acquiring-dagster · https://www.zenml.io/blog/dagster-vs-prefect
