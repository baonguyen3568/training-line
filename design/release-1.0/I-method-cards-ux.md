# Luồng I — Thẻ gallery theo PHƯƠNG PHÁP train (ADR-022): spec UX
bao-design · 2026-10-14. Người làm: bao-frontend (thẻ + sơ đồ). Dữ liệu: research/cleancode (`configs/reference_pipelines.yaml`).
Thay cho thẻ pipeline hiện tại. Trục chính là phương pháp; model và data chỉ là cấu hình đã đo, đổi được ở bước nguồn (ADR-021).

## 1. Bố cục thẻ (từ trên xuống)
1. **Tên phương pháp** (h2), kèm chip độ khó bên phải (`pipelines.difficulty.*`, đã có).
2. **Khi nào dùng:** đúng 1 câu, clamp 2 dòng (key `method.<id>.when`, xem §4). Không có câu mô tả thứ hai.
3. **Sơ đồ kiến trúc train** (§2), cao 56–64px cho mỗi giai đoạn; phương pháp có 2 giai đoạn thì có 2 hàng.
4. **Dòng meta:**
   - nguồn: publisher + icon external-link;
   - giấy phép (badge);
   - "Đo với: {model}" (text.secondary; model là danh từ riêng, font thường).
5. **Dòng số đo:** `pipelines.measured` "Đo trên DGX Spark" · thời gian · RAM đỉnh (· điểm nếu có), kèm "?".
   - Popover theo thứ tự: metric_help (nếu có) → so sánh (nếu có) → `score_single_run` → config → nguồn → `pipelines.swap_hint`.
   - Câu "đổi model/data ở bước nguồn" chỉ nằm trong popover này, không thành dòng riêng trên thẻ (luật No-Stray-Text).
6. **Footer:** nút **Mở** (primary, chỉ khi `status: available`) + **Xem nguồn** (hairline); bên trái là trạng thái nếu chưa available.

Thẻ KHÔNG còn hàng chip bước (MiniDiagram cũ); sơ đồ kiến trúc thay vai trò đó. Danh sách bước xem được khi mở pipeline.

## 2. Sơ đồ kiến trúc train (component `ArchDiagram`)
Vẽ bằng SVG từ dữ liệu (§3), không dùng ảnh tĩnh. Mỗi hàng (giai đoạn) có 3 khối nối nhau theo thứ tự:
**Phần nhìn ảnh → Bộ nối → Phần ngôn ngữ** (`arch.block.vision|projector|lm`).

Trạng thái khối không bao giờ chỉ dựa vào màu (luật Never-Colour-Alone):

| Trạng thái | Nền | Viền | Dấu hiệu | Nhãn trong aria/popover |
|---|---|---|---|---|
| `frozen` (đóng băng) | không | hairline nét đứt `border.subtle` | icon `lock` 10px | `arch.state.frozen` |
| `train` (train toàn bộ) | `accent` 16% | 1px `accent` | không | `arch.state.train` |
| `train_low_lr` | `accent` 8% | 1px `accent` | chữ "LR↓" 10px mono | `arch.state.train_low_lr` |
| `lora` | như frozen | như frozen | pill "LoRA" gắn mép phải khối (accent) | `arch.state.lora` |

Pill lượng tử nằm ở góc trên khối, font mono 10px: "4-bit", "FP8", "NVFP4", "FP8 giả lập" (`arch.quant.*`). Pill lượng tử dùng nền surface; màu accent chỉ dành cho trạng thái train.

Khối phụ nằm ngoài hàng chính, hairline, có icon:
- `teacher`: "Model thầy" (icon brain-circuit), đặt bên trái với mũi tên "sinh dữ liệu" vào hàng chính.
- `reference`: "Bản tham chiếu" (khối ma, opacity 0.6), đặt dưới khối LM.
- `rollout`: "Sinh câu trả lời" (icon play).
- `reward`: "Chấm điểm" (icon list-checks).

Nhãn giai đoạn ở đầu mỗi hàng: "GĐ1" / "GĐ2" (`arch.stage`, kèm {n}).

Chú giải: một nút "?" ở góc sơ đồ, popover gồm 4 mẫu trạng thái thu nhỏ kèm nhãn (`arch.legend_title`). Không vẽ chú giải thường trực trên thẻ.

A11y:
- SVG có `role="img"` và `aria-label` sinh từ dữ liệu, ví dụ "Giai đoạn 1: Phần nhìn ảnh đóng băng; Bộ nối train; Phần ngôn ngữ đóng băng."
- Phần tử con có `aria-hidden`.
- Kích thước tối thiểu: khối 64×28px; chữ trong khối 11px. Thẻ hẹp thì 3 khối xếp vừa 280px.

## 3. Dữ liệu cần trong `reference_pipelines.yaml` (research/cleancode chốt)
Đã chốt theo proposal 023 §5: dùng `train_arch` (`full` = train, `full_low_lr` = train_low_lr), `quant` tách riêng khỏi trạng thái, `extras`, `stages`, `measured_with`, `when_to_use_key` = `reference_pipeline.<id>.when`. Khối YAML dưới đây chỉ là minh hoạ ý.
```yaml
method: lora_sft            # id phương pháp → key method.<id>.*
arch:
  stages:
    - {vision: frozen, projector: frozen, lm: lora, quant: null}
  extras: []                # teacher | reference | rollout | reward
measured_with: {model: "Qwen3.5-2B", data: "CORD v2"}
```
Ví dụ:
- QLoRA: `lm: lora, quant: nf4` (vision bf16 nên không có pill).
- LLaVA 2 giai đoạn: GĐ1 `{vision: frozen, projector: train, lm: frozen}`, GĐ2 `{…, projector: train, lm: lora}`.
- PTQ: mọi khối frozen, `quant: fp8`.
- QAT: `lm: train, quant: fp8_fake`.

## 4. Câu "khi nào dùng" (design viết; key `method.<id>.when`, thêm vào i18n khi research chốt id)
| # | Phương pháp | vi |
|---|---|---|
| 1 | Full SFT | Khi có nhiều dữ liệu và cần model đổi cách trả lời nhiều nhất; tốn bộ nhớ nhất. |
| 2 | LoRA SFT | Lựa chọn mặc định: dạy thêm nhanh, ít bộ nhớ, model gốc giữ nguyên. |
| 3 | QLoRA 4-bit | Khi model lớn hơn phần bộ nhớ còn trống: giữ model gốc ở 4-bit rồi dạy LoRA. |
| 4 | 2 giai đoạn kiểu LLaVA | Khi dạy model nhìn một loại ảnh mới: dạy bộ nối ảnh–chữ trước, rồi mới dạy cách trả lời. |
| 5 | Mở phần nhìn ảnh, ảnh độ phân giải cao | Khi ảnh của bạn rất khác ảnh thường gặp, như tài liệu dày chữ: dạy cả phần nhìn ảnh với tốc độ học thấp. |
| 6 | Distillation bằng dữ liệu tổng hợp | Khi có ít đáp án mẫu: dùng một model lớn viết đáp án rồi dạy model nhỏ học theo. |
| 7 | DPO / MPO | Khi có cặp câu trả lời tốt và kém: dạy model chọn câu tốt hơn, bớt bịa. |
| 8 | GRPO | Khi đúng/sai chấm tự động được (số, JSON, đáp án ngắn): model tự thử nhiều câu rồi học từ điểm. |
| 9 | PTQ FP8 / NVFP4 | Khi cần model nhỏ hơn để chạy: nén rồi đo lại điểm, không train thêm. |
| 10 | QAT | Khi nén làm điểm giảm nhiều: train ngắn có giả lập nén để lấy lại điểm. |

EN viết cùng lúc khi thêm key. Câu nào có dấu ":" trong YAML phải đặt trong nháy.

## 5. Site `/template`
- Trang viết lại theo phương pháp, dùng chung `PipelineCard` mới.
- Phần mở đầu (design viết sau khi có thẻ): app gom phương pháp train phổ biến 2024–2026 thành pipeline mở ra là chạy, đổi model/dữ liệu ở bước nguồn; mỗi thẻ ghi số đo thật và nguồn.
- Nhóm bước "Làm sạch dữ liệu" (P4-4) không có thẻ riêng; trang nhắc tới nó trong một câu, link sang /docs/node-va-cong.

## 6. Nghiệm thu (ảnh)
1. Lưới 10 thẻ ở 1440px.
2. Thẻ LoRA (1 giai đoạn).
3. Thẻ LLaVA (2 giai đoạn).
4. Thẻ GRPO (có khối phụ).
5. Popover chú giải.
6. Thẻ ở viewport 390px.
