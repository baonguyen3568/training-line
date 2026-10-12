# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: m1-journey.e2e.ts >> M1 journey: build, run, save, reopen
- Location: e2e/m1-journey.e2e.ts:64:1

# Error details

```
Error: node model.hf_pick has no data-id
```

# Page snapshot

```yaml
- generic [ref=e3]:
  - banner [ref=e4]:
    - button "▣ VLM Studio" [ref=e5] [cursor=pointer]
    - navigation [ref=e6]:
      - button "Workflow" [ref=e7] [cursor=pointer]
      - button "Lịch sử chạy" [ref=e8] [cursor=pointer]
    - generic [ref=e9]:
      - button "Đơn giản" [pressed] [ref=e10] [cursor=pointer]
      - button "Nâng cao" [ref=e11] [cursor=pointer]
    - generic [ref=e12]:
      - generic "GPU · Rảnh" [ref=e13]
      - generic [ref=e14]: RAM 79/121 GB · còn trống 42 GB
      - generic [ref=e15]: 💾 Dung lượng còn trống 2000 GB
  - note [ref=e16]: 🧪 Chế độ demo (dữ liệu giả lập) — chưa có gì được train thật.
  - generic [ref=e17]:
    - complementary "Thư viện bước" [ref=e18]:
      - generic [ref=e19]:
        - strong [ref=e20]: Thư viện bước
        - button "Thu gọn thư viện" [ref=e21] [cursor=pointer]: ⟨
      - 'searchbox "Tìm bước… (vd: nén, LoRA, benchmark)" [active] [ref=e22]'
      - generic [ref=e23]:
        - heading "① Nguồn" [level=3] [ref=e24]
        - list [ref=e25]:
          - listitem [ref=e26]:
            - button "Chọn model gốc Chọn model có sẵn để bắt đầu. Danh sách đã được kiểm tra giấy phép." [ref=e27]:
              - generic [ref=e28]: Chọn model gốc
              - generic [ref=e29]: Chọn model có sẵn để bắt đầu. Danh sách đã được kiểm tra giấy phép.
          - listitem [ref=e30]:
            - button "Thư mục ảnh + câu hỏi/đáp án Dùng ảnh và câu hỏi/đáp án của bạn trong thư mục dự án." [ref=e31]:
              - generic [ref=e32]: Thư mục ảnh + câu hỏi/đáp án
              - generic [ref=e33]: Dùng ảnh và câu hỏi/đáp án của bạn trong thư mục dự án.
          - listitem [ref=e34]:
            - button "Dataset Hugging Face Lấy bộ dữ liệu có sẵn trên Hugging Face." [ref=e35]:
              - generic [ref=e36]: Dataset Hugging Face
              - generic [ref=e37]: Lấy bộ dữ liệu có sẵn trên Hugging Face.
      - generic [ref=e38]:
        - heading "② Chuẩn bị" [level=3] [ref=e39]
        - list [ref=e40]:
          - listitem [ref=e41]:
            - button "Chia dữ liệu học / kiểm tra Tách một phần dữ liệu để kiểm tra model sau khi học." [ref=e42]:
              - generic [ref=e43]: Chia dữ liệu học / kiểm tra
              - generic [ref=e44]: Tách một phần dữ liệu để kiểm tra model sau khi học.
      - generic [ref=e45]:
        - heading "③ Huấn luyện" [level=3] [ref=e46]
        - list [ref=e47]:
          - listitem [ref=e48]:
            - button "Fine-tune nhẹ (LoRA) Dạy model trả lời theo dữ liệu của bạn, tốn ít bộ nhớ." [ref=e49]:
              - generic [ref=e50]: Fine-tune nhẹ (LoRA)
              - generic [ref=e51]: Dạy model trả lời theo dữ liệu của bạn, tốn ít bộ nhớ.
          - listitem [ref=e52]:
            - button "Fine-tune toàn bộ Dạy model trả lời theo dữ liệu của bạn bằng cách cập nhật các phần model cần học." [ref=e53]:
              - generic [ref=e54]: Fine-tune toàn bộ
              - generic [ref=e55]: Dạy model trả lời theo dữ liệu của bạn bằng cách cập nhật các phần model cần học.
          - listitem [ref=e56]:
            - button "Gộp phần học thêm vào model Gộp LoRA vào model gốc thành một model hoàn chỉnh." [ref=e57]:
              - generic [ref=e58]: Gộp phần học thêm vào model
              - generic [ref=e59]: Gộp LoRA vào model gốc thành một model hoàn chỉnh.
      - generic [ref=e60]:
        - heading "⑤ Nén" [level=3] [ref=e61]
        - list [ref=e62]:
          - listitem [ref=e63]:
            - button "Nén model Làm model nhỏ và chạy nhanh hơn, chất lượng có thể giảm nhẹ." [ref=e64]:
              - generic [ref=e65]: Nén model
              - generic [ref=e66]: Làm model nhỏ và chạy nhanh hơn, chất lượng có thể giảm nhẹ.
      - generic [ref=e67]:
        - heading "⑥ Đo điểm" [level=3] [ref=e68]
        - list [ref=e69]:
          - listitem [ref=e70]:
            - button "Đo điểm benchmark Chấm điểm model trên các bộ đề chuẩn theo năng lực." [ref=e71]:
              - generic [ref=e72]: Đo điểm benchmark
              - generic [ref=e73]: Chấm điểm model trên các bộ đề chuẩn theo năng lực.
    - region "untitled" [ref=e74]:
      - toolbar "VLM Studio":
        - button "＋ Workflow trống" [ref=e75] [cursor=pointer]
        - button "📂 Mở workflow" [ref=e76] [cursor=pointer]
        - button "💾 Lưu workflow" [ref=e77] [cursor=pointer]
        - button "↶ Hoàn tác" [ref=e78] [cursor=pointer]
        - button "↷ Làm lại" [disabled] [ref=e79]
        - button "⧉ Nhân bản" [ref=e80] [cursor=pointer]
        - button "🗑 Xoá" [ref=e81] [cursor=pointer]
      - application [ref=e82]:
        - generic [ref=e84]:
          - generic:
            - generic:
              - group [ref=e85]:
                - generic [ref=e86]:
                  - generic [ref=e87]:
                    - generic [aria-hidden] [ref=e88]: "1"
                    - generic "Chọn model gốc" [ref=e89]
                  - generic [ref=e90]: nemotron-parse-v1.1
                  - generic [ref=e91]:
                    - list
                    - list [ref=e92]:
                      - listitem [ref=e93]: Model
                  - generic [ref=e95]: ○ Sẵn sàng
              - group [ref=e97]:
                - generic [ref=e98]:
                  - generic [ref=e99]:
                    - generic [aria-hidden] [ref=e100]: "2"
                    - generic "Thư mục ảnh + câu hỏi/đáp án" [ref=e101]
                  - generic [ref=e102]: sample-invoices
                  - generic [ref=e103]:
                    - list
                    - list [ref=e104]:
                      - listitem [ref=e105]: Dữ liệu
                  - generic [ref=e107]: ○ Sẵn sàng
              - group [ref=e109]:
                - generic [ref=e110]:
                  - generic [ref=e111]:
                    - generic [aria-hidden] [ref=e112]: "3"
                    - generic "Fine-tune toàn bộ" [ref=e113]
                  - generic [ref=e114]:
                    - list [ref=e115]:
                      - listitem [ref=e116]: Model gốc
                      - listitem [ref=e118]: Dữ liệu học
                      - listitem [ref=e120]: Dữ liệu kiểm tra (tuỳ chọn)
                    - list [ref=e122]:
                      - listitem [ref=e123]: Model hoàn chỉnh
                  - generic [ref=e125]: ○ Sẵn sàng
            - generic:
              - generic: ① Nguồn
              - generic: ② Chuẩn bị
              - generic: ③ Huấn luyện
              - generic: ④ Chạy thử
              - generic: ⑤ Nén
              - generic: ⑥ Đo điểm
              - generic: ⑦ Xuất
        - generic "Control Panel" [ref=e127]:
          - button "Zoom In" [ref=e128] [cursor=pointer]
          - button "Zoom Out" [ref=e131] [cursor=pointer]
          - button "Fit View" [ref=e134] [cursor=pointer]
          - button "Toggle Interactivity" [ref=e137] [cursor=pointer]
        - link "React Flow attribution" [ref=e141] [cursor=pointer]:
          - /url: https://reactflow.dev/attribution
          - text: React Flow
      - status [ref=e143]: Đã đặt "Chọn model gốc" vào làn ① Nguồn
    - complementary "Fine-tune toàn bộ" [ref=e144]:
      - heading "Fine-tune toàn bộ" [level=2] [ref=e145]
      - paragraph [ref=e146]: Dạy model trả lời theo dữ liệu của bạn bằng cách cập nhật các phần model cần học.
      - generic [ref=e147]:
        - generic [ref=e148]: Nhận từ — Model gốc
        - combobox "Nhận từ — Model gốc" [ref=e149]:
          - option "—" [selected]
          - option "Chọn model gốc · Model"
      - generic [ref=e150]:
        - generic [ref=e151]: Nhận từ — Dữ liệu học
        - combobox "Nhận từ — Dữ liệu học" [ref=e152]:
          - option "—" [selected]
          - option "Thư mục ảnh + câu hỏi/đáp án · Dữ liệu"
      - generic [ref=e153]:
        - generic [ref=e154]: Nhận từ — Dữ liệu kiểm tra (tuỳ chọn)
        - combobox "Nhận từ — Dữ liệu kiểm tra (tuỳ chọn)" [ref=e155]:
          - option "—" [selected]
          - option "Thư mục ảnh + câu hỏi/đáp án · Dữ liệu"
      - generic [ref=e156]:
        - generic [ref=e157]:
          - generic [ref=e158]: Thời lượng huấn luyện
          - spinbutton "Thời lượng huấn luyện" [ref=e159]: "2"
        - paragraph [ref=e160]: Số lượt model đọc hết dữ liệu. Nhiều lượt hơn thì lâu hơn và dễ học vẹt.
      - button "▸ Nâng cao (1)" [ref=e161] [cursor=pointer]
    - region "Kiểm tra trước khi chạy" [ref=e162]:
      - generic [ref=e163]:
        - strong [ref=e164]: "Kiểm tra trước khi chạy:"
        - generic [ref=e165]: ✓ Mọi thứ đã sẵn sàng
        - button "▶ Chạy" [disabled] [ref=e166]
```

# Test source

```ts
  1   | // M1 acceptance journey (BOARD M1-6): blank workflow -> drag 3 nodes -> connect -> Run (FakeWorker / mock)
  2   | // -> see node status -> save -> reopen. Selectors agreed with bao-frontend (36b3081); UI text via i18n keys.
  3   | import { readFileSync } from "node:fs";
  4   | import type { Locator, Page } from "@playwright/test";
  5   | import { expect, test, tr } from "./fixtures";
  6   | 
  7   | const NODES = ["model.hf_pick", "data.local_folder", "train.sft_vlm"] as const;
  8   | 
  9   | function node(page: Page, typeId: string): Locator {
  10  |   return page.getByTestId(`node-${typeId}`);
  11  | }
  12  | 
  13  | async function nodeId(page: Page, typeId: string): Promise<string> {
  14  |   const id = await node(page, typeId).getAttribute("data-id");
> 15  |   if (!id) throw new Error(`node ${typeId} has no data-id`);
      |                  ^ Error: node model.hf_pick has no data-id
  16  |   return id;
  17  | }
  18  | 
  19  | async function addFromLibrary(page: Page, typeId: string, how: "drag" | "dblclick"): Promise<void> {
  20  |   const search = page.getByRole("searchbox", { name: tr("library.search") });
  21  |   await search.fill(typeId.split(".").pop() ?? typeId);
  22  |   const item = page.getByTestId(`library-item-${typeId}`);
  23  |   if (how === "drag") {
  24  |     await item.dragTo(page.locator(".react-flow__pane"), { targetPosition: { x: 300, y: 200 } });
  25  |   } else {
  26  |     await item.dblclick();
  27  |   }
  28  |   await search.fill("");
  29  |   await expect(node(page, typeId)).toBeVisible();
  30  | }
  31  | 
  32  | async function buildGraph(page: Page): Promise<void> {
  33  |   await page.goto("/");
  34  |   await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  35  |   await expect(page.locator(".react-flow")).toBeVisible();
  36  |   await expect(page.locator(".react-flow__node")).toHaveCount(0);
  37  | 
  38  |   await addFromLibrary(page, NODES[0], "drag"); // real HTML5 drag-and-drop onto the canvas
  39  |   await addFromLibrary(page, NODES[1], "dblclick");
  40  |   await addFromLibrary(page, NODES[2], "dblclick");
  41  |   await expect(page.locator(".react-flow__node")).toHaveCount(3);
  42  | 
  43  |   const [model, data, train] = await Promise.all(NODES.map((t) => nodeId(page, t)));
  44  |   // Wire 1 by dragging handle to handle on the canvas.
  45  |   await page
  46  |     .locator(`[data-id="${model}"] .react-flow__handle.source[data-handleid="model"]`)
  47  |     .dragTo(
  48  |       page.locator(`[data-id="${train}"] .react-flow__handle.target[data-handleid="base_model"]`),
  49  |     );
  50  |   // Wire 2 through the inspector ("Nhận từ — <port>"), the keyboard-accessible path.
  51  |   await node(page, NODES[2]).click();
  52  |   await page.locator("#recv-train_data").selectOption(`${data}:dataset`);
  53  |   await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  54  |   await expect(page.locator("#recv-base_model")).toHaveValue(`${model}:model`);
  55  | }
  56  | 
  57  | test("first screen is the gallery and offers a blank workflow", async ({ page }) => {
  58  |   await page.goto("/");
  59  |   await expect(page.getByRole("heading", { level: 1, name: tr("gallery.title") })).toBeVisible();
  60  |   await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  61  |   await expect(page.locator(".react-flow")).toBeVisible();
  62  | });
  63  | 
  64  | test("M1 journey: build, run, save, reopen", async ({ page }, testInfo) => {
  65  |   await buildGraph(page);
  66  | 
  67  |   // Run: every node ends completed (or cached); the check bar shows no blocking issue.
  68  |   await expect(page.getByTestId("check-bar").locator('[data-code][class*="error"]')).toHaveCount(0);
  69  |   const run = page.getByRole("button", { name: tr("app.run") });
  70  |   await expect(run).toBeEnabled();
  71  |   await run.click();
  72  |   for (const t of NODES) {
  73  |     await expect(node(page, t)).toHaveAttribute("data-status", /^(completed|cached)$/, {
  74  |       timeout: 30_000,
  75  |     });
  76  |   }
  77  |   await expect(page.getByRole("button", { name: tr("app.run") })).toBeEnabled();
  78  | 
  79  |   // Save: a plain download of the workflow file.
  80  |   const [download] = await Promise.all([
  81  |     page.waitForEvent("download"),
  82  |     page.getByRole("button", { name: tr("workflow.save") }).click(),
  83  |   ]);
  84  |   const saved = testInfo.outputPath("saved.vlmflow.json");
  85  |   await download.saveAs(saved);
  86  |   const wf = JSON.parse(readFileSync(saved, "utf-8")) as {
  87  |     nodes: { id: string; type: string }[];
  88  |     edges: { from: string[]; to: string[] }[];
  89  |   };
  90  |   expect(wf.nodes.map((n) => n.type).sort()).toEqual([...NODES].sort());
  91  |   expect(wf.edges).toHaveLength(2);
  92  | 
  93  |   // Reopen: start from an empty workflow, open the saved file, the same graph comes back.
  94  |   await page
  95  |     .getByRole("button", { name: tr("gallery.blank_workflow") })
  96  |     .first()
  97  |     .click();
  98  |   await expect(page.locator(".react-flow__node")).toHaveCount(0);
  99  |   await page.getByTestId("workflow-file-input").setInputFiles(saved);
  100 |   await expect(page.locator(".react-flow__node")).toHaveCount(3);
  101 |   await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  102 |   for (const t of NODES) await expect(node(page, t)).toBeVisible();
  103 | });
  104 | 
```