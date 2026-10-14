# Họ kiến trúc VLM native trong image `vlm-automodel:26.08-7` (ADR-021, research 2026-10-14)

Nguồn: chạy NGAY trong image (transformers 5.12.1, `--network none`, không tải weights). Script dựng mỗi kiến trúc trên meta device từ config mặc định và đọc tên module con; danh sách AutoModel lấy từ `/opt/Automodel/examples/vlm_finetune` và `nemo_automodel/components/models`. 76 model_type trong `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.

**Lưu ý quan trọng cho resolver:** `tie_word_embeddings` và việc có vision_config hay không phụ thuộc TỪNG REPO (vd qwen3_5 mặc định tie=False nhưng Qwen3.5-2B là True; gemma4 mặc định không có vision tower). Resolver phải đọc `config.json` của repo, không suy từ họ. Tên module dưới đây là tên trong lớp `*ForConditionalGeneration` của transformers 5.12.

## Mức kiểm chứng
- **A — đã chạy trên GB10** (nhãn "Đã kiểm chứng" nếu có trong catalog).
- **B — AutoModel có recipe/implementation riêng** (nhãn "Chưa kiểm chứng: engine hỗ trợ chính thức").
- **C — transformers native, chưa thử** (nhãn "Chưa kiểm chứng: chạy được về lý thuyết").
- **X — không đưa vào 1.0** (không phải VLM chat, encoder-decoder, sinh ảnh, hoặc không dựng được).

| model_type | architecture | mức | vision | projector | language | LoRA bỏ qua (exclude) | qlora_skip_modules |
|---|---|---|---|---|---|---|---|
| qwen3_5 | Qwen3_5ForConditionalGeneration | A | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| gemma3 | Gemma3ForConditionalGeneration | B | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| gemma3n | Gemma3nForConditionalGeneration | B | model.vision_tower, model.embed_vision | — | model.language_model | *vision_tower*, *embed_vision*, *lm_head* | model.vision_tower, model.embed_vision |
| gemma4 | Gemma4ForConditionalGeneration | B | model.vision_tower (khi repo có vision_config) | model.embed_vision | model.language_model | *vision_tower*, *embed_vision*, *lm_head* | model.vision_tower, model.embed_vision |
| internvl | InternVLForConditionalGeneration | B | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| llava_onevision | LlavaOnevisionForConditionalGeneration | B | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| minimax_m3_vl | MiniMaxM3SparseForConditionalGeneration | B | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| mistral3 | Mistral3ForConditionalGeneration | B | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| qwen2_5_omni_thinker | Qwen2_5OmniThinkerForConditionalGeneration | B | visual | — | — | *visual*, *lm_head* | visual |
| qwen2_5_vl | Qwen2_5_VLForConditionalGeneration | B | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| qwen3_5_moe | Qwen3_5MoeForConditionalGeneration | B | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| qwen3_omni_moe_thinker | Qwen3OmniMoeThinkerForConditionalGeneration | B | visual | — | — | *visual*, *lm_head* | visual |
| qwen3_vl | Qwen3VLForConditionalGeneration | B | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| qwen3_vl_moe | Qwen3VLMoeForConditionalGeneration | B | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| aria | AriaForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| cohere2_vision | Cohere2VisionForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| cosmos3_omni | Cosmos3OmniForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| deepseek_vl | DeepseekVLForConditionalGeneration | C | model.vision_model | model.aligner | model.language_model | *vision_model*, *aligner*, *lm_head* | model.vision_model, model.aligner |
| deepseek_vl_hybrid | DeepseekVLHybridForConditionalGeneration | C | model.high_res_vision_model, model.high_res_vision_neck, model.high_res_vision_proj, model.vision_model | model.aligner | model.language_model | *high_res_vision_model*, *high_res_vision_neck*, *high_res_vision_proj*, *vision_model*, *aligner*, *lm_head* | model.high_res_vision_model, model.high_res_vision_neck, model.high_res_vision_proj, model.vision_model, model.aligner |
| ernie4_5_vl_moe | Ernie4_5_VLMoeForConditionalGeneration | C | model.vision_tower | model.resampler_model | model.language_model | *vision_tower*, *resampler_model*, *lm_head* | model.vision_tower, model.resampler_model |
| exaone4_5 | Exaone4_5_ForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| fast_vlm | FastVlmForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| gemma4_unified | Gemma4UnifiedForConditionalGeneration | C | model.vision_tower (khi repo có vision_config) | model.embed_vision | model.language_model | *vision_tower*, *embed_vision*, *lm_head* | model.vision_tower, model.embed_vision |
| glm46v | Glm46VForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| glm4v | Glm4vForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| glm4v_moe | Glm4vMoeForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| glm_ocr | GlmOcrForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| glmga | Glm46VForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| got_ocr2 | GotOcr2ForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| idefics2 | Idefics2ForConditionalGeneration | C | model.vision_model | model.connector | model.text_model | *vision_model*, *connector*, *lm_head* | model.vision_model, model.connector |
| idefics3 | Idefics3ForConditionalGeneration | C | ? | ? | ? | đọc từ config repo | đọc từ config repo |
| lfm2_vl | Lfm2VlForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| lighton_ocr | LightOnOcrForConditionalGeneration | C | model.vision_encoder, model.vision_projection | — | model.language_model | *vision_encoder*, *vision_projection*, *lm_head* | model.vision_encoder, model.vision_projection |
| llama4 | Llama4ForConditionalGeneration | C | vision_model | multi_modal_projector | language_model | *vision_model*, *multi_modal_projector*, *lm_head* | vision_model, multi_modal_projector |
| llava | LlavaForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| llava_next | LlavaNextForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| llava_next_video | LlavaNextVideoForConditionalGeneration | C | model.vision_tower, model.vision_resampler | model.multi_modal_projector, model.vision_resampler | model.language_model | *vision_tower*, *vision_resampler*, *multi_modal_projector*, *vision_resampler*, *lm_head* | model.vision_tower, model.vision_resampler, model.multi_modal_projector, model.vision_resampler |
| minicpmv4_6 | MiniCPMV4_6ForConditionalGeneration | C | model.vision_tower | model.merger | model.language_model | *vision_tower*, *merger*, *lm_head* | model.vision_tower, model.merger |
| mllama | MllamaForConditionalGeneration | C | model.vision_model | model.multi_modal_projector | model.language_model | *vision_model*, *multi_modal_projector*, *lm_head* | model.vision_model, model.multi_modal_projector |
| ovis2 | Ovis2ForConditionalGeneration | C | model.vision_tower, model.visual_embeddings_table | — | model.language_model | *vision_tower*, *visual_embeddings_table*, *lm_head* | model.vision_tower, model.visual_embeddings_table |
| paddleocr_vl | PaddleOCRVLForConditionalGeneration | C | model.visual | model.projector | model.language_model | *visual*, *projector*, *lm_head* | model.visual, model.projector |
| paligemma | PaliGemmaForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| pp_chart2table | GotOcr2ForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| qianfan_ocr | QianfanOCRForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| qwen2_vl | Qwen2VLForConditionalGeneration | C | model.visual | — | model.language_model | *visual*, *lm_head* | model.visual |
| shieldgemma2 | Gemma3ForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| smolvlm | SmolVLMForConditionalGeneration | C | ? | ? | ? | đọc từ config repo | đọc từ config repo |
| video_llama_3 | VideoLlama3ForConditionalGeneration | C | model.vision_model | model.projector | model.language_model | *vision_model*, *projector*, *lm_head* | model.vision_model, model.projector |
| video_llava | VideoLlavaForConditionalGeneration | C | — | model.multi_modal_projector | model.language_model | *multi_modal_projector*, *lm_head* | model.multi_modal_projector |
| vipllava | VipLlavaForConditionalGeneration | C | model.vision_tower | model.multi_modal_projector | model.language_model | *vision_tower*, *multi_modal_projector*, *lm_head* | model.vision_tower, model.multi_modal_projector |
| aya_vision | AyaVisionForConditionalGeneration | X | — | — | — | config mặc định lỗi; cần config repo | — |
| blip | BlipForConditionalGeneration | X | — | — | — | captioning cũ, không chat | — |
| blip-2 | Blip2ForConditionalGeneration | X | — | — | — | captioning, không chat template | — |
| chameleon | ChameleonForConditionalGeneration | X | — | — | — | sinh ảnh | — |
| deepseek_ocr2 | DeepseekOcr2ForConditionalGeneration | X | — | — | — | không dựng được, chưa kiểm | — |
| diffusion_gemma | DiffusionGemmaForBlockDiffusion | X | — | — | — | diffusion | — |
| emu3 | Emu3ForConditionalGeneration | X | — | — | — | sinh ảnh | — |
| evolla | EvollaForProteinText2Text | X | — | — | — | protein | — |
| florence2 | Florence2ForConditionalGeneration | X | — | — | — | seq2seq, prompt token đặc thù | — |
| fuyu | FuyuForCausalLM | X | — | — | — | kiến trúc không vision tower riêng, ít dùng | — |
| git | GitForCausalLM | X | — | — | — | captioning | — |
| granite4_vision | Granite4VisionForConditionalGeneration | X | — | — | — | không dựng được với config mặc định | — |
| idefics | IdeficsForVisionText2Text | X | — | — | — | thế hệ cũ (idefics1) | — |
| instructblip | InstructBlipForConditionalGeneration | X | — | — | — | không chat template chuẩn | — |
| instructblipvideo | InstructBlipVideoForConditionalGeneration | X | — | — | — | như trên | — |
| janus | JanusForConditionalGeneration | X | — | — | — | sinh + hiểu ảnh, chưa có recipe | — |
| kosmos-2 | Kosmos2ForConditionalGeneration | X | — | — | — | grounding đặc thù | — |
| kosmos-2.5 | Kosmos2_5ForConditionalGeneration | X | — | — | — | OCR đặc thù | — |
| mistral4 | Mistral4ForCausalLM | X | — | — | — | không có phần vision trong model mặc định | — |
| perception_lm | PerceptionLMForConditionalGeneration | X | — | — | — | không dựng được, chưa kiểm | — |
| pi0 | PI0ForConditionalGeneration | X | — | — | — | robotics | — |
| pix2struct | Pix2StructForConditionalGeneration | X | — | — | — | encoder-decoder, không chat | — |
| pp_formulanet | PPFormulaNetForConditionalGeneration | X | — | — | — | OCR công thức encoder-decoder | — |
| t5gemma2 | T5Gemma2ForConditionalGeneration | X | — | — | — | encoder-decoder | — |
| udop | UdopForConditionalGeneration | X | — | — | — | document encoder-decoder | — |
| vision-encoder-decoder | VisionEncoderDecoderModel | X | — | — | — | encoder-decoder (Donut), engine khác | — |

## Quy tắc chung (cho recipe tự sinh, xem phần 2 khi xong)
- Freeze mặc định: vision tower + projector đóng băng; LoRA chỉ trên language model (khớp template Qwen3.5 đã đo).
- `qlora_skip_modules` = vision + projector (lm_head tied do patch 0006 tự loại).
- Chat template: BẮT BUỘC processor/tokenizer của repo có `chat_template`; không có thì từ chối (không tự đoán template).
- Ảnh: nếu image processor có `min_pixels`/`max_pixels` (họ Qwen) thì đặt 200704/1003520 như catalog; họ khác dùng mặc định của processor (tile/size cố định).
- Model cần `trust_remote_code` (auto_map trong config) ngoài catalog: CHẶN (ADR-021 §3).

## Quyết định 1.0 (research 2026-10-14, sau kiểm CPU trên CORD + patch 0007/0008)
`configs/catalog/families.yaml` có 13 họ. Mức hiển thị: A "đã chạy trên GB10" / S "đã smoke" / B, C "chưa thử".

| họ | mức | recipe | image tối thiểu | ghi chú |
|---|---|---|---|---|
| qwen3_5 | A | qwen3.5-2b | 26.08-7 | R2/R7/QLoRA đo thật |
| qwen3_vl, qwen2_5_vl | B | qwen3-vl-2b-instruct (qwen2_5_collate_fn) | 26.08-7 | AutoModel có example |
| qwen2_vl | C | như trên | 26.08-7 | cùng layout/processor 2.5 |
| gemma4 | B | gemma-4-e2b-it | 26.08-7 | nhãn đúng trên -7 (kiểm CPU) |
| internvl (*-hf) | B | family-default | 26.08-7 | nhãn đúng trên -7; bản gốc OpenGVLab cần remote code → chặn |
| gemma3 | B | family-default (attn eager) | 26.08-8 | |
| llava_onevision | B | family-llava-onevision | 26.08-8 | trên -7: 0 token nhãn (processor render `assistant \n`) |
| mistral3 | B | family-default | 26.08-8 | model công khai nhỏ nhất 24B: RAM +20% |
| idefics3, smolvlm | C | family-default | 26.08-8 | trên -7: 0 token nhãn (template tokenizer không render content chuỗi) |
| llava, llava_next | C | family-default | 26.08-8 | |

Độ dài chuỗi mỗi mẫu CORD (ảnh + prompt + đáp án, processor_default): SmolVLM-256M ~530, InternVL3-1B ~1.7k (12 tile), LLaVA-OV-0.5B ~4.4k (anyres),
Gemma-4-E2B ~230, Qwen3.5-2B (pixel cap) ~560 (`outputs/research/fam/collate_check.py`).

Không đưa vào 1.0 (ngoài nhóm X ở trên), lý do cụ thể:
- mllama: chỉ có bản 11B/90B, gated, ảnh đi qua cross-attention layers (không phải token ảnh) — chưa có collate/nhãn kiểm được; xem lại sau 1.0.
- paligemma: prefix-LM, không có chat template → recipe chat không áp dụng.
- aria, llama4, ernie4_5_vl_moe, qwen3_vl_moe, qwen3_5_moe, glm4v_moe, minimax_m3_vl: MoE ≥ 25B tổng tham số, vượt RAM an toàn (≤ 80% RAM trống) cho LoRA bf16.
- qwen2_5_omni_thinker, qwen3_omni_moe_thinker, cosmos3_omni: omni (audio/video) — ngoài phạm vi ảnh–chữ 1.0.
- glm4v, glm46v, glm_ocr, exaone4_5, paddleocr_vl, deepseek_vl(_hybrid), cohere2_vision, fast_vlm, lfm2_vl, minicpmv4_6,
  ovis2, got_ocr2, lighton_ocr, qianfan_ocr, video_llama_3, video_llava, vipllava, llava_next_video, idefics2, gemma3n, gemma4_unified:
  native (mức C) nhưng chưa kiểm nhãn/collate; thêm từng họ khi có 1 lần chạy collate_check + smoke (quy trình: tải bản nhỏ nhất,
  `collate_check.py`, smoke 20 bước, rồi thêm dòng families.yaml).
