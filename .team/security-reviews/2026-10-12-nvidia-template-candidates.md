# SEC-REVIEW: model NVIDIA cho template (yêu cầu user 2026-10-12, danh sách từ P001a)

Reviewer: bao-security · 2026-10-12 · Phương pháp: HF API metadata tại SHA, đọc README/license gốc, đọc `config.json` **kể cả sub-config**, tải riêng các file `.py` (không tải weights, không chạy), grep red flag và đọc tay các call site.

## Bài học chung (đã thêm vào skill security-model-vetting)
`auto_map` bắc cầu sang repo khác (`org/repo--module.Class`) **luôn tải code theo `main`**: transformers 5.12.1 không kế thừa `revision` cho repo khác. Bằng chứng thực tế: `main` của `nvidia/C-RADIOv2-H` có commit "Update hf_model.py" ngày **2026-01-30**, sau khi Parse/Nano-VL đã phát hành. Nghĩa là load các model này hôm nay sẽ chạy code khác với code lúc NVIDIA phát hành. Cách pin đã kiểm chứng: xem hồ sơ Nemotron-Omni (§Cập nhật).

| # | Model @SHA | License | Remote code | Verdict |
|---|---|---|---|---|
| 1 | nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561 (4.9GB, 1 safetensors) | NVIDIA Open Model **License** (bản 10/2025, có điều khoản **guardrail-termination**: "bypass, disable, reduce the efficacy of… safety guardrail… without a substantially similar Guardrail → rights terminate"). base_model Qwen3-VL-2B-Instruct (Apache-2.0). commercial=yes | 0 file .py. **Chưa đọc được config.json** (gated=auto, trả 401), nên chưa loại trừ được auto_map bắc cầu | **CONDITIONAL**: chờ đọc config (U7). Nếu config là `qwen3_vl` native, không có auto_map → APPROVED |
| 2 | nvidia/Cosmos-Reason2-8B@a9fae2cf89dc64db96b12860417f0eb403013bb9 (17.5GB) | như #1 | như #1 | **CONDITIONAL** như #1 |
| 3 | nvidia/NVIDIA-Nemotron-Parse-2.0@b6742064f4a8cf22a10383ece5e7fbead355ac04 (3.6GB) | **OpenMDW-1.1** (dễ dãi: không hạn chế sử dụng, không share-alike; khi phân phối phải giữ bản license + notice; chấm dứt nếu kiện về patent/copyright). Tokenizer: **CC-BY-4.0**. Code vision encoder lấy từ C-RADIOv2-H: NVIDIA Open Model License (guardrail clause) | 12 .py, `auto_map` gốc trong repo. **`encoder.auto_map` → `nvidia/C-RADIOv2-H--hf_model.*` (bắc cầu, research chưa thấy vì nằm trong sub-config)**. `hf_nemotron_parse_modeling.py:400` hardcode `trust_remote_code=True`. Các file example/test/vllm_example không được import khi load | **APPROVED-WITH-CONDITIONS** (template 2): pin C-RADIOv2-H@0d8f4c18 bằng thư mục local đã hash (vá `encoder.auto_map`), hoặc dùng class native của AutoModel **nếu chứng minh được** đường native không tải remote code (test: HF_HUB_OFFLINE=1, không có HF modules cache → vẫn load được) |
| 4 | nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD@16853d6c682223e967d5622c99d0082cb994b771 (10.6GB, serve-only) + bản BF16@ca9543b1, FP8@e9550fde | NVIDIA Open Model License (guardrail clause) | 11 .py. `vision_config.auto_map` → C-RADIOv2-H (bắc cầu) | **Chưa review code** (ưu tiên 4). Áp cùng cách pin C-RADIOv2-H. vLLM ≥0.31.0 |

**C-RADIOv2-H@0d8f4c18c877166eda07ddae1386bcad256b7a6a**: 21 .py, 5325 dòng, chỉ có safetensors (không có .pth.tar như v4). License NVIDIA Open Model License (card trỏ bản 06/2024). Red flag giống C-RADIOv4-H: `torch.load`/`torch.hub.load`/open_clip/PaliGemma `from_pretrained` nằm trong các đường adaptor/pretrained, không đi tới khi `adaptor_names` rỗng (kiểm lại trong `encoder` config của Parse). Không có exec/eval/subprocess/socket.

**license_obligations đề xuất:**
- Cosmos-Reason2: `attribution_notice` ("Built on NVIDIA Cosmos" + "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"), `no_guardrail_bypass`, Apache-2.0 NOTICE của Qwen3-VL (upstream).
- Parse 2.0: `attribution_notice` (OpenMDW-1.1 + notices), `cc_by_attribution` (tokenizer), `no_guardrail_bypass` (code RADIO, nếu dùng đường remote code).

**Lưu ý sản phẩm (không phải security):** Cosmos-Reason2 là repo **gated**. Nếu làm template mặc định, mọi end-user phải có tài khoản HF, tự bấm accept license trên web và cấp token cho app trước khi chạy được template đầu tiên. Đây là ma sát lớn với người không biết code.

### sha256 file .py đã tải
```
# nvidia/NVIDIA-Nemotron-Parse-2.0@b6742064f4a8cf22a10383ece5e7fbead355ac04
6f258fd539041711ee5cf5ab506eb697f3a2a70a92e3ccefee97df654b7988ab  ./example_with_processor.py
d86dfd5621541cf2b3ad506e76576445c89eefc3903cc150fca4e9bbdad2d179  ./hf_logits_processor.py
60dfa11643701f4cadd704984a84dddb3f8a9269d64cb866da1f1608d9e5ae76  ./hf_nemotron_parse_config.py
339042c1065e168149ace17e8be116fbd5982f26d684c5819232c77083ef95f8  ./hf_nemotron_parse_modeling.py
57dab2ba78d76adbe9912fe75dfc402b025aa604d994ac57cfe02a87300e2df5  ./hf_nemotron_parse_processor.py
97631b0f5dfac795aa1018237bee25fdbb57917056fba78053d84f00232baada  ./latex2html.py
9dd64780489a522251ccf05ea09f7659d202aaef54aee3a03186130e5e338508  ./logitsprocs/nemotron_parse_vllm_logitprocs.py
28d25377301c1b1604da15076fe994f266234387884d665c303f07e760b12f14  ./postprocessing.py
8bcf3e0754701447c3dbbbf09519769d449d08f0c8e137683e1d1299d90719be  ./run_transformers_inference.py
f3712c7ab26ad7e3bb2b900d48f16dff740c5640ba2f742658e45a15531c0de3  ./test_golden.py
f2893f7dc735e6c06472bf740c385f3c0ea6edd38f348c616bb0e95b924417c8  ./vllm_example.py
9ead577c1a8fb661b4d6951552e36559f0b3d73c0b96bad2c032c88f2f83e32f  ./vllm_tied_patch/sitecustomize.py
# nvidia/C-RADIOv2-H@0d8f4c18c877166eda07ddae1386bcad256b7a6a
85b4c1d0a6d0c8c5ff9fdc8fb3afd7ee2be65407ee36600d6fedd5eabc4b7346  ./adaptor_base.py
938444bbb6c8beafd6b5c5faa51567847f9be4a696929dbcd67f6cd6e4ea67e9  ./adaptor_generic.py
0c720adfa8cddddae36b439c297756a4cd27066e2bdef8b648c862fca4bd8be4  ./adaptor_mlp.py
16963e07de300e5f3cbd5e343aed111d04e4e55a50e07f50ccc2ab930883ea1e  ./adaptor_registry.py
64ee35f128380c3bb9bc1a913e98ac328220be88964f2092686c637ef866fc79  ./cls_token.py
ca54bdc1dc2d6a091e5a28709a5be374f17a508d7c319b79e2f9609fe3741f03  ./common.py
5433d08522c28f46a69e710da1a86fc1d76fc03562dbb9b2b49baf5db21195d3  ./dinov2_arch.py
86853ae2ed28c902529cd9b78201350e7dba1cf54ea9e57578241ecc0d88d867  ./dual_hybrid_vit.py
f585ab4ee09da408f953f7080ed7380298aad63e51f7af00ad0de6e494a1c185  ./enable_cpe_support.py
b8e703400604ee5f9cdfac0de9b857b7f5126aad8bba03604b04f52f7641d6cb  ./enable_spectral_reparam.py
886bdce8f21a254c258c79aa50f16e6c2e67e7953576e84427a38a7175342b57  ./eradio_model.py
71e28c3e350a53347c6d6abde204338c34762d37d238cce8615d2b9a474be216  ./extra_models.py
e78cfb44e528b0228ed51515d8e2f759f5f83fd59f5324fffdbd3484922a3c95  ./extra_timm_models.py
a74401f5220c2e5960d39addce9ba40f481efca9ab44bb9d96535ed23e1ef62b  ./feature_normalizer.py
eb5a562d11edc655876eeb95bc1c3b4c7484b963dae41a3f76f7acd2a6906b9e  ./forward_intermediates.py
7120e573ddbf6e58e24cab4da01ea5c2213b18a565b3a09c1ba1f44f38c9a3d4  ./hf_model.py
72062d3b71877e1469e069cf090e5f35d6ad7abaeb451e56502f7f32dc6c1075  ./input_conditioner.py
d6b3569d03db1e1175df09745005f3d7443885f3208c50a563a3efbbaf28d0b0  ./open_clip_adaptor.py
ab5decb89b49b4a2c68b9a769738ad7e58a54e7515840977cbed6e5e6a457773  ./radio_model.py
5b83b4c7b90b797d854a6da2fd4093ce9210a52533d3f7ba6bf505793b6c68c3  ./vitdet.py
591a6ae511fa38c56ee3bea70dce16ebd7854a52be8191a5749b9682c01bbb88  ./vit_patch_generator.py
```

## Cập nhật: danh sách mới của research (Parse-2.0 bị loại vì lỗi kỹ thuật)
| # | Model @SHA | License | Remote code | Verdict |
|---|---|---|---|---|
| 5 | **Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203** (4.3GB) | Apache-2.0 (tag trên card của org Qwen chính chủ; repo không có file LICENSE riêng) | 0 .py, `qwen3_vl` native, không có auto_map ở mọi cấp | **APPROVED** (baseline N-4 / G1-harness). Cũng là base của Cosmos-Reason2-2B |
| 6 | **nvidia/NVIDIA-Nemotron-Parse-v1.1@fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4** (3.8GB) | NVIDIA Open Model License (bản có guardrail-termination), tokenizer CC-BY-4.0 theo card. commercial=yes | 9 .py (2499 dòng). **Có bắc cầu** (research lại bỏ sót): `encoder._name_or_path = nvidia/C-RADIOv2-H`, `encoder.auto_map → nvidia/C-RADIOv2-H--hf_model.*`. `hf_nemotron_parse_config.py:113` tự gọi `get_class_from_dynamic_module(*auto_map.split("--")[::-1])` → tải RADIOConfig từ `main`. `modeling.py:388` hardcode `trust_remote_code=True`. Native AutoModel 26.08 cũng hardcode fallback `"nvidia/C-RADIOv2-H"` + trust_remote_code=True (bao-backend xác minh). Ngoài ra không có exec/eval/subprocess/socket/pickle/network | **APPROVED-WITH-CONDITIONS**: bước prepare vá cả `encoder.auto_map` lẫn `encoder._name_or_path` sang thư mục local C-RADIOv2-H@0d8f4c18 đã verify hash (đường dẫn không được chứa `--`). Chạy `--network none` + `HF_HUB_OFFLINE=1`. Test build-from-config offline (backend) phải pass trước khi đưa vào template. Deps import: albumentations, cv2, einops, bs4 (bs4 chỉ dùng trong postprocessing) |

Nemotron-Parse-2.0: **LOẠI** (lý do kỹ thuật theo backend/research; về security đã APPROVED-WITH-CONDITIONS ở trên, giữ lại hồ sơ).
```
# nvidia/NVIDIA-Nemotron-Parse-v1.1@fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4
c498f89f73fe80ee7e59f83b55a150a53d21691b3493a3d17fb39da7c36d69e1  example.py
69e1da5f008acf4bb57267adb7226107f064ba45812c35a1526a69dc1382e0a0  hf_nemotron_parse_config.py
679200695e747ec16b8cfd061e289c960b4aadbc327a9c1ebaf9f864b1bd9925  hf_nemotron_parse_modeling.py
b316a5a34d04e8d4dadcddb9bb258b1ce08b258725368a41dd747ae42a716ef3  hf_nemotron_parse_processor.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  __init__.py
5513826743fab907f4afa03ea317ed6f91b83ce68493b98b978cc89491bf49f8  latex2html.py
96327e2eb9becc54a690a07ed01584dcd37b8b1cdb2f1d216ba3d1f5cbb2f1d4  postprocessing.py
d07f0e6c15ff50802e4496cabb1b815d2e88448978f5f7ef56376a9d4ca81944  test_golden.py
6c230c5f8c27acb1aa726ba7f31efcbbd28185cfca12aeab9f8ba83dceda57ad  test_vllm_golden.py
```

## Cập nhật: gate P-0 cho Parse-v1.1 — PASS (bao-backend, commit 80f0efd)
Container `vlm-automodel:26.08-1`, `--network none`, `HF_HUB_OFFLINE=1`, HF modules cache rỗng lúc đầu, non-root, cap-drop ALL. Build-from-config (meta device) thành công, 936M params. Log: `outputs/probe/p0-parse-build.log`.
Đường code thực tế chạy:
- config: NemotronParseConfig, remote, nạp từ bản copy đã verify hash;
- model: class **native** của NeMo AutoModel;
- encoder RADIO: remote, nạp từ bản pin `c_hyphen_radiov2_hyphen_h_0d8f4c18` (`radio_from_pinned_copy: true`). Không có tham chiếu nào tới `nvidia/C-RADIOv2-H`.

Security đã kiểm: 2 file `configs/models/pinned/*.sha256` **giống hệt** bảng hash của review này. `prepare.py` từ chối `.py` chưa vet, file thiếu hoặc hash lệch; copy file (không dùng symlink) cho `.py`; vá `auto_map` + `_name_or_path`; còn bất kỳ tham chiếu `org/repo--` nào thì từ chối.
**Verdict: Parse-v1.1 APPROVED cho template mặc định**, kèm điều kiện còn lại:
(1) `_copy_py` phải verify hash **trên bản copy** sau khi copy (hiện đang verify nguồn rồi mới copy → TOCTOU nhỏ);
(2) P-1 xác nhận processor (`hf_nemotron_parse_processor.py`) cũng nạp từ bản đã prepare (dùng cùng assert module path), và load weights thật không có missing key ở encoder;
(3) chạy với vLLM ≥0.31.0 khi serve.

## Cập nhật: Parse-2.0 dùng lại (user quyết định tự vá AutoModel) — xác nhận lại theo SHA mới nhất
- `nvidia/NVIDIA-Nemotron-Parse-2.0` main vẫn là **b6742064** (lastModified 2026-08-22). `nvidia/C-RADIOv2-H` main vẫn là **0d8f4c18**. Cả hai **không đổi** so với lúc review, nên verdict APPROVED-WITH-CONDITIONS giữ nguyên. `encoder.adaptor_names = None`, nên các đường torch.load/torch.hub của RADIO không được gọi tới.
- Hash pinned: `configs/models/pinned/nemotron-parse-2.0@b6742064.sha256` (top-level .py) + `…vllm-extras.sha256` (subdir).
- **Phát hiện mới:** repo có `vllm_tied_patch/sitecustomize.py` (48 dòng). Nội dung **lành tính**: monkey-patch `NemotronParseForConditionalGeneration.load_weights` của vLLM để tie `lm_head` với `decoder.embed_tokens`. Không có I/O hay network. Nhưng README hướng dẫn `export PYTHONPATH=<thư mục tải về>/vllm_tied_patch` → mọi interpreter Python trên host sẽ tự chạy code tải từ HF. **Quy tắc:** vendor file vào `third_party/patches/vllm/`, verify hash, chỉ mount ro vào container vLLM serve; tuyệt đối không đưa thư mục HF cache vào PYTHONPATH. Áp tương tự cho `logitsprocs/`.
