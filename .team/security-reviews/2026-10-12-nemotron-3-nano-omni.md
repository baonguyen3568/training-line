# SEC-REVIEW: remote code — nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

- Reviewer: bao-security · Ngày: 2026-10-12 · Phương pháp: tải **chỉ file .py** (không tải weights, không thực thi), grep red flag rồi đọc tay các call site.
- Repo chính: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e9932441de940c9a62185c870ea5bcd4cd24e2`, 13 file .py, 3961 dòng. License: NVIDIA Open Model Agreement (v. 2026-03-09), cho phép thương mại.
- **Phụ thuộc remote code bắc cầu**: `config.json → vision_config.auto_map = nvidia/C-RADIOv4-H--hf_model.RADIOModel`. Đây là code tải từ **repo thứ 2**, KHÔNG có revision, tức mặc định lấy `main`. `nvidia/C-RADIOv4-H@0057b339059c0b9e1b4ba996f975410ebbfdfcc8`: 26 file .py, 5763 dòng, có thêm `c-radio_v4-h_half.pth.tar` (pickle). License: NVIDIA Open Model License **bản 06/2024** (bản cũ, cần đọc riêng).

## Kết luận: APPROVED-WITH-CONDITIONS — chỉ cho team R&D. KHÔNG đưa vào catalog app v1.

### Phát hiện
| # | Mức | File | Vấn đề |
|---|---|---|---|
| N1 | **HIGH** | Omni `config.json` (vision_config.auto_map), `modeling.py:97` `AutoModel.from_config(..., trust_remote_code=True)` | Code RADIO được tải từ repo khác theo `main`. Pin SHA của Omni **không** pin được phần này. Ai kiểm soát `nvidia/C-RADIOv4-H` main là chạy được code trên máy mình. `trust_remote_code=True` bị hardcode trong code, không tắt được bằng tham số |
| N2 | MEDIUM | Omni `video_io.py:126-140` | Input `http(s)://…mp4` thì `urllib.request.urlretrieve` không timeout, không allowlist host → SSRF / treo / ghi file tạm lớn. Input `data:video/mp4;base64` thì decode ra file tạm không giới hạn kích thước. Đường local path đọc **bất kỳ file nào** process đọc được rồi base64 hoá |
| N3 | LOW (không đi tới với config hiện tại) | C-RADIO `eradio_model.py:1319-1393`, `enable_spectral_reparam.py:237` (`torch.load` không weights_only), `extra_models.py:105` (`torch.hub.load`), `siglip2_adaptor.py:37` (`trust_remote_code=True`), `open_clip_adaptor.py:25` | Chỉ chạy khi có `pretrained`/`adaptor_names`. Config Omni hiện có `adaptor_names: None`, `adaptor_configs: {}`, và weights lấy từ safetensors của Omni. Nếu config đổi thì các đường này sẽ được kích hoạt |
| — | OK | Cả hai repo | Không có `os.system/subprocess/eval/exec/socket/requests`. Không đọc `*_TOKEN`. Không có network lúc import (ngoài N2) |

### Điều kiện sử dụng
1. Tải trước **cả hai** repo ở đúng SHA (Omni chỉ lấy safetensors + .py + json; C-RADIO chỉ lấy .py + json, **loại `*.pth.tar`**). Mọi lần load chạy với `HF_HUB_OFFLINE=1`, để `main` của C-RADIO resolve về bản cache đã pin. Load lần đầu với `--network none`.
2. Trước mỗi lần load, kiểm sha256 của các .py trong HF modules cache khớp bảng dưới (test (c) tầng integration).
3. Video input: **chỉ file local nằm trong workspace của run**. Wrapper của mình chặn `http(s)://` và `data:`, và kiểm path bằng realpath.
4. Serve: vLLM **≥ 0.31.0** (GHSA-h3rc-6mm3-gc2m), không expose `/tokenize`, bind 127.0.0.1.
5. Trước khi đưa vào catalog app: đọc license C-RADIO (06/2024), tìm cách thay N1 bằng vendored module (tự sửa `auto_map` trỏ local) và giải quyết N2. Khi SHA mới ra thì review lại diff.

### sha256 các file đã review
```
# nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e9932441de940c9a62185c870ea5bcd4cd24e2
a3bbb1e79c002f92c2695a6f4718cf8af93cef7b1f60e30ffc0b4f5c5c3ddd79  audio_model.py
13dc1b61b72d624e6415b523fc038c3312788600d7ef2e45477fb951c259944c  configuration_nemotron_h.py
f13df0f4ed641c968c52a61f0ba46f6408fbfccf57fea5dc5ef9fafd76d8d245  configuration.py
27f16c9dccc06b28435e75d6b97877ab7cd2e5809029039b0c84e0b8f8602bb7  configuration_radio.py
79f0659f3532b5ebd941e497c4ccd0808879e5487673e6f9387ca29f6d789cd8  evs.py
360fd99b157c802ada9d752716e05c5d8f01fa9f2ef2514224c706599cec9d03  image_processing.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  __init__.py
046b199c440b6a2b0c7f71d2174dd8820ff94810e210c4a10ca1aaeaf4a71b93  modeling_nemotron_h.py
a666aeaa23ab98571412496772254f10aff84464b714819b8e11fa10a36f76b2  modeling.py
823b55ccbb472353032cb3b3a849d3ea77c3a47287af074caad995e3a6cc51e2  processing.py
f107d30e96bf9ff949fbf71b2429601d4971e51c8cbd8f27ab73b82f05d2ce59  processing_utils.py
05e2100cc8ba3b603ef81293a32d8854cea07e77c67bd32264f1f2b6bafc2667  video_io.py
9cc25c47e6d896439aff41faab505b3d2f4fe56afb9bb5b52121a589cc5163a8  video_processing.py
# nvidia/C-RADIOv4-H@0057b339059c0b9e1b4ba996f975410ebbfdfcc8
baa112b58158d75077e9bba8baba51b2c75f6a6deed2d8046aa5e5f2d5cb669a  adaptor_attn.py
f9d285cc76e70f977f9c78e748f610765f0bd068a13f86d804c07c765cece577  adaptor_base.py
3d1b5a015ff3754a9eb64cd56b4d04cae01e537a1ffa606adaf904c55fff07a1  adaptor_generic.py
e75a9ed1f7f58c749dec5a510d699f36f32fdb7f3b61cd8a7e3ccd5afd2c9b77  adaptor_mlp.py
2beac17dc5824c7d9f925dc782cdd4b49bd15cf206bf1cbaa9108a8966e33316  adaptor_module_factory.py
16963e07de300e5f3cbd5e343aed111d04e4e55a50e07f50ccc2ab930883ea1e  adaptor_registry.py
64ee35f128380c3bb9bc1a913e98ac328220be88964f2092686c637ef866fc79  cls_token.py
6cc79e1f7b61f4917602d2efc9e0a833fa74d60563f6aac74e4625de23723bab  common.py
5433d08522c28f46a69e710da1a86fc1d76fc03562dbb9b2b49baf5db21195d3  dinov2_arch.py
86853ae2ed28c902529cd9b78201350e7dba1cf54ea9e57578241ecc0d88d867  dual_hybrid_vit.py
3c1e3df656960f5c83b38ba5f632fc5b335edf1d6de67731396a865aaa5ce1ed  enable_cpe_support.py
7e2a0b6f8934ac6e4a4768e5ca84cb4ff09ba3726a470f1cf046bd5169b2e334  enable_damp.py
b8e703400604ee5f9cdfac0de9b857b7f5126aad8bba03604b04f52f7641d6cb  enable_spectral_reparam.py
bf76c60b991813eef45c76246cd7dcee3fefbf39bde3d2b6ac8576067012d05f  eradio_model.py
a938b96715266d2f65325e56d0d8e42465d23e944e905d2820023472785c55a9  extra_models.py
d7364d5abf5b617782366948f0e593d7a2df275b076094afa1ebc5276a9f7102  extra_timm_models.py
a74401f5220c2e5960d39addce9ba40f481efca9ab44bb9d96535ed23e1ef62b  feature_normalizer.py
eb5a562d11edc655876eeb95bc1c3b4c7484b963dae41a3f76f7acd2a6906b9e  forward_intermediates.py
d28c0a0238ca8854a9af5a9f003761332bfc9182601b3b7466d9242be91ee96f  hf_model.py
72062d3b71877e1469e069cf090e5f35d6ad7abaeb451e56502f7f32dc6c1075  input_conditioner.py
c2eb9166de48f5eb5f90f8a750e8ebe67da93698e794dc0bbdab5f013bab8e66  open_clip_adaptor.py
2be659f5cd5b9db9c49db70387b65253c74bc16b3eb86e976f0f693271d4b183  radio_model.py
1f40f7d77beefd69ad86aac61b12c82ed915b4012e6bea1628fc8abe80782851  siglip2_adaptor.py
3b83163ce2650ef04973538dd79716ab015d24344ea25d52cfb0c37b093ba4ba  utils.py
5b83b4c7b90b797d854a6da2fd4093ce9210a52533d3f7ba6bf505793b6c68c3  vitdet.py
092e9a0b8eca4fee162484ef3ef77b399f686e5e0ffc7be85fb2d69bc2f9ebd9  vit_patch_generator.py
```
