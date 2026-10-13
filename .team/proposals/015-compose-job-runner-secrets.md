# Proposal 015 — Compose: app tạo job container an toàn + secrets `.env.local` (M3 D-1/D-3, phần security)

Tác giả: bao-security · 2026-10-13 · Trạng thái: DRAFT, chờ bao-backend (đồng tác giả D-1/D-3) và PM chốt ADR
Liên quan: P003 D1/D4/D13/D16/D17, ADR-005, `src/vlm_pipeline/common/jobrun.py`

## 1. Vấn đề
Khi app chạy trong container, nó vẫn phải tạo job container (train/eval/quantize/serve). Mount `/var/run/docker.sock`
vào app tương đương **root trên host**, trong khi máy đang chạy kafka, elastic, banking-rag, comfy-ui. App là thành phần
có bề mặt tấn công lớn nhất của dự án: web UI, upload ảnh, workflow file từ người khác, plugin, chuỗi tokenizer…
Chỉ một lỗi RCE trong app là đủ chiếm cả máy.

## 2. Các phương án

| | A. socket-proxy lọc endpoint (tecnativa/docker-socket-proxy) | B. socket-proxy lọc **body** (tự viết hoặc authz plugin) | **C. job-runner sidecar với API kiểu hoá (đề xuất)** |
|---|---|---|---|
| Cách làm | HAProxy chỉ mở `CONTAINERS`, `POST` | Proxy parse JSON `POST /containers/create`, chặn field nguy hiểm | App gửi `JobSpec` (kiểu hoá, đóng) qua unix socket. Runner tự dựng `docker run` bằng code sẵn có (`docker_args`) |
| Chặn `Privileged`, `Binds: ["/:/host"]`, `PidMode: host` | **Không**: lọc theo URL, không theo body. `POST /containers/create` vẫn có privileged → vẫn root | Có, nếu liệt kê đủ. Docker API có hàng chục field nguy hiểm (`Binds`, `Mounts`, `VolumesFrom`, `Devices`, `DeviceCgroupRules`, `CapAdd`, `SecurityOpt`, `UsernsMode`, `CgroupnsMode`, `Sysctls`, `Runtime`, `CgroupParent`…) và mỗi version API lại thêm → denylist dễ sót | App **không có** đường nào chạm vào Docker API. Bề mặt chỉ là schema của ta |
| `exec` vào container khác (banking-rag) | Chặn nếu tắt `EXEC`, nhưng `CONTAINERS=1` vẫn cho stop/kill/rm **mọi** container | Phải tự lọc thêm theo tên/label | Runner chỉ thao tác container có label `project=vlm-training` + `vlm.owner=app`, tên `vlm-job-<run_id>` |
| Sửa daemon / sudo | Không | Authz plugin cần sửa `daemon.json` → **cấm** (ảnh hưởng mọi service) | Không |
| Công sức | Thấp | Cao, dễ sai | Vừa: tái dùng `jobrun.docker_args` (đã có kiểm env secret, `--network none`, `--cap-drop ALL`, image pin SEC-004) |

**Kết luận:** chọn **C**. A **không đạt** yêu cầu (vẫn tạo được privileged container). B là denylist trên một API lớn và đổi theo
version. C biến bài toán thành allowlist trên schema của chính ta.

## 3. Thiết kế C

```
 browser ──127.0.0.1:7870──▶ [vlm-app]  (non-root, read_only, cap_drop ALL, KHÔNG có docker.sock, KHÔNG có secrets)
                                 │ unix socket /run/vlm-runner/runner.sock (volume chung, 0660, gid riêng)
                                 ▼
                            [vlm-runner] (stdlib Python, ~vài trăm dòng, KHÔNG có port, KHÔNG có network ngoài)
                                 │ /var/run/docker.sock (chỉ runner có)
                                 ▼
                           docker daemon ──▶ vlm-job-<run_id>  (--network none, non-root, cap_drop ALL, mem limit)
```

### 3.1 API của runner (đóng, JSON, giới hạn 64 KB/request)
| Lệnh | Tham số app được gửi | Runner tự quyết |
|---|---|---|
| `submit` | `run_id` (`^[0-9]{8}-[0-9]{4}-[a-z0-9-]{1,48}$`), `job_kind` (enum: train / eval / quantize / probe / download / serve), `image_tag` (phải có trong IMAGES.lock, `is_security_approved`, image id khớp — SEC-004), `argv` (danh sách module/args theo allowlist của `job_kind`), `gpu: bool`, `memory_gb` (≤ trần cấu hình), `lock_token` | name `vlm-job-<run_id>`, labels, user, mounts, network, cap, env, GPU device |
| `stop` / `kill` | `run_id` | chỉ khi container tồn tại **và** có đủ 2 label của ta |
| `status` | `run_id` | `docker inspect` State (đã lọc field) |
| `logs` | `run_id`, `since`, `tail` ≤ 10 000 | stream stdout/stderr, đã redact token (§4.4) |
| `remove` | `run_id` | chỉ container đã exit và có label của ta |

Không có lệnh nào nhận path, mount, cờ docker, tên image tự do, biến env tự do, hay tên container tuỳ ý.
**Không** có `exec`, `pull`, `build`, `image rm`, `volume`, `network`. Pull image chỉ làm tay (PM/user) theo D1.

### 3.2 Mount do runner tính (thu hẹp so với hiện tại)
Hiện `docker_args` mount **cả project root** vào job (rw khi `read_only_project=False`). Với compose, như vậy job (kể cả
plugin chưa duyệt) đọc được `.env.local` nếu file nằm trong project root, và ghi được vào code của app. Đề xuất:
- `outputs/runs/<run_id>/` hoặc `outputs/graph/runs/<node_dir>/` → **rw** (chỉ thư mục của job đó);
- `data/` + `.cache/prepared/` + HF hub cache → **ro**;
- code `src/` → **ro** (overlay như hiện tại);
- **không bao giờ** mount: `.env*`, `.team/`, `~/.config/ntc-training/` (trừ đúng file policy.toml, bind ro), `/var/run/docker.sock`, `$HOME`.
- Đường dẫn nguồn là **host path**. Runner nhận `VLM_HOST_PROJECT_ROOT` từ `.env.local` (cấu hình triển khai, không lấy
  từ app). Runner resolve, đối chiếu dưới root đó, từ chối symlink ra ngoài.
- **Symlink ở source bind mount (docker daemon theo symlink trên host):** runner mount data_root + src (ro) vào chính
  nó ở đúng path host; trước mỗi bind đi từng component bằng O_NOFOLLOW, có symlink thì từ chối; source phải tồn tại,
  đúng loại, owner = user. Run dir rw do runner tạo mới ngay trước job; input chỉ ro. Giả định TOCTOU: không job đang
  chạy nào có quyền ghi vào thư mục cha của một source khác (có test). (security, review host_paths 2026-10-13)
- `download` là job_kind duy nhất có mạng (network `vlm-egress`, không có port) và là job duy nhất nhận token (§4.3).

### 3.3 Cứng hoá runner (vì runner = root-equivalent)
- Chỉ dùng stdlib (không FastAPI, không pydantic), có test riêng, review bắt buộc bởi security cho mọi thay đổi.
- Image runner build từ cùng base pin digest, `USER` non-root + `group_add: [<gid của docker.sock>]`, `read_only: true`,
  `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`, `mem_limit: 256m`, **không** `ports`, network `none`
  (unix socket không cần network).
- Socket `runner.sock` 0660 với một gid riêng `vlm-runner`. Chỉ app được `group_add` gid đó.
- Defense in depth (khuyến nghị, không bắt buộc v1): runner nói chuyện với docker qua `tecnativa/docker-socket-proxy`
  (`CONTAINERS=1 POST=1 EXEC=0 BUILD=0 IMAGES=0 VOLUMES=0 NETWORKS=0 SWARM=0 …`). Nếu runner bị chiếm thì vẫn không có
  `exec`/`build`. Proxy này phải pin digest và qua review image như mọi image.
- Mọi request ghi audit `outputs/system/runner.jsonl` (lệnh, run_id, image id, kết quả; không ghi env).

### 3.4 GPU lock, memguard
Giữ nguyên `.team/GPU_LOCK` (file). Runner kiểm `lock_token` khớp lock hiện tại trước khi `submit` một job có `gpu=true`.
Memguard của fixcode chạy trong runner hoặc trên host như hiện tại: D-4 quyết.

## 4. Secrets — `.env.local` (D-3)
### 4.1 Vị trí và quyền
- Một file `.env.local` ở **thư mục deploy** (cạnh `compose.yaml`), quyền 0600. Hiện thư mục deploy = project root, nên
  chỉ an toàn khi §3.2 đã bỏ mount cả project root. **Điều kiện tiên quyết**: không job nào được mount thư mục chứa `.env.local`.
- `.env.local.example` được commit: đủ key, value rỗng hoặc mặc định an toàn, comment tiếng Việt. Không có giá trị thật.
- `.gitignore` có `.env.*` (PM đã thêm) nhưng phải **un-ignore** `!.env.local.example`. `.dockerignore` của MỌI image
  phải có `.env*`, `**/.env*`, `.git`, `outputs/`, `data/`, `.cache/`.

### 4.2 Không vào image / layer
- Dockerfile không có `ARG`/`ENV` chứa `*TOKEN*|*KEY*|*SECRET*|*PASSWORD*` (gate trong security-pr-review §2, thêm grep
  vào `scripts/check.sh`).
- Build không cần secret. Nếu sau này cần thì dùng `RUN --mount=type=secret`.
- Kiểm sau build: `docker history --no-trunc` + `docker image inspect` không chứa giá trị token. Dùng `dive` hoặc tar
  layer + grep `hf_` khi review image.

### 4.3 Runtime: token chỉ tới đúng chỗ cần
| Service | Nhận gì |
|---|---|
| vlm-app | **không** token nào. Cấu hình không-bí-mật (port, profile, quota) qua `environment:` |
| vlm-runner | `HF_TOKEN`, `NGC_API_KEY` qua compose `secrets:` (nguồn `environment:` từ `.env.local`), mount `/run/secrets/*` 0400. **Không** qua `environment:` (sẽ lộ trong `docker inspect` và `/proc/1/environ`) |
| job `download` | runner mount riêng file `/run/secrets/hf_token` ro + `HF_TOKEN_PATH=/run/secrets/hf_token`. Không truyền `-e HF_TOKEN=` |
| mọi job khác | không có (giữ `_SECRET_NAME` deny hiện tại) |

Token phải là **Read** (U3, D12): runner gọi `whoami-v2` của HF lúc khởi động. Token có quyền write thì log cảnh báo
và từ chối job download (fail-closed, giống D12 Simple mode).

### 4.4 Log
- App và runner không bao giờ log `os.environ`, nội dung `/run/secrets`, hay header Authorization.
- Runner redact trong stream log: `hf_[A-Za-z0-9]{30,}`, `nvapi-[A-Za-z0-9_-]{20,}`, và giá trị chính xác của token
  đang nạp → `***`.
- `docker compose config` in ra env: hướng dẫn trong README cấm dán output đó vào issue/chat.

### 4.5 Secrets chỉnh qua trang Cài đặt (D-7, PM 2026-10-13)
Config phân tầng: mặc định → `.env.local` → DB (`outputs/app.db`, bảng settings), chỉnh qua UI. Yêu cầu security:
- **Phân loại field** ngay trong schema settings: `public` (port, ngôn ngữ, quota cá nhân…), `secret` (HF_TOKEN, NGC_API_KEY,
  WANDB_API_KEY), `admin_policy` (`deployment.profile`, approval policy/threshold, plugin allowlist, schedule apply).
  - `admin_policy` **không có** API ghi và không hiện như ô sửa trong UI. Chỉ đọc từ `~/.config/ntc-training/policy.toml`
    (D17). Env/DB chỉ được siết. API nhận key `admin_policy` thì 403, không lưu gì.
- **Secret trong DB phải mã hoá**: AEAD (AES-256-GCM, `cryptography` 50.0.2 — duyệt 2026-10-13, xem §4.6; ghi chú cũ "đã qua review" là SAI, chưa từng có review, hoặc libsodium secretbox),
  nonce ngẫu nhiên mỗi lần ghi, AAD = tên field (chống tráo ô).
  - Khoá 32 byte ở `~/.config/ntc-training/secrets.key` (home theo pwd, KHÔNG có env override; file 0600, sinh lần đầu bằng
    `secrets.token_bytes`). Không đặt khoá trong `.env.local` (app chỉ đọc khoá hạ tầng từ file đó).
  - Khoá **không** nằm trong DB, image hay thư mục mà job mount.
  - Mất khoá thì secret không đọc được: UI báo "nhập lại token", không crash.
- **Chỉ ghi, không đọc lại**: GET settings trả `{"set": true, "updated_at": …, "last4": "…ab12"}` (tối đa 4 ký tự cuối)
  cho field secret, không bao giờ trả giá trị. PUT nhận giá trị mới; DELETE xoá. Không có endpoint "reveal".
- **Kiểm token khi ghi**: HF token phải là Read (`whoami-v2`). Token write/fine-grained có quyền ghi thì từ chối kèm
  thông báo dễ hiểu (D12). Lời gọi kiểm này do runner thực hiện, vì app không có egress.
- **Ai giải mã**: chỉ runner (job `download`) cần plaintext. Đề xuất: app lưu ciphertext, runner giữ khoá và giải mã lúc
  submit download. App **không** có khoá, nên app bị chiếm cũng không đọc được token. Nếu backend thấy quá phức tạp cho
  v1: app có khoá nhưng plaintext chỉ tồn tại trong bộ nhớ lúc chuyển cho runner, không log, không cache.
- **Audit log** `outputs/system/settings_audit.jsonl`, append-only: `ts, field, action (set|delete|rejected), actor
  (session id rút gọn), client (127.0.0.1)`. Không ghi giá trị cũ hay mới của secret, chỉ ghi `last4` mới. Field public
  thì ghi giá trị cũ → mới.
- **Auth**: settings API nằm sau `SecurityMiddleware` (D13: Host/Origin, cookie session, CSRF cho PUT/DELETE). Không có
  ngoại lệ, không CORS. Body ≤ 16 KB. Key lạ thì 422 (schema đóng).
- **Test bắt buộc**: GET không bao giờ chứa giá trị secret (property test với token ngẫu nhiên); DB dump (`sqlite3 .dump`)
  không chứa plaintext; PUT `deployment.profile` → 403; PUT không có CSRF → 403; audit có dòng cho mỗi lần đổi.

## 5. Compose (D-1, phần security)
- `ports: ["127.0.0.1:${VLM_PORT:-7870}:7870"]` cho app; site/docs tĩnh cũng `127.0.0.1`. Không service nào bind `0.0.0.0`.
  Có test: `docker compose config` → mọi `published` có `host_ip: 127.0.0.1`.
- Không `network_mode: host`, `pid: host`, `ipc: host`, `privileged`. Có `cap_drop: [ALL]` + `no-new-privileges` cho
  mọi service, `read_only: true` + `tmpfs: /tmp` cho app/runner.
- Tên `vlm-*`, label `project=vlm-training`; `name: vlm` cho project compose. `docker compose down` không đụng service khác.
- File admin `~/.config/ntc-training/policy.toml` (D17) được mount **ro** từ host vào home của user trong container app và runner. App không bao giờ ghi file này. Reader lấy home bằng `pwd.getpwuid(os.getuid())`, không dùng `$HOME`.
- vLLM serve là một `job_kind=serve` của runner (port publish `127.0.0.1` + `--api-key` sinh mỗi lần, app giữ trong
  bộ nhớ), không phải service compose thường trực (tránh giữ ~GB RAM trên UMA).
- `deploy.resources` / `mem_limit` cho mọi service (UMA dùng chung với service khác).
- amd64/arm64: không ảnh hưởng thiết kế. Mọi image pin bằng **manifest list digest** + ghi digest từng arch trong
  IMAGES.lock. Chưa kiểm được artifact arch nào thì ghi `UNVERIFIED <arch>`.

## 6. Test bắt buộc (security gate cho D-1/D-3)
1. Runner từ chối: `run_id` sai regex; image không APPROVED; image id lệch; `job_kind` lạ; argv ngoài allowlist; request > 64 KB;
   stop/kill/remove container không có label của ta (tạo 1 container `vlm-test-foreign` không label trong test integration).
2. `docker inspect vlm-job-*`: `Privileged=false`, `NetworkMode=none` (trừ download), `CapDrop=[ALL]`,
   không có Binds nào ngoài allowlist, không có `.env.local`.
3. App container: không có `/var/run/docker.sock`, không có `/run/secrets`, `env` không có `*TOKEN*`.
4. Job thường: `cat /proc/1/environ` không có token; `ls /workspace/.env.local` → không tồn tại.
5. `docker history` của mọi image không chứa giá trị token (chạy với token giả `hf_TESTTOKEN…` đặt trong `.env.local`
   lúc build).
6. Compose: mọi port `127.0.0.1`; gitleaks chạy trên repo (đã có) + `.env.local.example` không có value thật.

### 6.1 Test hồi quy bắt buộc vì token có quyền ghi (user chấp nhận rủi ro, PM 2026-10-13, U3 đóng)
User giữ token hiện tại (có quyền write, có thể phát sinh chi phí). Vì vậy 4 bảo vệ sau là bắt buộc, mỗi cái có test hồi quy:
| Bảo vệ | Test | Owner |
|---|---|---|
| Chỉ job `download` first-party nhận token (file `/run/secrets`, không qua env) | runner: `submit` job_kind khác + yêu cầu token → từ chối; `docker inspect` job thường không có `HF_TOKEN*` | backend |
| `.env*` bị che trong mọi job (SEC-005) | `test_jobrun` mask + integration `test -s /workspace/.env` fail | backend (đã có) |
| Không có token trong outputs/log/UI | `scripts/security/scan_tokens.py paths outputs/` chạy sau mỗi e2e/integration; e2e UI assert response không khớp pattern | tester |
| Settings D-7 chỉ ghi (last4) | property test GET settings + `sqlite3 .dump` không chứa plaintext | backend |
| Không có token trong file tracked | CI gate `scan_tokens.py tracked` trong `scripts/check.sh` (bổ sung gitleaks) | security (đã có) |

## 7. Câu hỏi mở cho backend / PM
- Q1: Runner viết bằng Python stdlib (tái dùng `jobrun.docker_args`) hay Go nhỏ? Mình nghiêng Python vì tái dùng code đã review.
- Q2: Chế độ dev hiện tại (app chạy trên host, gọi `docker` trực tiếp) giữ lại song song? Đề xuất: giữ cho dev, nhưng app
  trên host cũng chuyển dần sang gọi runner để chỉ còn 1 đường tạo container.
- Q3: Nơi đặt `.env.local` khi user cài đặt thật (ngoài repo, vd `~/.local/share/vlm-app/`)? Ảnh hưởng §3.2.

### 7.1 Trả lời của backend (2026-10-13)
- **Q1: Python stdlib.** Đồng ý. Runner tái dùng nguyên `jobrun.docker_args` / `validate_rw_dirs` / `classify_exit` /
  memguard / `check_pinned_image` (đã review, có test). API trên unix socket chỉ gồm các thao tác kiểu hoá
  (`create_job(spec)`, `stop(name)`, `inspect(name)`, `logs(name, tail)`), trong đó `spec` là một pydantic model
  strict (extra="forbid"): image phải là một TAG có trong IMAGES.lock, argv là list, không có trường mount/privileged/
  network. Runner tự dựng argv bằng `docker_args`, không bao giờ nhận argv docker từ app. Package `vlm_pipeline.runner`
  nằm ở layer common/app như một service riêng; image runner = python slim + docker CLI (buildx 2 arch).
- **Q2: giữ chế độ dev, nhưng chỉ còn 1 đường tạo container.** `run_job` sẽ gọi qua một interface `JobLauncher`:
  `LocalDockerLauncher` (dev trên host, như hiện nay) và `RunnerLauncher` (compose, gọi socket). Cả hai dùng chung
  `docker_args`, nên luật mount/secret/image chỉ có một nguồn. Mặc định dùng runner khi có `VLM_RUNNER_SOCKET`.
- **Q3: `.env.local` ngoài repo khi cài thật**: `${XDG_CONFIG_HOME:-~/.config}/ntc-training/.env.local` (0600), còn
  dữ liệu ở `${XDG_DATA_HOME:-~/.local/share}/ntc-training/{cache,outputs,data,uploads}`. Compose mount hai chỗ này;
  project root trong container chỉ còn code (ro). Ở chế độ dev, file `.env.local` ở gốc repo vẫn được hỗ trợ và luôn
  bị mask trong job (54955d8). Việc thu hẹp mount (§3.2) dùng đúng tách biệt này: job chỉ thấy `cache` (ro, trừ
  thư mục HF modules), `outputs/<run>` (rw), và dữ liệu đầu vào cụ thể (ro).

### 7.2 Security chấp nhận 7.1, kèm 5 điều kiện (2026-10-13)
1. **App image không có docker CLI và không có docker.sock.** Trong compose, `LocalDockerLauncher` phải fail ngay (không
   có đường fallback). Có test: app container gọi `run_job` khi thiếu `VLM_RUNNER_SOCKET` → lỗi rõ ràng, không thử `docker`.
2. **`.env.local` không mount vào service nào.** Compose đọc nó trên host (`env_file` cho biến KHÔNG bí mật của runner,
   `secrets:` cho token). Thư mục `~/.config/<app>/` **không** được bind vào app, vì nếu bind cả thư mục thì app thấy luôn
   `.env.local`. App chỉ nhận đúng file `policy.toml` (bind file, ro). Runner nhận token qua `/run/secrets`.
3. **Đường dẫn policy không đi theo env.** `.env.local` và dữ liệu được dùng `XDG_*` (tiện cho người dùng). Nhưng
   `policy.toml` (D17) luôn đọc từ `pwd.getpwuid(uid).pw_dir/.config/<app>/policy.toml`, không theo `XDG_CONFIG_HOME`
   hay `HOME`, vì env không được phép nới policy (cùng lý do với admin_schedule). Tên thư mục `<app>` là MỘT hằng số
   dùng chung, đổi theo tên sản phẩm khi user chọn (M2-1). Không để "ntc-training" và "vlm-app" cùng tồn tại.
4. **Gốc xoá theo data root.** Khi dữ liệu chuyển sang `~/.local/share/<app>/`, `check_project_root` / retention / cache
   phải đổi sang data root đó, kèm cùng các kiểm resolve/symlink. CLAUDE.md hiện ghi "chỉ xoá trong repo", nên PM cập nhật
   quy tắc khi ADR chốt (cleanup làm).
5. **Spec pydantic strict trong runner: OK** (pydantic đã qua review dep). Runner image = python slim + docker CLI: pin
   digest, non-root + group docker gid, read_only, cap_drop ALL, không port; review image như mọi image (IMAGES.lock).

### 4.6 Chốt D-7 (security ↔ backend, 2026-10-13)
- **Dep:** `cryptography==50.0.2` (pyca, Apache-2.0 OR BSD-3-Clause, wheel abi3 manylinux aarch64 + x86_64, phát hành
  2026-09-30, qua cooldown 7 ngày). Đã có các bản sửa 2026: CVE-2026-69247 (oracle PKCS#7, sửa ở 50.0.0),
  CVE-2026-69248/69249 (X.509 path building, sửa ở 49.0.0), OpenSSL trong wheel (GHSA-537c-gmf6-5ccf). Chỉ dùng
  `cryptography.hazmat.primitives.ciphers.aead.AESGCM`. Không chọn PyNaCl (thêm libsodium, không cần).
  Ghi chú: image 26.08-6 có sẵn cryptography 49.0.0 (bị CVE-2026-69247 ảnh hưởng, nhưng ta không dùng PKCS#7 trong
  job); nâng khi rebuild image.
- **Khoá v1 do app giữ** (runner chưa có):
  - file `~/.config/ntc-training/secrets.key`: tạo bằng O_CREAT|O_EXCL|O_NOFOLLOW, mode 0600, thư mục 0700;
  - đọc bằng cùng kiểm của `read_policy` (O_NOFOLLOW + fstat: regular, owner = user, không g/o-readable).
    Sai thì từ chối dùng secret, KHÔNG tự sinh khoá mới đè lên;
  - không env override đường dẫn; khoá không bao giờ vào DB, log, job mount hay build context;
  - mất khoá → các secret hiện có thành "cần nhập lại".
- **Compose:** khi app chạy trong container, `secrets.key` do bước cài đặt tạo trên host và bind đúng 1 file (ro).
  App trong container không tạo khoá. Khi có runner thì chuyển khoá sang runner, schema DB giữ nguyên
  (ciphertext, nonce 12 B, AAD = tên field, key_id).
- App tạo `secrets.key` lúc user chạy là hành vi sản phẩm. Agent không tạo hay đọc file này. Test dùng tmp path.
