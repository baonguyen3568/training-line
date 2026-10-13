# Security Policy

Vietnamese version (source for the website): [`docs/legal/bao-mat-va-bao-cao-lo-hong.md`](docs/legal/bao-mat-va-bao-cao-lo-hong.md).

## Reporting a vulnerability

Please **do not** open a public issue. Report privately to:
**{{TODO_USER: security contact email or GitHub Private Vulnerability Reporting link (U7)}}**

Include the version or commit, OS and hardware, steps to reproduce (a proof of concept if you have one) and the impact
you expect.

We acknowledge within **3 business days** and give a first assessment within **10 business days**. We ask for
coordinated disclosure: please wait until a fix is released, or **90 days** after your report, whichever comes first.
On request we credit you in the changelog. We will not pursue legal action against good-faith research that stays on
your own machine, does not access other people's data, does not disrupt services, and is reported to us before
disclosure.

### In scope
- The application: local server, web UI, container job runner, workflow and plugin handling, and model download and
  verification steps.
- The project website.

### Out of scope
- Vulnerabilities in third-party software (PyTorch, NVIDIA NeMo, vLLM, Hugging Face libraries, Docker, NVIDIA drivers or
  the container toolkit). Report those upstream; we update pinned versions once a fix ships.
- Attacks that require existing read/write access to the user's machine or account.
- Missing hardening headers without a demonstrated exploit.

## Supported versions

Only the latest release receives security fixes.

## Security design (summary)

- The server binds `127.0.0.1` only. A per-start token is exchanged for an `HttpOnly`, `SameSite=Strict` session
  cookie. The server checks Host and Origin and requires a CSRF token on state-changing requests and WebSocket auth.
  There is no CORS.
- Train, eval and quantize jobs run in containers with `--network none`. Secret-like environment variables (tokens,
  keys) are never forwarded to jobs. Telemetry of bundled libraries is disabled.
- Supply chain:
  - container images are pinned by digest;
  - Python and npm dependencies are pinned with hashes and a 7-day release cooldown;
  - models are pinned to a commit and loaded from `safetensors` only;
  - `trust_remote_code` is limited to reviewed, hash-pinned code.
- Third-party nodes run in isolated containers without network, with declared permissions. Unreviewed nodes are
  blocked by default on shared hosts.
- No telemetry. See the privacy policy (`docs/legal/chinh-sach-quyen-rieng-tu.md`).
