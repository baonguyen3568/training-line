#!/usr/bin/env bash
# NTC-Training one-command install (Release 1.0 A4, ADR-015): checks the host, prepares the folders, the key and the
# policy file, builds the two images and starts the stack. Safe to run again: nothing that exists is overwritten.
#   scripts/install.sh              install / update and start
#   scripts/install.sh --dry-run    print what it would do, change nothing
#   scripts/install.sh --skip-download   do not offer to fetch the default template's model + dataset
# Never run as root: the app, the runner and every job run as YOUR user id (security F2).
set -euo pipefail

DRY_RUN=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
ROOT="${VLM_INSTALL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
CONFIG_DIR="${HOME}/.config/ntc-training"

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
run() { if (( DRY_RUN )); then say "[dry-run] $*"; else "$@"; fi; }

# ------------------------------------------------------------------------------------------------ host checks
[[ "$(id -u)" -ne 0 ]] || fail "do not run as root (or with sudo): run as the user who will use the app"
command -v docker >/dev/null || fail "docker is not installed"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 is missing (package docker-compose-plugin)"
[[ -S /var/run/docker.sock ]] || fail "/var/run/docker.sock not found: is the docker daemon running?"
docker info >/dev/null 2>&1 || fail "this user cannot use docker: add it to the 'docker' group, then log in again"
if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  say "WARNING: no 'nvidia' docker runtime: training jobs need the NVIDIA Container Toolkit"
fi
[[ -f "$ROOT/compose.yaml" && -f "$ROOT/pyproject.toml" ]] || fail "$ROOT is not the NTC-Training folder"

# ------------------------------------------------------------------------------------------------ settings file
# .env.local holds infrastructure settings (and later the secrets compose hands to the runner). Created once from
# the example; only VLM_DATA_ROOT / VLM_PORT / VLM_PROFILE are read here.
ENV_LOCAL="$ROOT/.env.local"
if [[ ! -e "$ENV_LOCAL" ]]; then
  run install -m 600 "$ROOT/.env.local.example" "$ENV_LOCAL"
  say "created .env.local (edit it to change the port or the data folder)"
fi
setting() {  # value of one KEY=value line, never the whole file into the environment
  [[ -f "$ENV_LOCAL" ]] || return 0
  sed -n "s/^${1}=\(.*\)$/\1/p" "$ENV_LOCAL" | tail -n 1
}
DATA_ROOT="$(setting VLM_DATA_ROOT)"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
PORT="$(setting VLM_PORT)"
PORT="${PORT:-7870}"
PROFILE="$(setting VLM_PROFILE)"
# These values reach .env.compose, which compose interpolates ($, spaces, ':' would change a mount): strict checks,
# the same rules the app applies to a data root (common/app_config.check_data_root).
check_data_root() {
  local d="$1"
  [[ "$d" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "VLM_DATA_ROOT must be an absolute path of letters, digits, . _ - / (got: $d)"
  [[ "$d/" != *"/../"* && "$d/" != *"/./"* && "$d" != */ ]] || fail "VLM_DATA_ROOT must not contain . or .. parts"
  [[ "$d" != "$HOME" && "$HOME/" != "$d/"* ]] || fail "VLM_DATA_ROOT must not be your home folder or a parent of it"
  local sys
  for sys in /etc /usr /bin /sbin /lib /lib64 /boot /proc /sys /dev /run /var/run; do
    [[ "$d" != "$sys" && "$d" != "$sys/"* ]] || fail "VLM_DATA_ROOT must not be under $sys"
  done
}
check_data_root "$DATA_ROOT"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || fail "VLM_PORT must be 1024..65535 (got: $PORT)"
case "$PROFILE" in ""|personal|shared_dev) ;; *) fail "VLM_PROFILE must be personal or shared_dev (got: $PROFILE)" ;; esac

# ------------------------------------------------------------------------------------------------ folders
for d in outputs .cache data uploads; do run mkdir -p "$DATA_ROOT/$d"; done
# The GPU lock the host scripts use too (one lock per machine): the team's .team/lock on a dev checkout, else the
# data root's own system folder (a user install keeps it out of the team coordination files).
if [[ -d "$ROOT/.team/lock" ]]; then LOCK_DIR="$ROOT/.team/lock"; else LOCK_DIR="$DATA_ROOT/system/lock"; fi
run mkdir -p "$LOCK_DIR"

# Download tokens (security download-egress §2, S1): one 0400 file per running job, on the host's tmpfs (gone at
# reboot, never in a backup or the project folder); the data root's gitignored system/ only if there is no tmpfs.
if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR}" ]]; then
  SECRETS_DIR="$XDG_RUNTIME_DIR/ntc-training/runner-secrets"
else
  SECRETS_DIR="$DATA_ROOT/system/runner-secrets"
fi
[[ "$SECRETS_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "unexpected secrets folder: $SECRETS_DIR"
run mkdir -p "$SECRETS_DIR"
run chmod 700 "$SECRETS_DIR"

# ------------------------------------------------------------------------------------------------ key + policy
# secrets.key: 32 random bytes, 0600, created once (the app container mounts it read-only and cannot create it).
# policy.toml: the admin policy (may stay empty); mounted read-only, never written by the app.
run mkdir -p "$CONFIG_DIR"
run chmod 700 "$CONFIG_DIR"
if [[ ! -e "$CONFIG_DIR/secrets.key" ]]; then
  if (( DRY_RUN )); then say "[dry-run] create $CONFIG_DIR/secrets.key (32 random bytes, 0600)"; else
    (umask 077; set -o noclobber; head -c 32 /dev/urandom > "$CONFIG_DIR/secrets.key")
  fi
fi
if [[ ! -e "$CONFIG_DIR/policy.toml" ]]; then
  if (( DRY_RUN )); then say "[dry-run] create $CONFIG_DIR/policy.toml (empty admin policy, 0644)"; else
    (umask 022; set -o noclobber
     printf '# NTC-Training admin policy (optional). See docs: [deployment], [storage].\n' > "$CONFIG_DIR/policy.toml")
  fi
fi

# ------------------------------------------------------------------------------------------------ compose env
# Only computed, non-secret values: compose reads them with --env-file; nothing here is a password or token.
COMPOSE_ENV="$ROOT/.env.compose"
COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
compose_env() {
  cat <<EOF
# Generated by scripts/install.sh: re-run it instead of editing this file.
VLM_UID=$(id -u)
VLM_GID=$(id -g)
DOCKER_GID=$(stat -c %g /var/run/docker.sock)
VLM_PORT=$PORT
VLM_PROFILE=$PROFILE
VLM_HOST_PROJECT_ROOT=$ROOT
VLM_DATA_ROOT=$DATA_ROOT
VLM_GPU_LOCK_DIR=$LOCK_DIR
VLM_POLICY_FILE=$CONFIG_DIR/policy.toml
VLM_SECRETS_KEY_FILE=$CONFIG_DIR/secrets.key
VLM_RUNNER_SECRETS=$SECRETS_DIR
VLM_BUILD_COMMIT=$COMMIT
EOF
}
if (( DRY_RUN )); then say "[dry-run] write $COMPOSE_ENV:"; compose_env; else
  (umask 077; compose_env > "$COMPOSE_ENV")
fi

# ------------------------------------------------------------------------------------------------ build + start
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/compose.yaml" --env-file "$COMPOSE_ENV")
# The port must be free, or already ours (a re-run): another app on it would get the browser's requests.
ours="$(docker ps --filter label=com.docker.compose.project=ntc-training --format '{{.Ports}}' 2>/dev/null || true)"
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && [[ "$ours" != *":$PORT->"* ]]; then
  fail "port $PORT is already used by another program: set another VLM_PORT in .env.local"
fi
run "${COMPOSE[@]}" build
run "${COMPOSE[@]}" up -d
# The default template's model + dataset, through THE app's download path (vlm-runner, egress proxy, verify, install).
if (( ! SKIP_DOWNLOAD )); then
  answer="y"
  large=()
  if [[ -t 0 ]]; then
    read -r -p "Download the default template's model and data now (a few GB)? [Y/n] " answer || true
    [[ "${answer:-y}" =~ ^[Nn] ]] || large=(--yes-large)  # typed at a terminal: big files confirmed too
  fi
  if [[ ! "${answer:-y}" =~ ^[Nn] ]]; then  # no terminal: never confirms a download over the threshold (security P1)
    run "${COMPOSE[@]}" exec -T app python -m vlm_pipeline.app.downloads --template invoice_to_json "${large[@]}" \
      || say "WARNING: the download did not finish; the app offers it again (\"Download\" on the step)."
  fi
fi
say ""
say "NTC-Training is starting on http://127.0.0.1:$PORT/"
say "Open the link printed by:  ${COMPOSE[*]} logs app | grep 'http://127.0.0.1'"
