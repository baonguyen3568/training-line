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
# The ids reach compose interpolation (mount options of the egress-control tmpfs): digits only.
[[ "$(id -u)" =~ ^[0-9]+$ && "$(id -g)" =~ ^[0-9]+$ ]] || fail "unexpected user or group id"
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
PORT="${PORT:-4001}"  # D-8: project servers live on 127.0.0.1:4001-4050
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

# ------------------------------------------------------------------------------------------------ LAN (optional)
# ADR-019: set VLM_LAN_BIND in .env.local (this machine's LAN IPv4) to open the app to the LAN through nginx + TLS.
# Unset: nothing listens beyond 127.0.0.1. The CA key never leaves ~/.config/ntc-training/tls (never mounted).
LAN_BIND="$(setting VLM_LAN_BIND)"
LAN_PORT="$(setting VLM_LAN_PORT)"; LAN_PORT="${LAN_PORT:-4000}"
EDGE_SUBNET="$(setting VLM_EDGE_SUBNET)"; EDGE_SUBNET="${EDGE_SUBNET:-10.231.47.0/29}"
TLS_DIR="$CONFIG_DIR/tls"
LAN_DIR="$CONFIG_DIR/lan"  # public files only (nginx.conf, the CA certificate): mounted read-only into the app
LAN_CONF="$LAN_DIR/nginx.conf"
PROXY_PEER=""
PUBLIC_ORIGINS=""
command -v python3 >/dev/null || fail "python3 is needed to check the network settings"
lan_value() { sed -n "s/^${1}=\(.*\)$/\1/p" <<<"$2"; }
# The app always joins the nginx<->app "edge" network: its subnet must not overlap any network of this machine.
edge_out="$(python3 -I "$ROOT/scripts/lib_lan.py" edge "$EDGE_SUBNET")" || fail "set another VLM_EDGE_SUBNET in .env.local"
EDGE_IP_RANGE="$(lan_value EDGE_IP_RANGE "$edge_out")"
# The download jobs' network (security download-egress): fixed, so the proxy listens and accepts on it only.
EGRESS_SUBNET="$(setting VLM_EGRESS_SUBNET)"; EGRESS_SUBNET="${EGRESS_SUBNET:-10.231.48.0/28}"
egress_out="$(python3 -I "$ROOT/scripts/lib_lan.py" edge "$EGRESS_SUBNET" "$EDGE_SUBNET")" \
  || fail "set another VLM_EGRESS_SUBNET in .env.local"
EGRESS_IP_RANGE="$(lan_value EDGE_IP_RANGE "$egress_out")"
EGRESS_PROXY_IP="$(lan_value PROXY_PEER "$egress_out")"
if [[ -n "$LAN_BIND" ]]; then
  command -v openssl >/dev/null || fail "openssl is needed for the LAN certificate (package openssl)"
  lan_out="$(python3 -I "$ROOT/scripts/lib_lan.py" lan "$LAN_BIND" "$LAN_PORT" "$(setting VLM_LAN_ALLOW)" \
    "$(hostname -s)")" || fail "fix the LAN settings in .env.local (VLM_LAN_*), or remove VLM_LAN_BIND"
  LAN_ALLOW="$(lan_value LAN_ALLOW "$lan_out")"
  PROXY_PEER="$(lan_value PROXY_PEER "$edge_out")"
  SERVER_NAMES="$(lan_value SERVER_NAMES "$lan_out")"
  PUBLIC_ORIGINS="$(lan_value PUBLIC_ORIGINS "$lan_out")"
  ours_lan="$(docker ps --filter label=com.docker.compose.project=ntc-training --format '{{.Ports}}' 2>/dev/null || true)"
  if (exec 3<>"/dev/tcp/$LAN_BIND/$LAN_PORT") 2>/dev/null && [[ "$ours_lan" != *"$LAN_BIND:$LAN_PORT->"* ]]; then
    fail "port $LAN_BIND:$LAN_PORT is already used by another program: set another VLM_LAN_PORT in .env.local"
  fi
  if (( DRY_RUN )); then
    say "[dry-run] LAN: CA + server certificate in $TLS_DIR for: $SERVER_NAMES"
    say "[dry-run] LAN: nginx on https://$LAN_BIND:$LAN_PORT (allow $LAN_ALLOW), config $LAN_CONF"
  else
    # shellcheck source=scripts/lib_lan_tls.sh
    source "$ROOT/scripts/lib_lan_tls.sh"
    new_ca=0; [[ -f "$TLS_DIR/ca.crt" ]] || new_ca=1
    # shellcheck disable=SC2086  # SERVER_NAMES is a validated, space-separated list
    make_lan_tls "$TLS_DIR" $SERVER_NAMES || fail "the LAN certificate could not be made"
    (( ! new_ca )) || say "created a local certificate authority: $TLS_DIR/ca.crt (install it on the devices that open the app)"
    render_nginx_conf "$ROOT/docker/nginx/nginx.conf.template" "$LAN_CONF" "$SERVER_NAMES" "$LAN_ALLOW" "$LAN_PORT" \
      || fail "could not write $LAN_CONF"
    install -m 644 "$TLS_DIR/ca.crt" "$LAN_DIR/ca.crt"  # the public certificate only: Settings offers it
  fi
fi

run mkdir -p "$LAN_DIR"  # always: the app mounts it (empty while LAN access is off)

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
VLM_EDGE_SUBNET=$EDGE_SUBNET
VLM_EDGE_IP_RANGE=$EDGE_IP_RANGE
VLM_EGRESS_SUBNET=$EGRESS_SUBNET
VLM_EGRESS_IP_RANGE=$EGRESS_IP_RANGE
VLM_EGRESS_PROXY_IP=$EGRESS_PROXY_IP
VLM_PUBLIC_ORIGINS=$PUBLIC_ORIGINS
VLM_PROXY_PEER=$PROXY_PEER
VLM_LAN_DIR=$LAN_DIR
EOF
  if [[ -n "$LAN_BIND" ]]; then
    printf '%s\n' "VLM_LAN_BIND=$LAN_BIND" "VLM_LAN_PORT=$LAN_PORT" "VLM_LAN_CONF=$LAN_CONF" "VLM_TLS_DIR=$TLS_DIR"
  fi
}
if (( DRY_RUN )); then say "[dry-run] write $COMPOSE_ENV:"; compose_env; else
  (umask 077; compose_env > "$COMPOSE_ENV")
fi

# ------------------------------------------------------------------------------------------------ build + start
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/compose.yaml" --env-file "$COMPOSE_ENV")
[[ -z "$LAN_BIND" ]] || COMPOSE+=(--profile lan)
# The port must be free, or already ours (a re-run): another app on it would get the browser's requests.
ours="$(docker ps --filter label=com.docker.compose.project=ntc-training --format '{{.Ports}}' 2>/dev/null || true)"
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && [[ "$ours" != *":$PORT->"* ]]; then
  fail "port $PORT is already used by another program: set another VLM_PORT in .env.local"
fi
run "${COMPOSE[@]}" build
# Upgrade: the proxy's control-socket volume is now a tmpfs owned by the app user. One made by an older install is
# root's (the proxy cannot bind its socket there): it only ever holds that socket, so it is recreated.
EGRESS_VOL=ntc-training_egress-control
if docker volume inspect "$EGRESS_VOL" >/dev/null 2>&1 \
  && [[ "$(docker volume inspect "$EGRESS_VOL" --format '{{index .Options "type"}}' 2>/dev/null)" != "tmpfs" ]]; then
  run "${COMPOSE[@]}" rm --stop --force egress-proxy runner
  run docker volume rm "$EGRESS_VOL"
fi
run "${COMPOSE[@]}" up -d
# A re-render replaces nginx.conf / the certificate files (new inodes): restart so nginx reads them.
[[ -z "$LAN_BIND" ]] || run "${COMPOSE[@]}" restart nginx
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
say "Open the link printed by:  ${COMPOSE[*]} logs app | grep 'open http'"
if [[ -n "$LAN_BIND" ]]; then
  say "On the LAN: https://$LAN_BIND:$LAN_PORT/ (each device must trust $TLS_DIR/ca.crt first)."
  say "Anyone with the link controls this project: share it only with people you trust (Settings: sign out everywhere)."
fi
