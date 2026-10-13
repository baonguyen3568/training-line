# shellcheck shell=bash
# LAN edge helpers (ADR-019), sourced by scripts/install.sh and scripts/qa/nginx_edge_smoke.sh (same code tested).
# Callers validate every value first (scripts/lib_lan.py): names are [0-9a-z.-], the subnet a CIDR, the port digits.

# make_lan_tls <tls_dir> <server names...>: a local CA with X.509 Name Constraints (once), then a server certificate
# (<= 397 days, renewed 30 days before its end). Even a leaked CA key cannot vouch for any other site on the devices
# that trust it. A changed name list needs a new CA (devices re-install it), so this refuses instead of widening.
make_lan_tls() {
  local dir="$1"; shift
  local names_now="$* localhost 127.0.0.1" constraints="critical" san="" n
  if [[ -f "$dir/ca.crt" && "$(cat "$dir/names" 2>/dev/null)" != "$names_now" ]]; then
    printf 'ERROR: the LAN address or name changed since the certificate authority was made: move %s away and re-run (then install the new ca.crt on your devices)\n' "$dir" >&2
    return 1
  fi
  mkdir -p "$dir"
  chmod 700 "$dir"
  for n in "$@"; do
    if [[ "$n" =~ ^[0-9.]+$ ]]; then
      constraints+=",permitted;IP:$n/255.255.255.255"; san+="${san:+,}IP:$n"
    else
      constraints+=",permitted;DNS:$n"; san+="${san:+,}DNS:$n"
    fi
  done
  constraints+=",permitted;DNS:localhost,permitted;IP:127.0.0.1/255.255.255.255"
  if [[ ! -f "$dir/ca.crt" ]]; then
    (umask 077
     openssl req -x509 -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -sha256 -days 3650 \
       -keyout "$dir/ca.key" -out "$dir/ca.crt" -subj "/CN=NTC-Training local CA ($1)" \
       -addext "basicConstraints=critical,CA:TRUE,pathlen:0" -addext "keyUsage=critical,keyCertSign,cRLSign" \
       -addext "nameConstraints=$constraints" 2>/dev/null) || return 1
    printf '%s' "$names_now" > "$dir/names"
    chmod 644 "$dir/ca.crt" "$dir/names"
  fi
  if [[ -f "$dir/server.crt" ]] && openssl x509 -checkend 2592000 -noout -in "$dir/server.crt" >/dev/null; then
    return 0
  fi
  (umask 077
   openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -sha256 \
     -keyout "$dir/server.key.new" -out "$dir/server.csr" -subj "/CN=$1" 2>/dev/null || exit 1
   printf '%s\n' "basicConstraints=critical,CA:FALSE" "keyUsage=critical,digitalSignature" \
     "extendedKeyUsage=serverAuth" "subjectAltName=$san" "subjectKeyIdentifier=hash" \
     "authorityKeyIdentifier=keyid" > "$dir/server.ext"
   openssl x509 -req -in "$dir/server.csr" -CA "$dir/ca.crt" -CAkey "$dir/ca.key" \
     -set_serial "0x$(openssl rand -hex 16)" -days 397 -sha256 -extfile "$dir/server.ext" \
     -out "$dir/server.crt.new" 2>/dev/null || exit 1
   rm -f "$dir/server.csr" "$dir/server.ext"
   # OpenSSL enforces the name constraints here: a certificate outside them is never installed.
   openssl verify -CAfile "$dir/ca.crt" "$dir/server.crt.new" >/dev/null || exit 1
   mv -f "$dir/server.key.new" "$dir/server.key"
   mv -f "$dir/server.crt.new" "$dir/server.crt") || {
    printf 'ERROR: could not make a server certificate that verifies against the local CA\n' >&2
    rm -f "$dir/server.key.new" "$dir/server.crt.new" "$dir/server.csr" "$dir/server.ext"
    return 1
  }
  chmod 644 "$dir/server.crt"
}

# render_nginx_conf <template> <out> <server names> <allow cidr> <port>: the reviewed template, three placeholders.
render_nginx_conf() {
  local template="$1" out="$2" names="$3" allow="$4" port="$5"
  [[ "$names" =~ ^[0-9a-z.\ -]+$ && "$allow" =~ ^[0-9./]+$ && "$port" =~ ^[0-9]+$ ]] || {
    printf 'ERROR: unexpected nginx values\n' >&2; return 1; }
  mkdir -p "$(dirname "$out")"
  (umask 022
   sed -e "s|@SERVER_NAMES@|$names|" -e "s|@LAN_ALLOW@|$allow|" -e "s|@LAN_PORT@|$port|g" "$template" > "$out.new" \
     && mv -f "$out.new" "$out")
}
