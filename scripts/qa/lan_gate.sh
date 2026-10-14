#!/usr/bin/env bash
# P019 §7 LAN gate, the part that runs ON this machine (read-only: no restart, no config change). The checks from
# another LAN device are a separate manual checklist (bao-tester).
#   scripts/qa/lan_gate.sh <evidence.json> [LAN_IP] [LAN_PORT]
# Hits the LAN IP itself (the listener is bound there), never 127.0.0.1 + --resolve. No secret is ever used: the
# token-in-URL check sends a dummy token, and logs are only COUNTED (never printed). Exit != 0 when a check fails.
set -uo pipefail

OUT="${1:?usage: lan_gate.sh <evidence.json> [LAN_IP] [LAN_PORT]}"
IP="${2:-192.168.2.178}"
PORT="${3:-4000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CA="${VLM_GATE_CA:-$HOME/.config/ntc-training/tls/ca.crt}"
NGINX="${VLM_GATE_NGINX:-ntc-training-nginx-1}"
APP="${VLM_GATE_APP:-ntc-training-app-1}"
URL="https://$IP:$PORT"
ROWS="$(mktemp)"
trap 'rm -f "$ROWS"' EXIT
fails=0

# record <name> <expected> <actual> <command shown>: PASS when actual == expected.
record() {
  local verdict=PASS
  [[ "$3" == "$2" ]] || { verdict=FAIL; fails=$((fails + 1)); }
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$verdict" "$2" "$3" "$4" >> "$ROWS"
  printf '%-4s %-28s expected=%s actual=%s\n' "$verdict" "$1" "$2" "$3"
}
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$@" 2>/dev/null || true; }
curl_exit() { curl -s -o /dev/null --max-time 8 "$@" >/dev/null 2>&1; echo "$?"; }

# --- listeners (ss): every non-loopback listener is listed; only the stack's ports (4000-4050) can fail.
LISTEN="$(ss -ltnH 2>/dev/null | awk '{print $4}' | sort -u)"
NONLOOP="$(grep -vE '^(127\.|\[::1\]|::1:)' <<<"$LISTEN" || true)"
ours_bad="$(awk -F: -v ip="$IP" -v port="$PORT" '{p=$NF; a=substr($0,1,length($0)-length(p)-1)}
  p>=4000 && p<=4050 && !(a==ip && p==port) {print}' <<<"$NONLOOP" | paste -sd, -)"
record "lan_listener" "yes" "$(grep -qxF "$IP:$PORT" <<<"$LISTEN" && echo yes || echo no)" "ss -ltn | $IP:$PORT"
record "stack_ports_lan_only" "" "$ours_bad" "ss -ltn: 4000-4050 not on loopback, other than $IP:$PORT"
record "not_on_wildcard" "no" "$(grep -qE "^(0\.0\.0\.0|\*|\[::\]):$PORT$" <<<"$LISTEN" && echo yes || echo no)" \
  "ss -ltn | 0.0.0.0/*/[::]:$PORT"
record "loopback_refused" "7" "$(curl_exit "http://127.0.0.1:$PORT/")" "curl http://127.0.0.1:$PORT/ (7 = refused)"

# --- TLS, certificate, CA
record "ca_name_constraints" "yes" "$(openssl x509 -noout -text -in "$CA" 2>/dev/null \
  | grep -A8 'X509v3 Name Constraints: critical' | grep -qF "IP:$IP/255.255.255.255" && echo yes || echo no)" \
  "openssl x509 -text ca.crt: critical Name Constraints permit IP:$IP/32"
# -verify_ip: code 0 must mean "a valid chain FOR THIS IP", not just a valid chain (tester review).
record "tls12_verified_for_ip" "0" "$(openssl s_client -connect "$IP:$PORT" -tls1_2 -CAfile "$CA" -verify_ip "$IP" \
  </dev/null 2>/dev/null | sed -n 's/.*Verify return code: \([0-9]*\).*/\1/p' | head -1)" \
  "openssl s_client -tls1_2 -CAfile ca.crt -verify_ip $IP"
# OpenSSL 3's default security level never even OFFERS TLS 1.1: lower it on the client, so that only the SERVER's
# alert ("protocol version", alert 70) can make this pass (tester review).
tls11="$(openssl s_client -connect "$IP:$PORT" -tls1_1 -cipher 'DEFAULT:@SECLEVEL=0' </dev/null 2>&1 || true)"
if grep -qE 'alert protocol version|alert number 70' <<<"$tls11"; then tls11_seen=yes
elif grep -q 'no protocols available' <<<"$tls11"; then tls11_seen="client cannot offer TLS1.1"
elif grep -qE 'Protocol *: TLSv1\.1|Cipher is ' <<<"$tls11"; then tls11_seen=no
else tls11_seen=unclear; fi
record "tls11_refused_by_server" "yes" "$tls11_seen" "openssl s_client -tls1_1 -cipher DEFAULT:@SECLEVEL=0"
record "cert_ip_san" "yes" "$(openssl s_client -connect "$IP:$PORT" </dev/null 2>/dev/null | openssl x509 -noout -ext \
  subjectAltName 2>/dev/null | grep -q "IP Address:$IP" && echo yes || echo no)" "server cert SAN"

# --- HTTP behaviour (each request reaches the app only when it should)
plain="$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' --max-time 8 "http://$IP:$PORT/x?y=1" 2>/dev/null || true)"
record "http_to_https" "301 $URL/x?y=1" "$plain" "curl http://$IP:$PORT/x?y=1"
record "wrong_host_no_reply" "52" "$(curl_exit --cacert "$CA" -H 'Host: evil.example' "$URL/")" \
  "curl -H 'Host: evil.example' (52 = empty reply, nginx 444)"
health="$(curl -s --max-time 8 --cacert "$CA" -w '\n%{http_code}' "$URL/api/health" 2>/dev/null || true)"
seen="${health##*$'\n'} $(grep -o 'unauthenticated' <<<"$health" | head -1)"
record "app_answers" "401 unauthenticated" "$seen" "curl --cacert ca.crt $URL/api/health (the app's own 401)"
DUMMY="lanGATEdummy$(printf '%031d' 0)"  # 43 chars, never a real token
record "url_token_refused" "401" "$(code --cacert "$CA" "$URL/?token=$DUMMY")" "curl $URL/?token=<dummy>"
headers="$(curl -s -D - -o /dev/null --max-time 8 --cacert "$CA" "$URL/" 2>/dev/null | tr -d '\r' | tr 'A-Z' 'a-z')"
for h in "x-content-type-options: nosniff" "x-frame-options: deny" "referrer-policy: no-referrer" \
  "content-security-policy: frame-ancestors 'none'" "permissions-policy:"; do
  record "header ${h%%:*}" "yes" "$(grep -qF "$h" <<<"$headers" && echo yes || echo no)" "curl -I $URL/"
done
record "no_server_version" "no" "$(grep -q '^server: nginx/' <<<"$headers" && echo yes || echo no)" "curl -I $URL/"

# --- logs: counted, never printed (the dummy token above must not appear: nginx logs $uri only)
logs="$(docker logs "$NGINX" 2>&1 || true)"
record "nginx_log_has_requests" "yes" "$( (( $(grep -c '"GET /api/health"' <<<"$logs") > 0 )) && echo yes || echo no)" \
  "docker logs $NGINX | count GET /api/health"
record "nginx_log_query_strings" "0" "$(grep -c 'token=' <<<"$logs")" "docker logs $NGINX | count 'token='"
record "nginx_log_dummy_token" "0" "$(grep -c "$DUMMY" <<<"$logs")" "docker logs $NGINX | count <dummy>"
app_logs="$(docker logs "$APP" 2>&1 || true)"
record "app_log_dummy_token" "0" "$(grep -c "$DUMMY" <<<"$app_logs")" "docker logs $APP | count <dummy>"

# --- information only (not a verdict): a foreign SNI with our IP. nginx picks the server by Host, not SNI.
sni="$(openssl s_client -connect "$IP:$PORT" -servername evil.example -CAfile "$CA" </dev/null 2>&1 || true)"
if grep -q 'Cipher is ' <<<"$sni"; then sni_seen="handshake completed (default server, same cert)"
else sni_seen="handshake refused"; fi
running_commit="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$APP" 2>/dev/null \
  | sed -n 's/^VLM_BUILD_COMMIT=//p')"

# --- evidence
python3 -I - "$ROWS" "$OUT" "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" \
  "$(docker inspect -f '{{.Image}}' "$NGINX" 2>/dev/null)" \
  "$(docker inspect -f '{{.Config.Image}}' "$NGINX" 2>/dev/null)" \
  "$(openssl x509 -noout -fingerprint -sha256 -in "$CA" 2>/dev/null | cut -d= -f2)" \
  "$(openssl s_client -connect "$IP:$PORT" </dev/null 2>/dev/null | openssl x509 -noout -fingerprint -sha256 2>/dev/null \
     | cut -d= -f2)" "$IP:$PORT" "$NONLOOP" "$running_commit" "$sni_seen" <<'PY'
import json, sys, time
rows, out, commit, image_id, image_ref, ca_fp, cert_fp, target, nonloop, running, sni = sys.argv[1:12]
checks = []
for line in open(rows, encoding="utf-8"):
    name, verdict, expected, actual, command = line.rstrip("\n").split("\t")
    checks.append({"check": name, "verdict": verdict, "expected": expected, "actual": actual, "command": command})
evidence = {
    "gate": "P019 §7 (on this machine)", "target": target, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "commit": {"repo_head": commit, "running_app": running or None,
               "note": None if running == commit else "the running app was built from another commit"},
    "nginx_image": {"id": image_id, "ref": image_ref},
    "info": {"foreign_sni_evil_example": sni},
    "ca_fingerprint_sha256": ca_fp, "server_cert_fingerprint_sha256": cert_fp,
    "non_loopback_listeners": [l for l in nonloop.splitlines() if l],
    "checks": checks, "passed": all(c["verdict"] == "PASS" for c in checks),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(evidence, fh, indent=1)
    fh.write("\n")
PY
echo
echo "evidence: $OUT ($fails failed)"
exit $(( fails > 0 ))
