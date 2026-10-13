#!/usr/bin/env bash
# ADR-019 LAN edge smoke: the pinned nginx with the config install.sh renders, in front of a stub "app" that echoes
# what it receives. Checks the edge's own promises (proposal 019 §3-§6), not the app's sign-in (unit tests do that).
#   scripts/qa/nginx_edge_smoke.sh <work dir>     (a fresh, empty directory; nothing outside it is written)
# Uses only containers/networks named vlm-edge-smoke*, published on 127.0.0.1:4049; removes them at the end.
set -euo pipefail

WORK="${1:?usage: nginx_edge_smoke.sh <empty work dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
IMAGE="$(awk -F' [|] ' '$1 == "edge:nginx" {print $2 "@" $3}' "$ROOT/docker/IMAGES.lock")"
STUB_IMAGE="${VLM_SMOKE_STUB_IMAGE:-ntc-training-app:local}"
NET=vlm-edge-smoke
PORT=4049
NAME=lan.test  # resolved to 127.0.0.1 by curl --resolve: never a real host
SUBNET=10.231.46.0/29
PEER=10.231.46.2
mkdir -p "$WORK"
[[ -z "$(ls -A "$WORK")" ]] || { echo "work dir must be empty: $WORK" >&2; exit 2; }
WORK="$(cd "$WORK" && pwd -P)"

cleanup() {
  docker rm -f vlm-edge-smoke-nginx vlm-edge-smoke-app >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

# shellcheck source=scripts/lib_lan_tls.sh
source "$ROOT/scripts/lib_lan_tls.sh"
make_lan_tls "$WORK/tls" "$NAME"
render_nginx_conf "$ROOT/docker/nginx/nginx.conf.template" "$WORK/nginx.conf" "$NAME" "$SUBNET" "$PORT"

docker network create --subnet "$SUBNET" "$NET" >/dev/null
# The stub app: echoes method, path, query and headers as JSON (any path, any method, any body size).
docker run -d --name vlm-edge-smoke-app --network "$NET" --ip 10.231.46.4 --network-alias app --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true "$STUB_IMAGE" python -c '
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def _echo(self):
        n = int(self.headers.get("content-length") or 0)
        while n > 0:
            n -= len(self.rfile.read(min(n, 1 << 16)))
        body = json.dumps({"method": self.command, "path": self.path, "headers": dict(self.headers.items())}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("x-frame-options", "SAMEORIGIN")  # the edge must replace it, not duplicate it
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    do_GET = do_POST = do_DELETE = _echo
    def log_message(self, *a): pass
ThreadingHTTPServer(("0.0.0.0", 7870), H).serve_forever()
' >/dev/null
docker run -d --name vlm-edge-smoke-nginx --network "$NET" --ip "$PEER" -p "127.0.0.1:$PORT:8443" \
  --read-only --cap-drop ALL --security-opt no-new-privileges:true --tmpfs /tmp:size=64m --user "$(id -u):$(id -g)" \
  -v "$WORK/nginx.conf:/etc/nginx/nginx.conf:ro" -v "$WORK/tls/server.crt:/etc/nginx/tls/server.crt:ro" \
  -v "$WORK/tls/server.key:/etc/nginx/tls/server.key:ro" --entrypoint nginx "$IMAGE" -g 'daemon off;' >/dev/null
for _ in $(seq 50); do
  curl -sf -o /dev/null --cacert "$WORK/tls/ca.crt" --resolve "$NAME:$PORT:127.0.0.1" "https://$NAME:$PORT/" && break
  sleep 0.2
done

fails=0
check() {  # check <description> <command...>
  if "${@:2}"; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); fi
}
URL="https://$NAME:$PORT"
C=(curl -s --max-time 10 --cacert "$WORK/tls/ca.crt" --resolve "$NAME:$PORT:127.0.0.1")

check "the CA carries name constraints" grep -q "Name Constraints" <(openssl x509 -noout -text -in "$WORK/tls/ca.crt")
check "the server certificate verifies with the CA" \
  bash -c 'openssl verify -CAfile "$1" "$2" >/dev/null' _ "$WORK/tls/ca.crt" "$WORK/tls/server.crt"
check "a changed name list never widens the CA" bash -c 'source "$1"; ! make_lan_tls "$2" other.test 2>/dev/null' _ \
  "$ROOT/scripts/lib_lan_tls.sh" "$WORK/tls"
check "the ca.key is 0600" test "$(stat -c %a "$WORK/tls/ca.key")" = 600
echo_json="$("${C[@]}" -H 'X-Forwarded-For: 6.6.6.6' -H 'X-Forwarded-Host: evil' -H 'Forwarded: for=6.6.6.6' \
  "$URL/api/system?token=SMOKESECRET123")"
host_seen="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["headers"].get("Host"))' <<<"$echo_json")"
check "the app gets Host = name:port" test "$host_seen" = "$NAME:$PORT"
check "no X-Forwarded-* / Forwarded reaches the app" \
  python3 -c 'import json,sys; h={k.lower() for k in json.load(sys.stdin)["headers"]}; sys.exit(bool(h & {"x-forwarded-for","x-forwarded-host","x-forwarded-proto","forwarded"}))' <<<"$echo_json"
headers="$("${C[@]}" -D - -o /dev/null "$URL/")"
for h in "x-content-type-options: nosniff" "x-frame-options: DENY" "referrer-policy: no-referrer" \
  "cross-origin-opener-policy: same-origin" "permissions-policy:" "content-security-policy: frame-ancestors 'none'" \
  "strict-transport-security: max-age=31536000"; do
  check "header $h" grep -qi "^$h" <<<"$headers"
done
check "x-frame-options only once" test "$(grep -ci '^x-frame-options' <<<"$headers")" = 1
check "no server version" bash -c '! grep -qi "^server: nginx/" <<<"$1"' _ "$headers"
check "a wrong Host gets no answer (444)" bash -c '! "$@" -o /dev/null "https://'"$NAME:$PORT"'/" -H "Host: evil.example"' _ "${C[@]}"
check "a request by IP with a foreign Host gets no answer" \
  bash -c '! curl -sk --max-time 5 -o /dev/null -H "Host: 127.0.0.1:'"$PORT"'" "https://127.0.0.1:'"$PORT"'/"'
check "plain HTTP with a foreign Host gets no answer" \
  bash -c '! curl -s --max-time 5 -o /dev/null -H "Host: evil.example" "http://127.0.0.1:'"$PORT"'/"'
plain="$(curl -s --max-time 5 -o /dev/null -w '%{http_code} %{redirect_url}' -H "Host: $NAME:$PORT" "http://127.0.0.1:$PORT/x?y=1")"
check "plain HTTP -> 301 https (got: $plain)" test "$plain" = "301 https://$NAME:$PORT/x?y=1"
check "TLS 1.1 refused" bash -c '! openssl s_client -connect 127.0.0.1:'"$PORT"' -servername '"$NAME"' -tls1_1 </dev/null >/dev/null 2>&1'
check "TLS 1.2 accepted" bash -c 'openssl s_client -connect 127.0.0.1:'"$PORT"' -servername '"$NAME"' -tls1_2 -CAfile "$1" </dev/null 2>/dev/null | grep -q "Verify return code: 0"' _ "$WORK/tls/ca.crt"
head -c 3000000 /dev/zero > "$WORK/big.bin"
big_api="$("${C[@]}" -o /dev/null -w '%{http_code}' -X POST --data-binary @"$WORK/big.bin" "$URL/api/graphs")"
check "3 MB to /api/graphs -> 413 (got: $big_api)" test "$big_api" = 413
big_up="$("${C[@]}" -o /dev/null -w '%{http_code}' -X POST --data-binary @"$WORK/big.bin" "$URL/api/uploads")"
check "3 MB to /api/uploads passes (got: $big_up)" test "$big_up" = 200
codes=""
for _ in $(seq 12); do
  codes+="$("${C[@]}" -o /dev/null -w '%{http_code}' -X POST -d '{}' "$URL/api/session") "
done
check "POST /api/session is rate limited (got: $codes)" grep -q 429 <<<"$codes"
get_session="$("${C[@]}" -o /dev/null -w '%{http_code}' "$URL/api/session")"
check "GET /api/session is not limited (got: $get_session)" test "$get_session" = 200
ws="$("${C[@]}" -H 'Upgrade: websocket' -H 'Connection: Upgrade' -H 'X-Forwarded-For: 6.6.6.6' -H 'Forwarded: for=6.6.6.6' \
  "$URL/api/runs/r1/events")"
ws_check() {  # ws_check <python expression over h (lower-cased headers)>
  python3 -c 'import json,sys; h={k.lower():v for k,v in json.load(sys.stdin)["headers"].items()}; sys.exit(not eval(sys.argv[1]))' "$1" <<<"$ws"
}
check "WebSocket path forwards Upgrade" ws_check 'h.get("upgrade") == "websocket"'
check "WebSocket path sends Host = name:port" ws_check 'h.get("host") == "'"$NAME:$PORT"'"'
check "WebSocket path drops X-Forwarded-* / Forwarded" \
  ws_check 'not ({"x-forwarded-for","x-forwarded-host","x-forwarded-proto","forwarded"} & set(h))'
logs="$(docker logs vlm-edge-smoke-nginx 2>&1)"
check "access log has requests" grep -q '"GET /api/system"' <<<"$logs"
check "access log has no query string / token" bash -c '! grep -q SMOKESECRET123 <<<"$1"' _ "$logs"

# The subnet allowlist: a config that does not allow our source must answer 403.
render_nginx_conf "$ROOT/docker/nginx/nginx.conf.template" "$WORK/deny.conf" "$NAME" "192.0.2.0/24" "$PORT"
cat "$WORK/deny.conf" > "$WORK/nginx.conf"  # same inode: the container's bind mount sees it
docker exec vlm-edge-smoke-nginx nginx -s reload >/dev/null 2>&1 || true
sleep 1
denied="$("${C[@]}" -o /dev/null -w '%{http_code}' "$URL/")"
check "a source outside VLM_LAN_ALLOW -> 403 (got: $denied)" test "$denied" = 403

echo
if (( fails )); then echo "nginx edge smoke: $fails check(s) FAILED"; exit 1; fi
echo "nginx edge smoke: all checks passed"
