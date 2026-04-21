#!/bin/bash
# start_reigen_all.sh — launch ngrok + Reigen API server together.
# Ctrl-C to stop both.
#
# Usage:
#   ./start_reigen_all.sh [PORT]       (default 8282)

set -u
PORT="${1:-8282}"
cd "$(dirname "$0")"

cleanup() {
    echo ""
    echo "[stop] shutting down ngrok (PID ${NGROK_PID:-?})..."
    [ -n "${NGROK_PID:-}" ] && kill "$NGROK_PID" 2>/dev/null || true
    echo "[stop] done."
}
trap cleanup EXIT

echo "[1/2] launching ngrok http $PORT (background → ngrok.log)..."
ngrok http "$PORT" --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!

# Wait up to 10s for ngrok web API to report the tunnel URL
URL=""
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    URL=$(curl -s --max-time 1 http://localhost:4040/api/tunnels 2>/dev/null \
        | grep -oE 'https://[a-zA-Z0-9.-]+\.ngrok-free\.dev' | head -1 || true)
    [ -n "$URL" ] && break
done

echo ""
echo "================================================================"
if [ -n "$URL" ]; then
    echo "  PUBLIC URL : $URL"
    echo ""
    echo "  友達に渡すもの:"
    echo "    URL      : $URL"
    echo "    API Key  : (.reigen_api_key の中身を手渡し)"
    echo "    Client   : reigen_friend_client.py  (or curl $URL/client.py)"
    echo "    Docs     : FRIEND_README.md  (or curl $URL/readme)"
    echo ""
    echo "  友達側の使用例 (Python):"
    echo "    from reigen_friend_client import run_mimir_remote"
    echo "    r = run_mimir_remote('$URL', 'YOUR_KEY',"
    echo "                        eval_fn=my_fn,"
    echo "                        param_ranges=[(-5,5)]*8,"
    echo "                        time_budget=300)"
    echo "    print(r['best_params'], r['best_score'], r['tool_used'])"
else
    echo "  PUBLIC URL : (not ready yet — check ngrok.log)"
fi
echo "  LOCAL      : http://localhost:$PORT"
echo "  ngrok PID  : $NGROK_PID      (log: ngrok.log)"
echo "================================================================"
echo "  Endpoints: /mimir, /mimir/start, /reigen, /reigen/start, /owl, /health"
echo "  Ctrl-C to stop BOTH ngrok and the Reigen server."
echo ""

echo "[2/2] launching Reigen + mimir API server (foreground)..."
./serve_reigen.sh "$PORT"
