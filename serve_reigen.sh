#!/bin/bash
# serve_reigen.sh — オーディン / mimir / Reigen API server launcher for ngrok sharing.
#
# Usage:
#   ./serve_reigen.sh [PORT]
#
# First run:
#   echo "your_secret_token_here" > .reigen_api_key
#   ./serve_reigen.sh
#
# In a SECOND terminal (after server is up):
#   ngrok http 8282
#   → public URL: https://xxxx.ngrok-free.dev
#   → share this URL + the API key with your friend
#
# Endpoints available:
#   /odin/*   — オーディン (4 specialist 並列、2026-04-23 new default)
#   /mimir/*  — 単独 mimir (簡単問題向け)
#   /reigen/* — Reigen (legacy / advanced)
#
# Friend uses `reigen_friend_client.run_odin_remote(server_url, api_key, ...)`.
# Default experience_id="genesis" so self_params accumulate across all friends + you.

set -e

PORT="${1:-8282}"

# Resolve API key
if [ -f .reigen_api_key ]; then
    REIGEN_API_KEY="$(head -n 1 .reigen_api_key | tr -d '\r\n[:space:]')"
    if [ -z "$REIGEN_API_KEY" ]; then
        echo "WARNING: .reigen_api_key is empty. Use: echo 'your_token' > .reigen_api_key"
        exit 1
    fi
else
    echo "No .reigen_api_key file found."
    echo "Create one with:  echo 'your_secret_token' > .reigen_api_key"
    echo "Then re-run this script."
    exit 1
fi

# Detect Python on Windows (user's path) with fallback
PY="/c/Users/user/AppData/Local/Programs/Python/Python312/python.exe"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || command -v python || true)"
    if [ -z "$PY" ]; then
        echo "ERROR: Python not found. Edit this script to set correct path."
        exit 1
    fi
fi

export REIGEN_API_KEY
export PYTHONIOENCODING=utf-8

echo "================================================================"
echo "  Reigen API server"
echo "  port       : $PORT"
echo "  api key    : ${REIGEN_API_KEY:0:4}****${REIGEN_API_KEY: -2}   (length ${#REIGEN_API_KEY})"
echo "  python     : $PY"
echo "================================================================"
echo "  In another terminal:   ngrok http $PORT"
echo "  Then share URL + key with friends. Ctrl-C to stop."
echo "================================================================"
echo ""

"$PY" twelve/agent/api.py --port "$PORT"
