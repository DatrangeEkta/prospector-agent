#!/bin/bash
# Start the Prospector Agent web app.
#   scripts/run_server.sh           this computer only:        http://127.0.0.1:8081
#   scripts/run_server.sh --team    colleagues on the network: http://<this Mac's address>:8081
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src" || exit 1
PY="$ROOT/.venv/bin/python"   # the project .venv when it exists
[ -x "$PY" ] || PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
[ -x "$PY" ] || PY=python3
PORT="${PORT:-8081}"
HOST=127.0.0.1
if [ "$1" = "--team" ]; then
  HOST=0.0.0.0
  IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
  echo "Colleagues open: http://${IP:-THIS-MAC-ADDRESS}:$PORT   (works while this computer is on and awake)"
fi
echo "Open: http://127.0.0.1:$PORT    Stop: Ctrl+C"
exec "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT"
