#!/bin/bash
# Monthly Prospector run, fully local: start Ollama, check interests.xlsx, scan, rebuild the radar page.
#   run_monthly.sh            Desktop button: runs now (a second run this month rebuilds the same month's page)
#   run_monthly.sh --auto     scheduled run (launchd): no questions, macOS notification at the end
set -o pipefail  # a failing step is not hidden by the tee to the log
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="$ROOT/src"
DATA="${PROSPECTOR_DATA:-$ROOT/data}"
PY="$ROOT/.venv/bin/python"   # the project .venv when it exists
[ -x "$PY" ] || PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
[ -x "$PY" ] || PY=python3
LOG="$DATA/scans/monthly_run.log"
STAMP="$DATA/scans/last_run.txt"
AUTO=0; [ "$1" = "--auto" ] && AUTO=1
cd "$AGENT" || exit 1
mkdir -p "$DATA/scans"

notify() { osascript -e "display notification \"$1\" with title \"Technique Prospector\"" >/dev/null 2>&1; }
finish() { [ $AUTO = 0 ] && { echo; read -r -p "Press Enter to close this window."; }; exit "$1"; }

# The scheduled run happens once a month; a manual run any time just refreshes this month's page
THIS_MONTH=$(date +%Y-%m)
if [ $AUTO = 1 ] && [ -f "$STAMP" ] && [ "$(cut -c1-7 "$STAMP")" = "$THIS_MONTH" ]; then
  echo "$(date) already ran this month, skipped" >> "$LOG"; exit 0
fi

echo "1/3  Starting the local AI (Ollama)..."
if ! curl -s -m 2 http://localhost:11434/api/tags >/dev/null; then
  open -a Ollama
  for _ in $(seq 1 30); do curl -s -m 2 http://localhost:11434/api/tags >/dev/null && break; sleep 2; done
fi
curl -s -m 2 http://localhost:11434/api/tags >/dev/null || echo "     Ollama did not start: the shortlist will have no notes."

echo "2/3  Checking interests.xlsx..."
if ! "$PY" prospector.py settings 2>&1 | tee -a "$LOG"; then
  notify "Fix interests.xlsx before the scan can run"
  finish 1
fi

echo "3/3  Scanning new papers (about a minute)..."
if "$PY" prospector.py scan 2>&1 | tee -a "$LOG"; then
  date +%Y-%m-%d > "$STAMP"
  N=$(grep -o 'top [0-9]*' "$LOG" | tail -1 | grep -o '[0-9]*')
  W=$(tail -40 "$LOG" | grep -c "WARNING")
  if [ "$W" -gt 0 ]; then notify "${N:-New} picks ready, $W warning(s): check the run report before sharing."
  else notify "${N:-New} picks ready. radar.html is ready to share."; fi
  "$PY" prospector.py quality 2>&1 | tee -a "$LOG" || true
  echo
  echo "Done. The radar page is open in your browser."
  echo "To share it: upload this file to SharePoint or Teams:"
  echo "  $DATA/radar/radar.html"
  [ $AUTO = 0 ] && open "$DATA/radar/radar.html"
  open -R "$DATA/radar/radar.html"   # show the file in Finder, ready to drag into SharePoint / Teams
  finish 0
else
  notify "The scan failed. See data/scans/monthly_run.log"
  echo "The scan failed. Details: $LOG"
  finish 1
fi
