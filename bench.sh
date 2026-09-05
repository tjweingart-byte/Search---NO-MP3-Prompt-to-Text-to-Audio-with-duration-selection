#!/usr/bin/env bash
# The voice bench: search and playback, nothing else.
#
#     ./bench.sh
#
# Same setup as ./start.sh - it reuses the same virtual environment, the same
# key in ~/.fam/env and the same voice store - but serves one screen instead of
# the whole app, so a candidate voice can be judged without myFAM, DailyFAM,
# Explore or the profile in the way.
set -uo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8000}
VENV=.fam-venv

# The same environment ./start.sh builds, built the same way, so the two
# scripts share one install rather than each having their own.
if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up (one time, about a minute)…"
  for candidate in python3.12 python3.11 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
      2>/dev/null && { SYS_PY=$candidate; break; }
  done
  [ -n "${SYS_PY:-}" ] || { echo "Python 3.10+ is needed."; exit 1; }
  "$SYS_PY" -m venv "$VENV" || exit 1
fi
PY="$VENV/bin/python"
if [ ! -f "$VENV/.installed" ] || [ requirements.txt -nt "$VENV/.installed" ]; then
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt || exit 1
  touch "$VENV/.installed"
fi

printf '\n\033[1m==>\033[0m FAM voice bench\n'
"$PY" - <<'PYEOF'
from config import settings
from tts import default_voice, list_voices
print(f"  Default voice : {default_voice()}")
print( "  Voices        :")
for v in list_voices():
    print(f"      {v.id:<34} {v.label}")
print(f"  Ask FAM uses  : {settings.model}"
      f"{'  (no API key - canned script)' if not settings.anthropic_api_key else ''}")
PYEOF
printf '\n  Open  http://localhost:%s\n\n  Control-C to stop.\n\n' "$PORT"

( sleep 2
  if command -v open >/dev/null 2>&1; then open "http://localhost:${PORT}"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:${PORT}"
  fi ) >/dev/null 2>&1 &

exec "$PY" -m uvicorn bench_app:app --host 0.0.0.0 --port "$PORT"
