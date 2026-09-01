#!/usr/bin/env bash
# The whole loop in one command.
#
#   ./dev.sh          tests, checks, preview build, then serve on your LAN
#   ./dev.sh check    tests and checks only, no server
#   ./dev.sh preview  build the phone preview only
#
# The LAN address it prints is openable from a phone on the same wifi, with
# real audio and real generation. The preview file needs no server at all but
# has no model behind it - use it for layout and flow, the server for sound.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
PORT=${PORT:-8000}
MODE=${1:-serve}

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if [ "$MODE" != "preview" ]; then
  step "Tests"
  $PY -m pytest tests/ -q

  step "Interface parses"
  $PY tools/check_js.py

  step "Interface is styled"
  $PY tools/check_css.py
fi

step "Phone preview"
$PY preview/build_preview.py
if command -v node >/dev/null 2>&1 && $PY -c "import playwright" 2>/dev/null; then
  $PY tools/smoke_preview.py || echo "  (smoke test failed - the preview still built)"
fi

[ "$MODE" = "check" ] && exit 0
[ "$MODE" = "preview" ] && exit 0

# The address a phone can actually reach. localhost is useless from a phone,
# and hunting for it by hand is the manual step this removes.
LAN=$($PY - <<'PYEOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
except Exception:
    print("")
finally:
    s.close()
PYEOF
)

step "Server"
if [ -n "$LAN" ]; then
  printf '  On this machine   http://localhost:%s\n' "$PORT"
  printf '  \033[1mOn your phone     http://%s:%s\033[0m   (same wifi)\n' "$LAN" "$PORT"
else
  printf '  http://localhost:%s   (could not work out this machine'"'"'s LAN address)\n' "$PORT"
fi
printf '  Preview file      preview/fam-preview.html   (open anywhere, no server)\n\n'

[ -f .env ] && set -a && . ./.env && set +a
exec $PY -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
