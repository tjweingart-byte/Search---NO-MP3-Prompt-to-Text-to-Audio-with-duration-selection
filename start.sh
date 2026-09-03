#!/usr/bin/env bash
# One command, from a machine that has nothing but Python.
#
#     ./start.sh
#
# It installs what it needs into a folder next to this file, asks for any key
# that is missing, offers to download a voice, and then starts the app and
# opens it in your browser. Run it again any time - everything it did the first
# time is remembered, so the second run just starts the app.
#
# Nothing is installed system-wide and nothing is written outside this folder,
# except the two API keys, which go in ~/.fam/env deliberately: outside the
# project, so they survive you replacing this folder with a newer copy.
set -uo pipefail
cd "$(dirname "$0")"

bold=$(printf '\033[1m'); dim=$(printf '\033[2m'); off=$(printf '\033[0m')
green=$(printf '\033[32m'); red=$(printf '\033[31m'); amber=$(printf '\033[33m')

say()  { printf '\n%s==>%s %s%s\n' "$bold" "$off" "$1" "$off"; }
ok()   { printf '  %s✓%s %s\n' "$green" "$off" "$1"; }
warn() { printf '  %s!%s %s\n' "$amber" "$off" "$1"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$red" "$1" "$off"; exit 1; }

PORT=${PORT:-8000}
VENV=.fam-venv

# --- Python -----------------------------------------------------------------
say "Checking Python"
SYS_PY=""
for candidate in python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      SYS_PY=$candidate
      break
    fi
  fi
done
[ -n "$SYS_PY" ] || die "Python 3.10 or newer is needed. Install it from python.org and run this again."
ok "$($SYS_PY --version)"

# --- Dependencies -----------------------------------------------------------
# In a virtual environment on purpose. A plain `pip install` on a recent macOS
# either fails with "externally-managed-environment" or quietly installs into a
# Python the app is not using - both of which look like the app being broken.
if [ ! -x "$VENV/bin/python" ]; then
  say "Setting up (one time, about a minute)"
  "$SYS_PY" -m venv "$VENV" || die "Could not create a virtual environment in $VENV"
fi
PY="$VENV/bin/python"

if [ ! -f "$VENV/.installed" ] || [ requirements.txt -nt "$VENV/.installed" ]; then
  say "Installing what the app needs"
  "$PY" -m pip install --quiet --upgrade pip || warn "pip could not update itself; carrying on"
  "$PY" -m pip install --quiet -r requirements.txt || die "Installing dependencies failed. The output above says why."
  touch "$VENV/.installed"
fi
ok "dependencies ready"

# --- Keys -------------------------------------------------------------------
# Both are stored by their own script, which verifies the key before writing it.
# Neither is ever written into the source or into this folder.
say "Checking your keys"

if "$PY" -c 'import sys; from config import settings; sys.exit(0 if settings.anthropic_api_key else 1)'; then
  ok "Anthropic key found (this is what writes the episodes)"
else
  warn "No Anthropic key yet. Without it every episode is the same canned sample."
  printf '\n'
  "$PY" setup_key.py || die "No Anthropic key was stored, so there is nothing to listen to yet. Run ./start.sh again."
fi

if "$PY" -c 'import sys; from config import settings; sys.exit(0 if settings.wellsaid_api_key else 1)'; then
  ok "WellSaid key found (Chase J and Kai M will be in the voice list)"
else
  printf '\n  %sWellSaid is optional.%s Without it the app works exactly as before,\n' "$bold" "$off"
  printf '  with Piper as the voice. With it you also get Chase J and Kai M.\n'
  printf '  Get a key from studio.wellsaidlabs.com under Settings -> API.\n'
  printf '  %s(looked for it in %s)%s\n\n' "$dim" "${FAM_ENV_FILE:-$HOME/.fam/env}" "$off"

  # No "would you like to?" gate. There used to be one, and `read` answers
  # itself the instant stdin is not an interactive terminal - so the question
  # appeared and was declined in the same breath, which reads exactly like
  # never being asked at all. setup_wellsaid.py does its own asking and
  # already treats an empty line as "nothing changed", so it can simply be
  # run: pasting a key is the answer, and pressing Return is the other one.
  if [ -t 0 ]; then
    "$PY" setup_wellsaid.py || warn "No WellSaid key stored. Piper still works; run ./start.sh again to retry."
  else
    # Being honest about it beats a prompt that cannot be answered.
    warn "Not running in an interactive terminal, so the key cannot be typed here."
    printf '  Open Terminal, cd to this folder, and run:  ./start.sh\n'
    printf '  Or, to add just the key:  %s setup_wellsaid.py\n' "$PY"
  fi
fi

# --- Voice ------------------------------------------------------------------
say "Checking the voice"
if "$PY" -c 'import sys; from tts import PiperEngine; sys.exit(0 if PiperEngine.available() else 1)' 2>/dev/null; then
  ok "Piper voice installed"
else
  warn "No Piper voice yet - without one you would hear a placeholder tone."
  printf '  Downloading one now (about 60 MB, once)…\n\n'
  "$PY" setup_voices.py || warn "The voice download did not finish. WellSaid will still work if its key is set."
fi

# --- What you are about to hear ---------------------------------------------
say "Ready"
"$PY" - <<'PYEOF'
from config import settings
from tts import default_voice, list_voices

voices = list_voices()
print(f"  Episodes written by : {settings.model}")
print(f"  Default voice       : {default_voice()}")
print( "  Voices you can pick :")
for v in voices:
    mark = "  <- WellSaid" if v.engine == "wellsaid" else ""
    print(f"      {v.label}{mark}")
if not any(v.engine == "wellsaid" for v in voices):
    print("      (no WellSaid voices - its key is not set)")
PYEOF

cat <<EOM

  Open  http://localhost:${PORT}  - it should open by itself in a moment.

  To hear the same episode in a different voice:
    search for something, wait for it to play, then tap the pill under
    the play button -> Voice -> Chase J (WellSaid) or Kai M (WellSaid).
    The words are identical - only the speaker changes, and it costs
    no extra Claude call.

  ${dim}Watch this window: every sentence WellSaid speaks prints a line
  starting "provider=wellsaid". No such lines means you are hearing Piper.${off}

  Press Control-C to stop.

EOM

( sleep 2
  if command -v open >/dev/null 2>&1; then open "http://localhost:${PORT}"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:${PORT}"
  fi ) >/dev/null 2>&1 &

exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
