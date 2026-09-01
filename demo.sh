#!/usr/bin/env bash
# The whole product, running, with every tab able to play a real episode.
#
#   ./demo.sh              check, seed if needed, then serve
#   ./demo.sh --no-seed    skip the seeding offer
#   ./demo.sh --anyway     start even if the key or the voice is missing
#
# This is not `./dev.sh` and not the phone preview. dev.sh runs the checks; the
# preview runs the interface on fixtures with no model behind it. This runs the
# real server, so what you hear is what the model wrote.
#
# It refuses to start quietly broken. Missing the API key means every episode
# is the same canned sample, and missing a voice model means playback is a
# placeholder tone - both of which look like a working demo from across the
# room, which is how this project has lost the most time.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
PORT=${PORT:-8000}
SEED=1
ANYWAY=0
for arg in "$@"; do
  case "$arg" in
    --no-seed) SEED=0 ;;
    --anyway)  ANYWAY=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

# .env is where the key lives, and every check below depends on having read it.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# One command from a machine that has nothing but Python. A demo that needs a
# README before it runs is a demo nobody sees.
if ! $PY -c "import fastapi, uvicorn" 2>/dev/null; then
  printf '\n\033[1mFirst run: nothing is installed yet.\033[0m\n'
  printf '  This will create .venv here and install into it (~1 min, nothing global).\n'
  if [ -t 0 ]; then
    printf '  Install now? [Y/n] '
    read -r reply
  else
    reply=n
  fi
  case "$reply" in
    [nN]*) printf '  Then run:  %s -m pip install -r requirements.txt\n' "$PY"; exit 1 ;;
  esac
  $PY -m venv .venv
  PY=".venv/bin/python"
  $PY -m pip install --quiet --upgrade pip
  $PY -m pip install --quiet -r requirements.txt
  printf '  Installed.\n'
elif [ -x .venv/bin/python ] && ! $PY -c "import fastapi" 2>/dev/null; then
  PY=".venv/bin/python"
fi

# The voice model is a separate download and the app is honest about not having
# it: with none installed, playback is a placeholder tone rather than speech.
if ! $PY -c "
import sys, tts
sys.exit(0 if [v for v in tts.list_voices() if v.engine != 'debug'] else 1)
" 2>/dev/null; then
  printf '\n\033[1mNo voice model installed.\033[0m Without one, playback is a tone.\n'
  if [ -t 0 ]; then
    printf '  Download one now (~60 MB, once, shared by every copy of the app)? [Y/n] '
    read -r reply
  else
    reply=n
  fi
  case "$reply" in
    [nN]*) printf '  Then run:  %s setup_voices.py\n' "$PY" ;;
    *) $PY setup_voices.py || printf '  Voice install failed - see the message above.\n' ;;
  esac
fi

# Without a key every episode is the same canned sample, which is the one thing
# a demo of the writing cannot be. setup_key.py stores it once per machine in
# ~/.fam/env - outside the project, so the next copy of the app finds it - and
# refuses to store one Claude does not accept.
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -t 0 ] && [ "$ANYWAY" -eq 0 ]; then
  printf '\n\033[1mNo API key, so nothing can be written.\033[0m\n'
  $PY setup_key.py || printf '  Continuing without one.\n'
  # setup_key wrote the file; pick it up for the checks below.
  FAM_ENV="${FAM_ENV_FILE:-$HOME/.fam/env}"
  if [ -f "$FAM_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$FAM_ENV"
    set +a
  fi
fi

set +e
$PY tools/demo_preflight.py
READY=$?
set -e

if [ "$READY" -ge 2 ] && [ "$ANYWAY" -eq 0 ]; then
  printf '\033[1mNot starting.\033[0m Fix the lines marked above, or run\n'
  printf '  ./demo.sh --anyway   to see the interface knowing the audio is fake.\n\n'
  exit 1
fi

# Explore is the one tab that cannot fill itself - it replays other people's
# episodes and refuses to generate - so an unseeded demo has a dead tab in it.
if [ "$SEED" -eq 1 ] && [ "$READY" -eq 1 ] && [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  printf '\033[1mExplore is empty.\033[0m Seeding writes 8 real episodes '
  printf '(8 model calls, no audio).\n'
  if [ -t 0 ]; then
    printf '  Seed now? [y/N] '
    read -r reply
    case "$reply" in
      [yY]*) $PY tools/seed_demo.py ;;
      *) printf '  Skipped. Run  %s tools/seed_demo.py  whenever you want it.\n' "$PY" ;;
    esac
  else
    printf '  Not a terminal, so not asking. Run: %s tools/seed_demo.py\n' "$PY"
  fi
fi

# The address a phone can actually reach. localhost is useless from a phone.
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

printf '\n\033[1m== Demo\033[0m\n'
printf '  On this machine   http://localhost:%s\n' "$PORT"
if [ -n "$LAN" ]; then
  printf '  \033[1mOn your phone     http://%s:%s\033[0m   (same wifi)\n' "$LAN" "$PORT"
fi
cat <<'TOUR'

  Where to press, and what each one proves:

    search     type a question, pick a length, press play.
               This is the surface to judge the writing on. Listen to the
               last two sentences: they should land and stop, never tease.
    myFAM      tap any tile. Same pipeline, the question comes from the
               shared bank instead of the box. A tile someone already
               played starts instantly - that is the cache, not a trick.
    DailyFAM   open a starter mix and play it through. A mix holds topics,
               never audio, so it is a different set of episodes each day.
    explore    swipe the feed. Nothing here is written on demand: it replays
               what other listeners generated, and a card that is no longer
               cached says so rather than quietly writing a new one.
    profile    only what the event log actually holds. If it looks thin,
               that is honest - play a few episodes and come back.

  Go Deeper, after any episode, offers the follow-up the model predicted
  while writing it. That line is never spoken.

  Ctrl-C to stop.
TOUR
printf '\n'

exec $PY -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "$PORT"
