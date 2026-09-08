#!/usr/bin/env bash
# The real-world validation gate: production FAM, on a real GPU, with the real
# voice, answering real questions.
#
#   export ANTHROPIC_API_KEY='...'
#   bash tools/pod_production_test.sh
#
# Everything before the server starts is free and fast, and every one of those
# gates exists because something once reported OK while being wrong. In
# particular this refuses to measure anything until it has proved that the
# process really selected Chatterbox - not that Chatterbox is installed, not
# that a device was configured, but that `engine_report()` says the server is
# NOT running on the interim voice. A pod that quietly fell back to Piper would
# produce a full set of plausible numbers about the wrong engine.
#
# Nothing here starts, stops, resizes or pays for a pod. Every one of those is
# a human action, taken outside this script.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
PORT="${PORT:-8000}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="pod-results/$STAMP"
LOG="$OUT/server.log"

#: The questions. One that the model can answer from what it knows, one that
#: the freshness heuristic will route to research - because those are two
#: different products and only measuring the first would flatter the result.
QUERY_KNOWN="${QUERY_KNOWN:-why do stock markets close overnight}"
QUERY_FRESH="${QUERY_FRESH:-what happened in the news today}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\n\033[1mGATE FAILED: %s\033[0m\n\nStop here. Nothing further runs.\n' "$1" >&2; exit 1; }

START=$(date +%s)
elapsed() { printf '[%dm%02ds] ' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

mkdir -p "$OUT"

say "Gate 1: the card (free, instant)"
command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi. Chatterbox refuses CPU - on CPU it is slower than speech, so an episode would starve mid-sentence."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  $GPU"
case "$GPU" in
  *4090*) echo "  ok - the card every previous Chatterbox measurement was taken on" ;;
  *) echo "  NOTE: not a 4090. The run is still valid, but its numbers are not"
     echo "        comparable with the earlier ones. Say so in the write-up." ;;
esac

say "Gate 2: the credential is in this shell (free, instant)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY is not exported. This is the real pipeline; it is not stubbed."
echo "  set (${#ANTHROPIC_API_KEY} characters) - whether it is accepted is Gate 6"

say "Gate 3: this pod is running the bundle you packed (free, instant)"
# POD.txt, not git. The bundle carries no .git on purpose - a rented card is
# somewhere to run FAM for an hour, not somewhere to leave a token - so
# `git rev-parse` here exits 128. The revision this pod is running comes from
# the text the packer wrote, and is stamped into the results so a run is
# traceable without a repository.
[ -f POD.txt ] || fail "no POD.txt. Extract the tarball from tools/pack_for_pod.py and run this from inside FAM/."
sed 's/^/  /' POD.txt
cp POD.txt "$OUT/POD.txt"
REVISION=$(python -c 'import sys;sys.path.insert(0,"tools");from pack_for_pod import pod_txt_field;print(pod_txt_field(open("POD.txt").read(),"revision"))' 2>/dev/null || true)
[ -n "$REVISION" ] || fail "POD.txt names no revision. Re-pack with tools/pack_for_pod.py."
echo "  revision $REVISION  (from POD.txt; this bundle has no .git, by design)"

REFERENCE="${CHATTERBOX_REFERENCE:-}"
if [ -z "$REFERENCE" ]; then
  REFERENCE=$(ls ../pod-voice/*.wav 2>/dev/null | head -1 || true)
  [ -n "$REFERENCE" ] || REFERENCE=$(ls pod-voice/*.wav 2>/dev/null | head -1 || true)
fi
[ -n "$REFERENCE" ] && [ -f "$REFERENCE" ] || fail "no reference recording found. Re-pack with tools/pack_for_pod.py, which ships one, or export CHATTERBOX_REFERENCE."
REFERENCE=$(cd "$(dirname "$REFERENCE")" && pwd)/$(basename "$REFERENCE")
[ -f "${REFERENCE%.wav}.rights.json" ] || fail "no rights record beside $REFERENCE. A cloned voice is somebody's voice; the engine will refuse, and so does this."

# Verified, not displayed. This printed both digests and told a human to
# compare them, which is the shape of check this project has lost the most time
# to: it answers a cheaper question than the one being asked and reports OK.
EXPECTED=$(python -c 'import sys;sys.path.insert(0,"tools");from pack_for_pod import voice_digest;print(voice_digest(open("POD.txt").read()))' 2>/dev/null || true)
ACTUAL=$(sha256sum "$REFERENCE" | cut -c1-16)
echo "  voice    $(basename "$REFERENCE")  sha256 $ACTUAL"
[ -n "$EXPECTED" ] || fail "POD.txt names no voice digest, so the recording cannot be verified. Re-pack."
[ "$EXPECTED" = "$ACTUAL" ] || fail "the reference recording is not the one that was packed. POD.txt says $EXPECTED, this file is $ACTUAL. Re-pack and re-upload; do not measure a voice you cannot identify."
echo "  matches POD.txt - this is the recording that was packed"
export CHATTERBOX_REFERENCE="$REFERENCE"

say "Gate 4: the build passes its own tests here (free, ~30s)"
pip install -q -r requirements.txt >/dev/null 2>&1 || fail "pip install -r requirements.txt failed"
# The suite is hermetic: tests/conftest.py clears every setting config.py reads,
# including the CHATTERBOX_REFERENCE exported just above. That export is correct
# for the server and wrong for a test asserting where the reference resolves by
# default, and it used to fail one. The suite must read the code, not this pod.
python -m pytest tests/ -q >"$OUT/pytest.log" 2>&1 || {
  tail -20 "$OUT/pytest.log"
  fail "the test suite fails on this pod (full output in $OUT/pytest.log). Fix that before spending GPU time - the numbers would be about a broken build."; }
tail -1 "$OUT/pytest.log" | sed 's/^/  /'

say "Gate 5: dependencies (~10 min on a fresh pod, a no-op on a warm one)"
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
elapsed; echo "installing..."
pip install -q -r requirements-chatterbox.txt || fail "pip install -r requirements-chatterbox.txt failed. Run: python tools/diagnose_chatterbox.py"
elapsed; echo "installed"

say "Gate 6: production really selects Chatterbox (free, ~1 min for the weights)"
# Not "is chatterbox importable" and not "is a device configured", but "what
# would this server actually speak with". `interim: true` means it fell back.
python - <<'PY' || fail "production did not select Chatterbox. The output above names the reason; fix exactly that. Do not proceed - a run on the interim voice measures the wrong engine."
import json, sys
from tts import ChatterboxEngine, engine_report
ok, detail = ChatterboxEngine.diagnose()
print(f"  diagnose      {'available' if ok else 'UNAVAILABLE'}: {detail}")
report = engine_report()
print(f"  selected      {report['selected']}")
print(f"  interim       {report['interim']}")
print(f"  voices        {[v['id'] for v in report['voices']]}")
if not ok or report["selected"] != "chatterbox" or report["interim"]:
    sys.exit(1)
print("  ok - the production engine is the one that will speak")
PY

say "Gate 7: the credential is accepted, and the model loads (~15s, a few cents)"
python - <<'PY' || fail "the warm-up failed. The message above is the real one."
import asyncio, time
import tts
started = time.perf_counter()
asyncio.run(tts.warm_up())
loaded = tts.ChatterboxEngine._loaded
assert loaded, "warm_up() finished without a resident model - it swallows exceptions by design, so read the WARNING above"
print(f"  model resident on {list(loaded)} after {time.perf_counter()-started:.1f}s")
PY

# ---------------------------------------------------------------------------
# The measurement. Caching is off: a cached script would make the second run
# report a Claude time of nearly zero and the comparison would be void.
# ---------------------------------------------------------------------------
export CACHE_ENABLED=0

run_pipeline() {  # $1 = legacy|phase6
  local pipeline="$1"
  say "Episode set: STREAMING_PIPELINE=$pipeline"
  STREAMING_PIPELINE="$pipeline" python -m uvicorn app:app \
      --host 127.0.0.1 --port "$PORT" >>"$LOG" 2>&1 &
  local server=$!
  # `trap` rather than a kill at the end: a failed gate must not leave a
  # server holding the GPU on a card that is being billed by the minute.
  trap 'kill '"$server"' 2>/dev/null || true' EXIT

  local waited=0
  until curl -sf "$BASE/api/health" >/dev/null 2>&1; do
    kill -0 "$server" 2>/dev/null || { tail -20 "$LOG"; fail "the server exited during startup. Its output is above and in $LOG"; }
    sleep 1; waited=$((waited+1))
    [ "$waited" -lt 180 ] || fail "the server did not answer /api/health within 3 minutes"
  done
  echo "  up after ${waited}s (warm-up included)"
  curl -s "$BASE/api/health" | python -c 'import json,sys; h=json.load(sys.stdin); e=h.get("engine",{}); print(f"  health: mode={h.get(\"mode\")} engine={e.get(\"selected\")} interim={e.get(\"interim\")}")'

  local status=0
  for pair in "known:$QUERY_KNOWN" "fresh:$QUERY_FRESH"; do
    local label="${pair%%:*}" query="${pair#*:}"
    echo
    python tools/pod_episode.py --base "$BASE" --minutes 3 --log "$LOG" \
        --pipeline "$pipeline" \
        --out "$OUT/${pipeline}_${label}" "$query" || status=$?
  done

  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  trap - EXIT
  return "$status"
}

STATUS=0
# legacy first, so phase6's numbers are read against a baseline taken on the
# same card in the same session rather than against a remembered one.
#
# Only the phase6 runs are scored on decoupling. Legacy structurally cannot
# prove it - its queue is bounded, so a truncated episode cancels the producer
# while the model is still writing and there is no completion instant to
# compare against. Requiring it of the baseline made a wholly successful
# validation end with "ONE OR MORE EPISODES DID NOT PROVE DECOUPLING", which
# is a category error: the baseline is there to be compared against, not to
# pass the test the comparison exists to make.
run_pipeline legacy || STATUS=$?
run_pipeline phase6 || STATUS=$?

elapsed; say "Done"
echo "  $OUT/"
echo "    server.log                 every request's complete timeline"
echo "    legacy_known/episode.wav   listen to these; the numbers do not say"
echo "    phase6_known/episode.wav   whether it sounds like FAM"
echo "    */episode.json             listener view, headers and marks, per run"
echo
echo "  Copy the directory to your Mac BEFORE terminating the pod:"
echo "    scp -r root@<pod>:/workspace/FAM/$OUT ./pod-results/"
echo
if [ "$STATUS" -ne 0 ]; then
  echo "  ONE OR MORE PHASE 6 EPISODES DID NOT PROVE DECOUPLING. The artefacts"
  echo "  are written anyway - that is the evidence. Read the verdict lines"
  echo "  above. The legacy runs are a baseline and are never scored on this."
else
  echo "  Every phase6 episode proved decoupling. The legacy runs are the"
  echo "  baseline they were measured against."
fi
echo "  THEN TERMINATE THE POD."
exit "$STATUS"
