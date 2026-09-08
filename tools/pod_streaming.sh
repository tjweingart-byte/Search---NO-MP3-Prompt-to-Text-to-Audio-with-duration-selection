#!/usr/bin/env bash
# The concurrent pipeline on the 4090 you already have running.
#
#   export EXA_API_KEY='...'
#   export ANTHROPIC_API_KEY='...'
#   bash tools/pod_streaming.sh experiments/references/working/reference_3.wav
#
# Claude and Chatterbox run at the same time, as pipeline.py does it: the first
# speakable chunk goes to the voice while the model is still writing. The
# overlap is asserted - a run whose first TTS started at or after Claude
# finished exits non-zero and says so.
#
# Written for a pod that has already run tools/pod_combined.sh, so the install
# is a no-op and the weights are cached. It is safe to run on a fresh pod too;
# the install gate just takes its usual ten minutes.
set -euo pipefail

REFERENCE="${1:-}"
[ -n "$REFERENCE" ] || {
  echo "usage: bash tools/pod_streaming.sh <reference.wav>" >&2
  echo "  the selected FAM voice, e.g. experiments/references/working/reference_3.wav" >&2
  exit 1; }

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf '\nGATE FAILED: %s\n\nStop here. Nothing further will run.\n' "$1" >&2; exit 1; }

START=$(date +%s)
elapsed() { printf '[%dm%02ds] ' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="experiments/results/streaming_4090_$STAMP"

say "Gate 1: the right GPU (free, instant)"
command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi; this is not a GPU pod"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  $GPU"
case "$GPU" in
  *4090*) echo "  ok - the same card as every earlier Chatterbox run" ;;
  *) fail "expected an RTX 4090, got '$GPU'. Terminate and redeploy." ;;
esac

say "Gate 2: credentials are present in this shell (free, instant)"
[ -n "${EXA_API_KEY:-}" ]       || fail "EXA_API_KEY is not exported. This is the real pipeline; it will not be stubbed."
[ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY is not exported."
echo "  both set - whether they work is Gate 5, not this one"

say "Gate 3: the voice and the code are the current ones (free, instant)"
[ -f "$REFERENCE" ] || fail "no reference voice at $REFERENCE - re-pack with tools/pack_for_pod.py --reference"
echo -n "  voice sha256 "; sha256sum "$REFERENCE" | cut -c1-16
[ -f experiments/concurrent_pipeline.py ] || fail "this pod is running an older tarball: experiments/concurrent_pipeline.py is missing. Re-pack on the Mac and copy it across."
[ -f POD.txt ] && { echo "  POD.txt says:"; sed 's/^/    /' POD.txt; }

say "Gate 4: the wiring, proved here with no model and no API calls (free, ~5s)"
python tools/streaming_4090_experiment.py --dry-run --runs 1 \
  --out "experiments/results/streaming_STUB_$STAMP" \
  || fail "the stub run did not overlap. The harness is wrong, not the card - do not spend GPU time on it."
echo "  the stub overlapped; the code path and the event ordering are sound"

say "Gate 5: dependencies (a no-op on a pod that has already run the combined experiment)"
elapsed; echo "checking..."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
pip install -q -r experiments/requirements-chatterbox.txt || fail "pip install failed"
pip install -q -r experiments/requirements.txt anthropic || fail "pip install (exa-py, anthropic) failed"
elapsed; echo "ready"

say "Gate 6: preflight - GPU, watermarker, voice, chunker, and two real API calls"
python tools/streaming_4090_experiment.py --preflight --device cuda \
  --reference "$REFERENCE" || fail "preflight failed - fix exactly what it named"

say "The experiment: Claude and Chatterbox overlapping, cold then warm"
elapsed; echo "running..."
# `set -e` would abort before STATUS could be read, and a failed overlap
# assertion is a result to report rather than a crash to hide.
STATUS=0
python tools/streaming_4090_experiment.py --device cuda \
  --reference "$REFERENCE" --out "$OUT" || STATUS=$?

elapsed; say "Done"
echo "  $OUT/"
echo "    results.json    every mark, chunk, queue sample and playback figure"
echo "    ANALYSIS.md     did it overlap, and by how much"
echo "    events.jsonl    the raw timeline"
echo "    audio/cold_episode.wav   the whole thing, with production's 0.12s gaps"
echo "    audio/cold/chunk_NN.wav  each chunk as it was spoken"
echo
if [ "$STATUS" -ne 0 ]; then
  echo "  THE RUN DID NOT OVERLAP. Read the 'Did it actually overlap?' section."
  echo "  The results are written anyway - that is the evidence for why."
fi
echo "  COPY THAT DIRECTORY TO YOUR MAC, THEN TERMINATE THE POD."
echo "  From the Mac:  scp -r root@<pod>:/workspace/FAM/$OUT experiments/results/"
exit "$STATUS"
