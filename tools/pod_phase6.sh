#!/usr/bin/env bash
# Phase 6 on the 4090 you already have running: Claude decoupled from Chatterbox.
#
#   export EXA_API_KEY='...'
#   export ANTHROPIC_API_KEY='...'
#   bash tools/pod_phase6.sh experiments/references/working/reference_3.wav
#
# Phase 5 proved the overlap and then reported a Claude time that was mostly
# our own queue. This puts a cheap character-bounded script buffer between the
# reader and the voice, batches later sentences into speech-sized chunks, and
# asserts that TTS saturation never reaches upstream again.
#
# Written for a pod that has already run Phase 5, so the install is a no-op and
# the weights are cached. Safe on a fresh pod too; the install gate just takes
# its usual ten minutes.
set -euo pipefail

REFERENCE="${1:-}"
[ -n "$REFERENCE" ] || {
  echo "usage: bash tools/pod_phase6.sh <reference.wav>" >&2
  echo "  the selected FAM voice: experiments/references/working/reference_3.wav" >&2
  exit 1; }

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf '\nGATE FAILED: %s\n\nStop here. Nothing further will run.\n' "$1" >&2; exit 1; }

START=$(date +%s)
elapsed() { printf '[%dm%02ds] ' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="experiments/results/phase6_4090_$STAMP"

say "Gate 1: the right GPU (free, instant)"
command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi; this is not a GPU pod"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  $GPU"
case "$GPU" in
  *4090*) echo "  ok - the same card as Phase 5, so the comparison is architecture only" ;;
  *) fail "expected an RTX 4090, got '$GPU'. A different card would make this a hardware comparison too. Terminate and redeploy." ;;
esac

say "Gate 2: credentials are present in this shell (free, instant)"
[ -n "${EXA_API_KEY:-}" ]       || fail "EXA_API_KEY is not exported. This is the real pipeline; it will not be stubbed."
[ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY is not exported."
echo "  both set - whether they work is Gate 5"

say "Gate 3: the voice and the code are the current ones (free, instant)"
[ -f "$REFERENCE" ] || fail "no reference voice at $REFERENCE - re-pack with tools/pack_for_pod.py --reference"
echo -n "  voice sha256 "; sha256sum "$REFERENCE" | cut -c1-16
[ -f experiments/decoupled_pipeline.py ] || fail "this pod is running an older tarball: experiments/decoupled_pipeline.py is missing. Re-pack on the Mac and copy it across."
[ -f experiments/speech_assembler.py ] || fail "experiments/speech_assembler.py is missing; re-pack on the Mac."
[ -f POD.txt ] && { echo "  POD.txt says:"; sed 's/^/    /' POD.txt; }

say "Gate 4a: the decoupled build passes its assertions here (free, ~10s)"
python tools/phase6_experiment.py --dry-run --runs 1 \
  --out "experiments/results/phase6_STUB_$STAMP" \
  || fail "the stub run failed its own assertions. The harness is wrong, not the card - do not spend GPU time on it."

say "Gate 4b: the coupled build FAILS them (free, ~10s)"
# An assertion that has never rejected anything is decoration. This rebuilds
# Phase 5's coupling on purpose and requires the run to fail.
python tools/phase6_experiment.py --dry-run --coupled --runs 1 \
  --out "experiments/results/phase6_STUB_coupled_$STAMP" \
  || fail "the coupled fixture did not fail as designed - the decoupling assertion has no teeth, so no Phase 6 result would mean anything"
echo "  the coupled fixture failed, as it must; the assertion has teeth"

say "Gate 5: dependencies (a no-op on a pod that has already run Phase 5)"
elapsed; echo "checking..."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
pip install -q -r experiments/requirements-chatterbox.txt || fail "pip install failed"
pip install -q -r experiments/requirements.txt anthropic || fail "pip install (exa-py, anthropic) failed"
elapsed; echo "ready"

say "Gate 6: preflight - GPU, watermarker, voice, chunker, and two real API calls"
python tools/phase6_experiment.py --preflight --device cuda \
  --reference "$REFERENCE" || fail "preflight failed - fix exactly what it named"

say "Phase 6: cold, then warm"
elapsed; echo "running..."
# `set -e` would abort before STATUS could be read, and a failed assertion is a
# result to report rather than a crash to hide.
STATUS=0
python tools/phase6_experiment.py --device cuda \
  --reference "$REFERENCE" --out "$OUT" || STATUS=$?

elapsed; say "Done"
echo "  $OUT/"
echo "    ANALYSIS.md     six plain-English answers, then the waterfall"
echo "    results.json    every mark, chunk, buffer sample and headroom figure"
echo "    events.jsonl    the raw timeline"
echo "    chunks.json     raw sentences and assembled chunks, side by side"
echo "    audio/cold_episode.wav, audio/warm_episode.wav"
echo
echo "  Then, against Phase 5's own results.json if you still have it:"
echo "    python tools/fit_chunk_policy.py <phase5>/results.json --compare $OUT/results.json"
echo
if [ "$STATUS" -ne 0 ]; then
  echo "  PHASE 6 ASSERTIONS FAILED. Read the executive result at the top of"
  echo "  ANALYSIS.md. The artefacts are written anyway - that is the evidence."
fi
echo "  COPY THAT DIRECTORY TO YOUR MAC, THEN TERMINATE THE POD."
echo "  From the Mac:  scp -r root@<pod>:/workspace/FAM/$OUT experiments/results/"
exit "$STATUS"
