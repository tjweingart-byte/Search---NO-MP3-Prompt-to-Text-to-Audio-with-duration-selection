#!/usr/bin/env bash
# The combined experiment on a rented 4090, as one command with gates.
#
#   export EXA_API_KEY='...'
#   export ANTHROPIC_API_KEY='...'
#   bash tools/pod_combined.sh experiments/references/working/reference_N.wav
#
# Two experiments, kept apart in one timestamped directory:
#   A  Chatterbox Base alone - cold load, first generation, warm buckets
#   B  the whole request     - Exa, Claude, Chatterbox, cold then warm
#
# Every gate before the install is free and instant, so a wrong pod or a
# missing key is caught in seconds rather than after a 4 GB download. The two
# API gates cost about $0.006 between them and are the point: "a key is set"
# is not "the key works".
#
# The keys live in this shell only. Nothing here writes them to a file, and
# the bundle tools/pack_for_pod.py builds never contained them. Rotate them
# after the pod is terminated if you would rather not trust a rented machine.
set -euo pipefail

REFERENCE="${1:-}"
[ -n "$REFERENCE" ] || {
  echo "usage: bash tools/pod_combined.sh <reference.wav>" >&2
  echo "  name the voice the Phase 3 identity bake-off chose. There is no" >&2
  echo "  default: a default would quietly benchmark the wrong voice." >&2
  exit 1; }

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf '\nGATE FAILED: %s\n\nStop here. Nothing further will run.\n' "$1" >&2; exit 1; }

START=$(date +%s)
elapsed() { printf '[%dm%02ds] ' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="experiments/results/combined_4090_$STAMP"

say "Gate 1: the right GPU (free, instant)"
command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi; this is not a GPU pod"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  $GPU"
case "$GPU" in
  *4090*) echo "  ok - RTX 4090, the same card as every earlier Chatterbox run" ;;
  *) fail "expected an RTX 4090, got '$GPU'. A different card makes this a GPU comparison as well as a pipeline one. Terminate and redeploy." ;;
esac

say "Gate 2: credentials are present in this shell (free, instant)"
[ -n "${EXA_API_KEY:-}" ]       || fail "EXA_API_KEY is not exported. Experiment B is the real pipeline; it will not be stubbed."
[ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY is not exported."
echo "  both set - whether they work is Gate 6, not this one"

say "Gate 3: the inputs are the Mac's inputs (free, instant)"
[ -f experiments/chunks/first_chunks.json ] || fail "no corpus at experiments/chunks/first_chunks.json - the tarball did not carry it"
echo -n "  corpus sha256 "; sha256sum experiments/chunks/first_chunks.json | cut -c1-16
[ -f "$REFERENCE" ] || fail "no reference voice at $REFERENCE - pack it with tools/pack_for_pod.py --reference"
echo -n "  voice  sha256 "; sha256sum "$REFERENCE" | cut -c1-16
[ -f POD.txt ] && { echo "  POD.txt says:"; sed 's/^/    /' POD.txt; }
python tools/extract_chunks.py --verify || fail "corpus verify failed"
echo "  Compare the count, buckets and CUT=0 against the Mac before continuing."

say "Gate 4: dependencies (the slow part, ~5-15 min)"
elapsed; echo "installing..."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
echo "  HF_HOME=$HF_HOME  (weights cached here; on a network volume they survive the pod)"
pip install -q -r experiments/requirements-chatterbox.txt || fail "pip install failed"
# The Exa adapter, plus the one app dependency the model stage reaches for.
# `experiments/generate.py` calls production's own script_generator, which
# imports anthropic_client -> anthropic. Nothing else in requirements.txt is
# needed here (no fastapi, no piper), so it is named rather than installed
# wholesale onto a card that is being billed by the minute.
pip install -q -r experiments/requirements.txt anthropic || fail "pip install (exa-py, anthropic) failed"
elapsed; echo "installed"

say "Gate 5: the stack actually imports"
python tools/diagnose_chatterbox.py || fail "diagnose_chatterbox reported a problem - read its prescription above"

say "Gate 6: preflight - GPU, watermarker, voice, corpus, and two real API calls"
python tools/combined_4090_experiment.py --preflight --device cuda \
  --reference "$REFERENCE" || fail "preflight failed - fix exactly what it named"

say "A: Chatterbox Base alone (downloads ~4 GB the first time)"
elapsed; echo "running..."
python tools/combined_4090_experiment.py --experiment a --device cuda \
  --reference "$REFERENCE" --out "$OUT" || fail "experiment A failed"

say "B: the whole request - Exa, Claude, Chatterbox, cold then warm"
elapsed; echo "running..."
python tools/combined_4090_experiment.py --experiment b --device cuda \
  --reference "$REFERENCE" --out "$OUT" \
  || { echo; echo "experiment B failed. A's numbers are already written and"; \
       echo "readable: $OUT/bench_chatterbox_base.json"; \
       python tools/combined_4090_experiment.py --experiment analyse --out "$OUT" || true; \
       fail "experiment B failed"; }

elapsed; say "Done"
echo "  $OUT/"
echo "    results.json                 both experiments"
echo "    ANALYSIS.md                  the waterfall, written out"
echo "    bench_chatterbox_base.json   A, on its own"
echo "    pipeline.json                B, cold and warm"
echo "    events.jsonl                 every mark, raw"
echo "    audio/                       what it actually said"
echo
echo "  COPY THAT DIRECTORY TO YOUR MAC, THEN TERMINATE THE POD."
echo "  From the Mac:  scp -r root@<pod>:/workspace/FAM/$OUT experiments/results/"
