#!/usr/bin/env bash
# Phase 1 on a rented GPU, as one command with gates.
#
# Typing interactively on a billing card is the avoidable cost: the Phase 1
# measurement itself is a few minutes, while setup is twenty. This runs the
# whole sequence, stops at the first gate that fails, and says what to do.
#
#   bash tools/pod_phase1.sh
#
# Every gate before the install is free and instant, so a wrong pod is caught
# in seconds rather than after a 4 GB download.
set -euo pipefail

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf '\nGATE FAILED: %s\n\nStop here. Nothing further will run.\n' "$1" >&2; exit 1; }

START=$(date +%s)
elapsed() { printf '[%dm%02ds] ' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

say "Gate 1: the right GPU (free, instant)"
if ! command -v nvidia-smi >/dev/null 2>&1; then fail "no nvidia-smi; this is not a GPU pod"; fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  $GPU"
case "$GPU" in
  *4090*) echo "  ok - RTX 4090, matching the previous benchmark" ;;
  *) fail "expected an RTX 4090, got '$GPU'. A different card makes the comparison a GPU-vs-GPU one too. Terminate and redeploy." ;;
esac

say "Gate 2: the corpus is the Mac's corpus (free, instant)"
[ -f experiments/chunks/first_chunks.json ] || fail "no corpus at experiments/chunks/first_chunks.json - the tarball did not carry it"
echo -n "  sha256 "; sha256sum experiments/chunks/first_chunks.json | cut -c1-16
[ -f POD.txt ] && { echo "  POD.txt says:"; sed 's/^/    /' POD.txt; }
python tools/extract_chunks.py --verify || fail "corpus verify failed"
echo "  Compare the count, buckets and CUT=0 against the Mac before continuing."

say "Gate 3: dependencies (this is the slow part, ~5-15 min)"
elapsed; echo "installing..."
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
echo "  HF_HOME=$HF_HOME  (weights cached here; on a network volume they survive the pod)"
pip install -q -r experiments/requirements-chatterbox.txt || fail "pip install failed"
elapsed; echo "installed"

say "Gate 4: the stack actually imports"
python tools/diagnose_chatterbox.py || fail "diagnose_chatterbox reported a problem - read its prescription above"

say "Gate 5: preflight"
python tools/chatterbox_first_audio.py --local --preflight --device cuda \
  || fail "preflight failed - fix what it named"

say "Gate 6: three generations, to prove the card (downloads ~4 GB the first time)"
elapsed; echo "smoke run..."
python tools/chatterbox_first_audio.py --local --device cuda --trials 1 --max-chunks 1 \
  || fail "smoke run failed"
echo "  Expect well under 1s each. Near 10s means something is wrong - stop and report."

say "Phase 1: the stage split"
elapsed; echo "running..."
mkdir -p experiments/results
python tools/chatterbox_stage_split.py --device cuda \
  --out experiments/results/chatterbox_stage_split.json || fail "stage split failed"

elapsed; say "Done"
echo "  experiments/results/chatterbox_stage_split.json"
echo
echo "  COPY THAT FILE TO YOUR MAC, THEN TERMINATE THE POD."
echo "  In JupyterLab: right-click the file -> Download."
