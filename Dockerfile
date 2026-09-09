# Runs FAM anywhere that takes a container: Render, Fly, Railway, Cloud Run.
# Kept deliberately plain - there is no build step, because the interface is a
# static file and the server is one Python process.
FROM python:3.12-slim

# espeak-ng is a development engine only, reachable through TTS_ENGINE and
# never selected by production - nothing falls back to it. It is installed so
# that a container can be used for local work; an image that is meant to speak
# needs a GPU and requirements-chatterbox.txt, and without those FAM reports
# `interim: true` and plays a placeholder tone rather than a worse voice.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Both files, for the reason the GPU image installs both: `exa_py` lives only
# in requirements-exa.txt, and without it RESEARCH_BACKEND=exa - the default -
# reports "exa_py is not installed" and every researched episode FAILS rather
# than quietly searching another way. That is the one dependency gap between
# this image and Dockerfile.gpu that is fixable on a CPU host, and it is pure
# Python: no torch, no CUDA, no meaningful size.
#
# requirements-chatterbox.txt is deliberately NOT installed here, and adding it
# would not help. `ChatterboxEngine.diagnose()` refuses a CPU device before it
# ever looks for the model - "no GPU: Chatterbox on CPU is slower than speech"
# - so on a CPU host installing it swaps one unavailable-reason for another,
# still serves the placeholder tone, and costs gigabytes of image to do it.
# An image meant to speak is Dockerfile.gpu, which needs a card.
COPY requirements.txt requirements-exa.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-exa.txt

COPY . .

# All five databases live on a mounted disk where the host provides one, so
# they survive a redeploy. Without a disk they are ephemeral and every deploy
# is a fresh start - which is fine for a preview and not for real listeners.
#
# Every one of the five is named here on purpose. Three used to be, and the two
# that were missing (social, attachments) were written to the image's WORKDIR
# instead of the disk, so every redeploy silently discarded every listener's
# name, handle and echo. A store added later must be added here too, and
# tests/test_data_paths.py fails if one is not.
ENV CACHE_PATH=/data/scripts.db \
    MYFAM_DB=/data/myfam.db \
    MIXES_DB=/data/mixes.db \
    SOCIAL_DB=/data/social.db \
    ATTACHMENTS_PATH=/data/attachments.db \
    ACCOUNTS_DB=/data/accounts.db \
    PORT=8000
RUN mkdir -p /data

# The validated settings, pinned exactly as Dockerfile.gpu pins them, so a
# container from either image resolves the same generation path and neither
# depends on a dashboard field being retyped correctly.
#
# The first two are already the code defaults; naming them makes the image
# self-describing and makes a hand-rollback to `legacy` visible in
# /api/health rather than invisible from outside.
#
# ANSWER_FIRST=1 is the one that is NOT the code default, and it is the reason
# this block exists. `config._answer_first_default()` derives the value from
# the research backend - Exa retrieves in about half a second, so there is no
# wait to cover - which resolves `exa` to answer_first=False. But the
# configuration that was listened to and judged good (Phase 6, Chatterbox,
# reference_3, ~2.992s to first audio, research handing off mid-episode) ran
# with the cover ON: it predates that derivation (commit 91d9dad) and
# `tools/pod_production_test.sh` never set the variable. Dockerfile.gpu pins it
# for exactly this reason and carries the full argument; this image was left
# out, so it deployed a configuration nobody has heard.
#
# Provisional in both images, and they come out together: run
# `ANSWER_FIRST=1 bash tools/pod_production_test.sh` on a card, compare against
# the same harness without it, and let the numbers decide whether the cover
# belongs in config.py.
ENV STREAMING_PIPELINE=phase6 \
    RESEARCH_BACKEND=exa \
    CHATTERBOX_DEVICE=auto \
    ANSWER_FIRST=1

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
