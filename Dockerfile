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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    METERING_DB=/data/metering.db \
    PORT=8000
RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
