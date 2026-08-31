# Runs FAM anywhere that takes a container: Render, Fly, Railway, Cloud Run.
# Kept deliberately plain - there is no build step, because the interface is a
# static file and the server is one Python process.
FROM python:3.12-slim

# espeak-ng is the fallback voice. Piper voices are downloaded on first run
# into ~/.fam/voices (see voice_store.py); espeak means the container can
# always speak, even before that has happened.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Databases live on a mounted disk where the host provides one, so mixes and
# listening history survive a redeploy. Without a disk they are ephemeral and
# every deploy is a fresh start - which is fine for a preview and not for real
# listeners.
ENV CACHE_PATH=/data/scripts.db \
    MYFAM_DB=/data/myfam.db \
    MIXES_DB=/data/mixes.db \
    PORT=8000
RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
