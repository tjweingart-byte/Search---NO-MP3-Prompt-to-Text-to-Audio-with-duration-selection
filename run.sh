#!/usr/bin/env sh
# Start the server. Reads .env if present.
set -e
[ -f .env ] && . ./.env && export $(grep -v '^#' .env | sed 's/=.*//' | xargs)
exec python3 -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
