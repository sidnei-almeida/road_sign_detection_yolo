#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -d "venv/bin" ]; then
	source venv/bin/activate
fi

PORT="${PORT:-8000}"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
