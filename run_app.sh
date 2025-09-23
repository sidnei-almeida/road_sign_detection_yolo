#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -d "venv/bin" ]; then
	source venv/bin/activate
fi

exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
