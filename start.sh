#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8051

# Create venv if it doesn't exist
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Install/upgrade dependencies
echo "Installing dependencies..."
.venv/bin/pip install -q -r Codebase/requirements.txt

# Generate XAI artefacts if they haven't been computed yet
if [ ! -f "Codebase/models/saved/shap_rf_values.npy" ]; then
    echo "Computing XAI artefacts (SHAP + Integrated Gradients) — first run only..."
    .venv/bin/python Codebase/models/compute_xai.py
fi

echo "Starting AIFI Tactile Dashboard..."
.venv/bin/python run.py &
SERVER_PID=$!

# Wait until the server responds before opening the browser
echo "Waiting for server..."
for i in $(seq 1 20); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "200"; then
        break
    fi
    sleep 1
done

if command -v open &>/dev/null; then
    open "http://127.0.0.1:$PORT"        # macOS
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:$PORT"   # Linux
fi

echo "Dashboard running at http://127.0.0.1:$PORT  (Press Ctrl+C to stop)"
wait $SERVER_PID
