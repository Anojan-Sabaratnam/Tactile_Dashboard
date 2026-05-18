#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Install/upgrade dependencies
echo "Installing dependencies..."
.venv/bin/pip install -q -r Codebase/requirements.txt

echo "Starting AIFI Tactile Dashboard..."
.venv/bin/python run.py &
SERVER_PID=$!

# Wait for server to be ready then open browser
sleep 2
if command -v open &>/dev/null; then
    open http://127.0.0.1:8050        # macOS
elif command -v xdg-open &>/dev/null; then
    xdg-open http://127.0.0.1:8050   # Linux
fi

echo "Dashboard running at http://127.0.0.1:8050  (Press Ctrl+C to stop)"
wait $SERVER_PID
