import sys
from pathlib import Path

# Add Codebase to the path so all internal imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent / "Codebase"))

from dashboard.app import app

if __name__ == "__main__":
    app.run(debug=False, port=8050)
