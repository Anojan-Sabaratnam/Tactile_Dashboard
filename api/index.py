import sys
from pathlib import Path

# Add the Codebase folder to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "Codebase"))

from dashboard.app import app as dash_app

# Vercel Serverless Functions need the Flask app instance named `app`
app = dash_app.server
