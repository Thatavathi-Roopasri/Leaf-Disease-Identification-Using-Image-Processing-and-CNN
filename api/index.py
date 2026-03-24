import os
import sys

# Ensure project root is importable when running as a Vercel serverless function.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app_improved import app
