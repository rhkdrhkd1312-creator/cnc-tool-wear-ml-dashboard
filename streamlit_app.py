"""Streamlit Cloud entry point.

Deploy settings:
  Main file: streamlit_app.py
  Python: 3.12
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.dashboard  # noqa: F401
