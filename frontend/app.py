# frontend/app.py
"""
Wrapper entrypoint.

This project uses: frontend/app_streamlit.py
If someone runs `streamlit run frontend/app.py`, we forward to app_streamlit.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "app_streamlit.py"

runpy.run_path(str(TARGET), run_name="__main__")
