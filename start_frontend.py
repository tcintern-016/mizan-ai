"""
Pakistan Law Assistant — Frontend launcher.
Run from the project root: python start_frontend.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "frontend" / "app.py"

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Pakistan Law Assistant — Frontend")
    print("  Opening at: http://localhost:8501")
    print(f"{'='*60}\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
        cwd=str(ROOT),
    )
